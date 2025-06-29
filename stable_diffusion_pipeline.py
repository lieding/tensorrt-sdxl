#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cuda import cudart
from diffusers import EulerDiscreteScheduler # Only EulerDiscreteScheduler is directly used now
# Other schedulers might be indirectly available via from_pretrained if get_path resolves to them,
# but not directly instantiated.
import inspect # Used by StableDiffusionPipeline.infer for scheduler.step
from models import (
    UNetXLModel, # UNetXL TRT engine wrapper is still used
)
# Removed: get_path (no longer used here)
# Removed: LoraLoader, make_tokenizer, CLIPModel, CLIPWithProjModel, UNetModel, VAEModel, VAEEncoderModel
# Removed: get_clip_embedding_dim (as CLIPModel TRT wrapper is not used)
# Removed: hashlib, json (related to LoRA/Refit)

import numpy as np # Used in preprocess_images (if kept)
import nvtx # For profiling
import onnx # For UNetXL ONNX export/optimization
import os
import pathlib
import tensorrt as trt
import time
import torch
# from typing import Optional, List # No longer explicitly used in simplified signatures
from utilities import (
    PIPELINE_TYPE,
    TRT_LOGGER,
    Engine, # For UNetXL TRT Engine
    save_image, # Utility to save images
)
# Removed: get_refit_weights, merge_loras, prepare_mask_and_masked_image, unload_model (LoRA/Inpaint/Refit related)

class StableDiffusionPipeline:
    """
    Application showcasing the acceleration of Stable Diffusion pipelines using NVidia TensorRT.
    """
    def __init__(
        self,
        pipeline_type=PIPELINE_TYPE.XL_BASE, # Simplified: Default to SDXL base
        max_batch_size=4, # Max batch size for SDXL, can be adjusted
        denoising_steps=50,
        guidance_scale=7.5,
        device='cuda',
        output_dir='.',
        hf_token=None,
        verbose=False,
        nvtx_profile=False,
        vae_scaling_factor=0.13025, # SDXL specific VAE scaling, from text2_img_sdxl.py
        framework_model_dir='pytorch_model',
        return_latents=False,
    ):
        """
        Initializes the Diffusion pipeline, simplified for SDXL.

        Args:
            pipeline_type (PIPELINE_TYPE):
                Type of current pipeline. Should be XL_BASE.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            denoising_steps (int):
                The number of denoising steps.
            guidance_scale (float):
                Guidance scale.
            device (str):
                PyTorch device to run inference. Default: 'cuda'
            output_dir (str):
                Output directory for log files and image artifacts
            hf_token (str):
                HuggingFace User Access Token.
            verbose (bool):
                Enable verbose logging.
            nvtx_profile (bool):
                Insert NVTX profiling markers.
            vae_scaling_factor (float):
                VAE scaling factor for SDXL.
            framework_model_dir (str):
                Cache directory for framework checkpoints.
            return_latents (bool):
                Skip decoding the image and return latents instead.
        """

        self.denoising_steps = denoising_steps
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = (guidance_scale > 1.0)
        self.vae_scaling_factor = vae_scaling_factor
        # self.version = "xl-1.0" # Removed, SDXL base is assumed. Model paths will be hardcoded.

        self.max_batch_size = max_batch_size

        self.framework_model_dir = framework_model_dir
        self.output_dir = output_dir
        for directory in [self.framework_model_dir, self.output_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        self.hf_token = hf_token
        self.device = device
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile

        # Pipeline type
        self.pipeline_type = pipeline_type # Should always be XL_BASE now
        if not self.pipeline_type.is_sd_xl_base():
            raise ValueError(f"This pipeline is simplified for SDXL Base. Unsupported pipeline_type: {self.pipeline_type.name}")

        # Stages for SDXL Base
        self.stages = ['clip', 'clip2', 'unetxl']
        if not return_latents:
            self.stages.append('vae')
        self.return_latents = return_latents

        # Schedulers are ignored as per user request.
        # Diffusers' schedulers are typically configured and used outside this class's __init__
        # or directly within the inference loop by calling scheduler.set_timesteps and scheduler.step.
        # For simplicity, we'll assume a compatible scheduler is passed or set up elsewhere if needed.
        # self.scheduler will be initialized before inference if required by the chosen flow.
        # Using hardcoded SDXL base model identifier for scheduler
        self.scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        print(f"[I] Using EulerDiscreteScheduler for SDXL.")


        self.config = {}
        # SDXL specific configs
        self.config['vae_torch_fallback'] = True # As VAE is loaded traditionally (PyTorch)
        self.config['clip_hidden_states'] = True


        # initialized in loadEngines() / or assumed pre-loaded for VAE/CLIP
        self.models = {} # For TRT UNetXL
        self.torch_models = {} # For traditionally loaded VAE, CLIPs
        self.engine = {} # For TRT UNetXL
        self.shared_device_memory = None


        # LoRA is removed for simplification. If needed, it can be handled externally.
        self.lora_loader = None
        self.lora_scales = dict()

        # initialized in loadResources()
        self.events = {}
        self.generator = None
        self.markers = {}
        self.seed = None
        self.stream = None
        # Tokenizers will be loaded alongside their respective CLIP models traditionally.
        self.tokenizer = None
        self.tokenizer2 = None

    def loadResources(self, image_height, image_width, batch_size, seed):
        # Initialize noise generator
        if seed:
            self.seed = seed
            self.generator = torch.Generator(device="cuda").manual_seed(seed)

        # Create CUDA events and stream
        # 'vae_encoder' event removed as VAE encoder is not used.
        # Event keys for CLIPs ('clip', 'clip2') are implicitly handled by profile_start/stop if those exact names are used.
        # Let's ensure event creation matches actual profile_start/stop names.
        # encode_prompt uses 'clip_encode'. decode_latent uses 'vae'. denoise_latent uses 'denoise'.
        # For multiple CLIP encoders, it might be better to have distinct events if we want to time them separately in print_summary.
        # The current print_summary tries to access self.events['clip'] and self.events['clip2'].
        # Let's create events for 'clip', 'clip2', 'denoise', 'vae'.
        for stage in ['clip', 'clip2', 'denoise', 'vae']: # Adjusted event names
            self.events[stage] = [cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]]
        self.stream = cudart.cudaStreamCreate()[1]

        # Allocate buffers only for UNetXL TRT engine
        if 'unetxl' in self.engine:
            unetxl_engine = self.engine['unetxl']
            unetxl_model_obj = self.models['unetxl']
            unetxl_engine.allocate_buffers(shape_dict=unetxl_model_obj.get_shape_dict(batch_size, image_height, image_width), device=self.device)
        # VAE and CLIPs are PyTorch models, their memory is managed by PyTorch.

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e[0])
            cudart.cudaEventDestroy(e[1])

        for engine in self.engine.values():
            del engine

        if self.shared_device_memory:
            cudart.cudaFree(self.shared_device_memory)

        cudart.cudaStreamDestroy(self.stream)
        del self.stream

    def cachedModelName(self, model_name):
        # Simplified: inpaint logic removed as inpainting is not supported in this simplified pipeline.
        return model_name

    def getOnnxPath(self, model_name, onnx_dir, opt=True, suffix=''):
        onnx_model_dir = os.path.join(onnx_dir, self.cachedModelName(model_name)+suffix+('.opt' if opt else ''))
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'model.onnx')

    def getEnginePath(self, model_name, engine_dir, enable_refit=False, suffix=''):
        return os.path.join(engine_dir, self.cachedModelName(model_name)+suffix+('.refit' if enable_refit else '')+'.trt'+trt.__version__+'.plan')

    # getWeightsMapPath removed as it was related to refit/LoRA which are removed.
    # getRefitNodesPath removed as it was related to refit/LoRA which are removed.

    def loadEngines(
        self,
        engine_dir,
        framework_model_dir,
        onnx_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        static_batch=False,
        static_shape=True,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.

        Args:
            engine_dir (str):
                Directory to store the TensorRT engines.
            framework_model_dir (str):
                Directory to store the framework model ckpt.
            onnx_dir (str):
                Directory to store the ONNX models.
            onnx_opset (int):
                ONNX opset version to export the models.
            opt_batch_size (int):
                Batch size to optimize for during engine building.
            opt_image_height (int):
                Image height to optimize for during engine building. Must be a multiple of 8.
            opt_image_width (int):
                Image width to optimize for during engine building. Must be a multiple of 8.
            static_batch (bool):
                Build engine only for specified opt_batch_size.
            static_shape (bool):
                Build engine only for specified opt_image_height & opt_image_width. Default = True.
            enable_refit (bool):
                Build engines with refit option enabled.
            enable_all_tactics (bool):
                Enable all tactic sources during TensorRT engine builds.
            timing_cache (str):
                Path to the timing cache to speed up TensorRT build.
        """
        # Create directories if missing
        for directory in [engine_dir, onnx_dir]: # engine_dir and onnx_dir still needed for UNetXL
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        # Tokenizers are loaded traditionally alongside their CLIP models
        print("[I] Assuming Tokenizer, CLIP, and VAE models are loaded traditionally (e.g., HuggingFace Hub).")

        # Load CLIP and VAE models (PyTorch versions)
        # These will be stored in self.torch_models
        # For CLIP, CLIP2, and VAE, we'll use HuggingFace's from_pretrained.
        # Their ONNX/TRT conversion is assumed to be handled externally if needed.

        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        from diffusers import AutoencoderKL

        # CLIPTextModel (text_encoder) for SDXL Base
        if 'clip' in self.stages:
            print(f"[I] Loading CLIPTextModel (text_encoder) from HuggingFace Hub for SDXL (subfolder: text_encoder)...")
            # Using hardcoded SDXL base model identifier
            sdxl_base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            self.tokenizer = CLIPTokenizer.from_pretrained(sdxl_base_model_id, subfolder="tokenizer", token=self.hf_token)
            self.torch_models['clip'] = CLIPTextModel.from_pretrained(
                sdxl_base_model_id,
                subfolder="text_encoder",
                torch_dtype=torch.float16, # SDXL typically uses fp16
                use_safetensors=True,
                token=self.hf_token
            ).to(self.device)
            print("[I] CLIPTextModel loaded.")

        # CLIPTextModelWithProjection (text_encoder_2) for SDXL Base and Refiner
        if 'clip2' in self.stages:
            print(f"[I] Loading CLIPTextModelWithProjection (text_encoder_2) from HuggingFace Hub for SDXL (subfolder: text_encoder_2)...")
            # Using hardcoded SDXL base model identifier
            sdxl_base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            self.tokenizer2 = CLIPTokenizer.from_pretrained(sdxl_base_model_id, subfolder="tokenizer_2", token=self.hf_token)
            self.torch_models['clip2'] = CLIPTextModelWithProjection.from_pretrained(
                sdxl_base_model_id,
                subfolder="text_encoder_2",
                torch_dtype=torch.float16, # SDXL typically uses fp16
                use_safetensors=True,
                token=self.hf_token
            ).to(self.device)
            print("[I] CLIPTextModelWithProjection loaded.")

        # VAE
        if 'vae' in self.stages:
            print(f"[I] Loading AutoencoderKL (VAE) from HuggingFace Hub for SDXL...")
            # Using hardcoded SDXL base model identifier
            sdxl_base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            self.torch_models['vae'] = AutoencoderKL.from_pretrained(
                sdxl_base_model_id, # VAE is often part of the main model path for SDXL
                subfolder="vae",
                torch_dtype=torch.float16, # VAE can also be fp16 for SDXL
                use_safetensors=True,
                token=self.hf_token
            ).to(self.device)
            print("[I] AutoencoderKL (VAE) loaded.")


        # Load UNetXL TensorRT Engine
        models_args = {'version': self.version, 'pipeline': self.pipeline_type, 'device': self.device,
            'hf_token': self.hf_token, 'verbose': self.verbose, 'framework_model_dir': framework_model_dir,
            'max_batch_size': self.max_batch_size}

        if 'unetxl' in self.stages:
            # Note: LoRA related params (lora_scales, lora_dict, lora_alphas) are removed for simplification
            self.models['unetxl'] = UNetXLModel(**models_args, fp16=True, # SDXL UNet is typically fp16
                do_classifier_free_guidance=self.do_classifier_free_guidance)
        else:
            raise ValueError("UNetXL stage is required for SDXL pipeline.")


        # Configure UNetXL model for TensorRT engine loading/building
        model_name = 'unetxl' # Only processing unetxl here
        obj = self.models[model_name]

        # LoRA suffix and related logic removed
        onnx_unetxl_path = self.getOnnxPath(model_name, onnx_dir, opt=False)
        onnx_opt_unetxl_path = self.getOnnxPath(model_name, onnx_dir) # opt=True by default
        # Refit is disabled for simplification as LoRA is removed.
        engine_unetxl_path = self.getEnginePath(model_name, engine_dir, enable_refit=False)


        # Export UNetXL ONNX model if it doesn't exist
        if not os.path.exists(engine_unetxl_path) and not os.path.exists(onnx_opt_unetxl_path):
            if not os.path.exists(onnx_unetxl_path):
                 print(f"Exporting ONNX model for {model_name}: {onnx_unetxl_path}")
                 # enable_lora_merge is False as LoRA is removed
                 obj.export_onnx(onnx_unetxl_path, onnx_opt_unetxl_path, onnx_opset, opt_image_height, opt_image_width, enable_lora_merge=False)
            else:
                print(f"[I] Found cached ONNX model for {model_name}: {onnx_unetxl_path}")

            # Optimize UNetXL ONNX model if the optimized version doesn't exist
            if not os.path.exists(onnx_opt_unetxl_path): # This check might be redundant if export_onnx handles it
                print(f"Optimizing ONNX model for {model_name}: {onnx_opt_unetxl_path}")
                onnx_opt_graph = obj.optimize(onnx.load(onnx_unetxl_path))
                if onnx_opt_graph.ByteSize() > 2147483648: # 2GB limit for single file
                    onnx.save_model(
                        onnx_opt_graph,
                        onnx_opt_unetxl_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        convert_attribute=False)
                else:
                    onnx.save(onnx_opt_graph, onnx_opt_unetxl_path)
        elif not os.path.exists(onnx_opt_unetxl_path) and os.path.exists(onnx_unetxl_path):
            # This case handles if raw ONNX exists but optimized doesn't
            print(f"Optimizing existing ONNX model for {model_name}: {onnx_opt_unetxl_path}")
            onnx_opt_graph = obj.optimize(onnx.load(onnx_unetxl_path))
            if onnx_opt_graph.ByteSize() > 2147483648:
                onnx.save_model(
                    onnx_opt_graph,
                    onnx_opt_unetxl_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False)
            else:
                onnx.save(onnx_opt_graph, onnx_opt_unetxl_path)
        else:
            print(f"[I] Found cached optimized ONNX model for {model_name}: {onnx_opt_unetxl_path}")


        # Build TensorRT engine for UNetXL
        engine = Engine(engine_unetxl_path)
        if not os.path.exists(engine_unetxl_path):
            update_output_names = obj.get_output_names() + obj.extra_output_names if obj.extra_output_names else None
            # For SDXL UNet, fp16 is typical. tf32 might not be used unless specific layers benefit.
            engine.build(onnx_opt_unetxl_path,
                fp16=True, # UNetXL is fp16
                tf32=False, # Typically False for UNet
                input_profile=obj.get_input_profile(
                    opt_batch_size, opt_image_height, opt_image_width,
                    static_batch=static_batch, static_shape=static_shape
                ),
                enable_refit=False, # Refit disabled due to LoRA removal
                enable_all_tactics=enable_all_tactics,
                timing_cache=timing_cache,
                update_output_names=update_output_names)
        self.engine[model_name] = engine

        # Load TensorRT engine for UNetXL
        self.engine[model_name].load()
        # Refit logic removed as LoRA and enable_refit are removed/disabled.

        # Torch models (CLIPs, VAE) are already loaded into self.torch_models earlier.
        # No need to iterate self.models for torch model loading.

    def calculateMaxDeviceMemory(self):
        max_device_memory = 0
        for model_name, engine in self.engine.items():
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def activateEngines(self, shared_device_memory=None):
        if shared_device_memory is None:
            max_device_memory = self.calculateMaxDeviceMemory()
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.shared_device_memory = shared_device_memory
        # Load and activate TensorRT engine for UNetXL
        if 'unetxl' in self.engine:
            self.engine['unetxl'].activate(reuse_device_memory=self.shared_device_memory)
        # VAE and CLIPs are PyTorch models, no TRT engine activation needed.

    def runEngine(self, model_name, feed_dict):
        # This method is now only used for UNetXL
        if model_name != 'unetxl' or model_name not in self.engine:
            raise ValueError(f"runEngine is intended for 'unetxl' TRT model, but called with {model_name}")
        engine = self.engine[model_name]
        # use_cuda_graph parameter removed, hardcode to False for simplicity
        return engine.infer(feed_dict, self.stream, use_cuda_graph=False)

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width):
        # SDXL uses float32 for latents initially
        latents_dtype = torch.float32
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=self.generator)
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def profile_start(self, name, color='blue'):
        if self.nvtx_profile:
            self.markers[name] = nvtx.start_range(message=name, color=color)
        if name in self.events:
            cudart.cudaEventRecord(self.events[name][0], 0)

    def profile_stop(self, name):
        if name in self.events:
            cudart.cudaEventRecord(self.events[name][1], 0)
        if self.nvtx_profile:
            nvtx.end_range(self.markers[name])

    # preprocess_images removed as it's not used in the simplified SDXL base txt2img flow.
    # Corresponding profiler calls for 'preprocess' are also removed.

    # preprocess_controlnet_images removed as ControlNet support is removed.

    def encode_prompt(self, prompt, negative_prompt, encoder_model, tokenizer, profiler_event_name: str, pooled_outputs=False, output_hidden_states=False):
        # This method now directly uses the passed PyTorch encoder_model and tokenizer
        self.profile_start(profiler_event_name, color='green')

        # Tokenize prompt (positive and negative)
        text_input_ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        uncond_input_ids = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Get embeddings from the PyTorch model
        # Positive prompt
        prompt_outputs = encoder_model(
            text_input_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        text_embeddings = prompt_outputs.last_hidden_state if output_hidden_states else prompt_outputs.text_embeds
        pooled_prompt_embeds = prompt_outputs.pooler_output if hasattr(prompt_outputs, 'pooler_output') and prompt_outputs.pooler_output is not None else prompt_outputs.text_embeds # Fallback for models without distinct pooler_output like CLIPTextModel

        # Negative prompt
        uncond_outputs = encoder_model(
            uncond_input_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        uncond_embeddings = uncond_outputs.last_hidden_state if output_hidden_states else uncond_outputs.text_embeds
        pooled_uncond_embeds = uncond_outputs.pooler_output if hasattr(uncond_outputs, 'pooler_output') and uncond_outputs.pooler_output is not None else uncond_outputs.text_embeds


        # Classifier-Free Guidance
        if self.do_classifier_free_guidance:
            if output_hidden_states: # Concatenate hidden states
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            else: # Concatenate final embeddings (e.g., text_embeds from CLIPTextModelWithProjection)
                 # For CLIPTextModel (encoder 1), text_embeds is (batch_size, seq_len, hidden_size)
                 # For CLIPTextModelWithProjection (encoder 2), text_embeds is (batch_size, proj_dim)
                 # The handling here assumes that if output_hidden_states is False, we're dealing with the already projected embeddings for CLIP2
                 # or the equivalent final layer output for CLIP1 that doesn't need further selection.
                if text_embeddings.ndim == 3 and uncond_embeddings.ndim == 3: # e.g. CLIP L last_hidden_state
                    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                elif text_embeddings.ndim == 2 and uncond_embeddings.ndim == 2: # e.g. CLIP G text_embeds (projected)
                    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                else:
                    raise ValueError(f"Mismatched embedding dimensions for concatenation: uncond {uncond_embeddings.shape}, text {text_embeddings.shape}")


            if pooled_outputs: # Concatenate pooled embeddings
                pooled_output = torch.cat([pooled_uncond_embeds, pooled_prompt_embeds])
        else: # No CFG
            if pooled_outputs:
                pooled_output = pooled_prompt_embeds
            # text_embeddings remains positive prompt's embeddings

        # Ensure correct dtype (SDXL often uses float16 for embeddings)
        text_embeddings = text_embeddings.to(dtype=torch.float16)
        if pooled_outputs:
            pooled_output = pooled_output.to(dtype=torch.float16)

        self.profile_stop(profiler_event_name)

        if pooled_outputs:
            return text_embeddings, pooled_output
        return text_embeddings, None # Return None for pooled if not requested to maintain tuple structure

    # from diffusers (get_timesteps) - Simplified for txt2img only
    def get_timesteps(self, num_inference_steps):
        """
        Sets the scheduler's timesteps. For txt2img, strength and denoising_start are not used.
        """
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        return self.scheduler.timesteps, num_inference_steps

    def denoise_latent(self,
        latents,
        text_embeddings,
        denoiser='unet',
        timesteps=None,
        step_offset=0,
        mask=None,
        masked_image_latents=None, # Keep for potential inpainting-like SDXL variants, though main path simplified
        # image_guidance removed as it's mostly for img2img/ControlNet guidance strength
        # controlnet_imgs, controlnet_scales removed
        text_embeds=None, # This is the pooled CLIP G output for SDXL
        time_ids=None): # This is the time embedding for SDXL

        # controlnet_imgs preprocessing removed.
        # assert for image_guidance removed.

        self.profile_start('denoise', color='blue')
        for step_index, timestep in enumerate(timesteps): # timesteps is now passed directly
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # Predict the noise residual
            # self.torch_inference logic removed, UNetXL is always TRT.
            timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

            params = {"sample": latent_model_input, "timestep": timestep_float, "encoder_hidden_states": text_embeddings}
            # controlnet_imgs parameter removed from params
            if text_embeds is not None: # For SDXL
                params.update({'text_embeds': text_embeds})
            if time_ids is not None: # For SDXL
                params.update({'time_ids': time_ids})

            if denoiser != 'unetxl':
                 raise ValueError(f"Simplified pipeline expects 'unetxl' denoiser, got {denoiser}")
            noise_pred = self.runEngine(denoiser, params)['latent']

            # Perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # from diffusers (prepare_extra_step_kwargs)
            extra_step_kwargs = {}
            if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
                # TODO: configurable eta
                eta = 0.0
                extra_step_kwargs["eta"] = eta
            if "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()):
                extra_step_kwargs["generator"] = self.generator

            latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]

        latents = 1. / self.vae_scaling_factor * latents

        self.profile_stop('denoise')
        return latents

    def encode_image(self, input_image):
        # VAE Encoder is not part of the simplified SDXL txt2img/refiner flow directly in this class.
        # If img2img capabilities for SDXL were to be added back, this would need to be revisited.
        # For now, removing it.
        raise NotImplementedError("VAEEncoderModel (encode_image) is not supported in this simplified SDXL pipeline.")

    def decode_latent(self, latents):
        # VAE decoding will use the PyTorch model from self.torch_models['vae']
        self.profile_start('vae', color='red') # Use 'vae' to match event key in loadResources

        # Ensure VAE is loaded if this stage is reached
        if 'vae' not in self.torch_models:
            raise RuntimeError("VAE model not loaded into self.torch_models but decode_latent was called.")

        vae_model = self.torch_models['vae']
        # SDXL VAE may benefit from float32 for precision, or use fp16 if loaded that way
        # Original code had a float32 conversion for torch_fallback, let's be explicit.
        original_dtype = latents.dtype
        if vae_model.dtype == torch.float32 and latents.dtype != torch.float32:
            latents = latents.to(dtype=torch.float32)

        images = vae_model.decode(latents).sample

        # If latents were upcast, consider if output needs to match original (usually not for images)
        # images = images.to(original_dtype) # This might not be desired if original was fp16 and VAE prefers fp32 for decode

        self.profile_stop('vae_decode')
        return images

    def print_summary(self, denoising_steps, walltime_ms, batch_size):
        print('|-----------------|--------------|')
        print('| {:^15} | {:^12} |'.format('Module', 'Latency'))
        print('|-----------------|--------------|')
        # VAE Encoder part removed
        # SDXL Base always uses two encoders ('clip' and 'clip2')
        if 'clip' in self.events and 'clip2' in self.events:
             clip_time = cudart.cudaEventElapsedTime(self.events['clip'][0], self.events['clip'][1])[1]
             clip2_time = cudart.cudaEventElapsedTime(self.events['clip2'][0], self.events['clip2'][1])[1]
             print('| {:^15} | {:>9.2f} ms |'.format('CLIPs Enc', clip_time + clip2_time ))
        elif 'clip' in self.events: # Should not happen if clip2 is always there for base
            clip_time = cudart.cudaEventElapsedTime(self.events['clip'][0], self.events['clip'][1])[1]
            print('| {:^15} | {:>9.2f} ms |'.format('CLIP Enc', clip_time))
        elif 'clip2' in self.events: # Should not happen if clip is always there for base
            clip2_time = cudart.cudaEventElapsedTime(self.events['clip2'][0], self.events['clip2'][1])[1]
            print('| {:^15} | {:>9.2f} ms |'.format('CLIP2 Enc', clip2_time))


        # ControlNet suffix removed from UNet display name
        print('| {:^15} | {:>9.2f} ms |'.format('UNetXL x '+str(denoising_steps), cudart.cudaEventElapsedTime(self.events['denoise'][0], self.events['denoise'][1])[1]))
        if 'vae' in self.events and not self.return_latents: # Check if VAE event exists and we are not returning latents
            print('| {:^15} | {:>9.2f} ms |'.format('VAE-Dec', cudart.cudaEventElapsedTime(self.events['vae'][0], self.events['vae'][1])[1]))
        print('|-----------------|--------------|')
        print('| {:^15} | {:>9.2f} ms |'.format('Pipeline', walltime_ms))
        print('|-----------------|--------------|')
        print('Throughput: {:.2f} image/s'.format(batch_size*1000./walltime_ms))

    def save_image(self, images, pipeline, prompt):
        # Save image
        image_name_prefix = pipeline+'-fp16'+''.join(set(['-'+prompt[i].replace(' ','_')[:10] for i in range(len(prompt))]))+'-'
        save_image(images, self.output_dir, image_name_prefix)

    def infer(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        input_image=None,
        image_strength=0.75,
        mask_image=None,
        controlnet_scales=None,
        aesthetic_score=6.0,
        negative_aesthetic_score=2.5,
        warmup=False,
        verbose=False,
        save_image=True,
    ):
        """
        Run the SDXL Base diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            warmup (bool):
                Indicate if this is a warmup run.
            save_image (bool):
                Save the generated image (if applicable)
            input_image (torch.Tensor, optional):
                For SDXL Refiner, these are the latents from the base model.
            image_strength (float, optional):
                For SDXL Refiner, strength of transformation.
            aesthetic_score (float, optional):
                 For SDXL Refiner, aesthetic score.
            negative_aesthetic_score (float, optional):
                 For SDXL Refiner, negative aesthetic score.
        """
        assert len(prompt) == len(negative_prompt)
        batch_size = len(prompt)

        # Spatial dimensions of latent tensor
        latent_height = image_height // 8
        latent_width = image_width // 8

        if self.generator and self.seed: # Ensure generator exists and seed is set
            self.generator.manual_seed(self.seed)

        current_denoising_steps = self.denoising_steps # Store for summary

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER): # Removed redundant trt.Runtime for VAE decode
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # current_denoising_steps will be self.denoising_steps due to simplification of get_timesteps
            timesteps, current_denoising_steps = self.get_timesteps(self.denoising_steps)
            denoise_kwargs = {'timesteps': timesteps}

            # Latent Initialization for SDXL Base
            if not self.pipeline_type.is_sd_xl_base(): # Should always be true due to __init__ check
                raise NotImplementedError(f"Pipeline type {self.pipeline_type} not supported. Only SDXL Base.")

            latents = self.initialize_latents(batch_size=batch_size,
                unet_channels=4, # Standard for SD
                latent_height=latent_height,
                latent_width=latent_width)

            # CLIP text encoder(s) - SDXL Base specific
            # Using torch_models for CLIP as they are loaded via HuggingFace
            if 'clip' not in self.torch_models or 'clip2' not in self.torch_models:
                raise RuntimeError("CLIP models ('clip' and 'clip2') not loaded in self.torch_models.")

            text_embeddings_clipL, _ = self.encode_prompt(
                prompt, negative_prompt,
                encoder_model=self.torch_models['clip'],
                tokenizer=self.tokenizer,
                profiler_event_name='clip', # For CLIP L
                output_hidden_states=True)

            text_embeddings_clipG, pooled_embeddings_clipG = self.encode_prompt(
                prompt, negative_prompt,
                encoder_model=self.torch_models['clip2'],
                tokenizer=self.tokenizer2,
                profiler_event_name='clip2', # For CLIP G
                pooled_outputs=True,
                output_hidden_states=True)

            text_embeddings = torch.cat([text_embeddings_clipL, text_embeddings_clipG], dim=-1)

            # Time embeddings for SDXL Base
            original_size = (image_height, image_width)
            crops_coords_top_left = (0, 0)
            target_size = (image_height, image_width)

            add_time_ids = _get_add_time_ids( # Calling simplified _get_add_time_ids
                original_size, crops_coords_top_left, target_size,
                dtype=text_embeddings.dtype, # pooled_embeddings_clipG.dtype can also be used
                device=self.device,
                do_classifier_free_guidance=self.do_classifier_free_guidance
            )

            add_time_ids = add_time_ids.repeat(batch_size, 1)
            # denoise_kwargs already contains timesteps
            denoise_kwargs.update({'text_embeds': pooled_embeddings_clipG, 'time_ids': add_time_ids})

            # UNet denoiser (UNetXL TRT engine)
            latents = self.denoise_latent(latents, text_embeddings, denoiser='unetxl', **denoise_kwargs)

            # VAE decode latent (if applicable) using torch_models['vae']
            if self.return_latents:
                images = latents * self.vae_scaling_factor # Return scaled latents
            else:
                if 'vae' not in self.torch_models:
                    raise RuntimeError("VAE model not loaded in self.torch_models, but VAE decoding is requested.")
                images = self.decode_latent(latents) # decode_latent will use self.torch_models['vae']

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

        walltime_ms = (e2e_toc - e2e_tic) * 1000.
        if not warmup:
            self.print_summary(current_denoising_steps, walltime_ms, batch_size)
            if not self.return_latents and save_image:
                self.save_image(images, self.pipeline_type.name.lower(), prompt)

        return (images, walltime_ms) # Return processed images (or latents) and walltime

# Helper function for SDXL Base time IDs
def _get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype, device, do_classifier_free_guidance):
    # Logic for XL_BASE
    add_time_ids_list = list(original_size + crops_coords_top_left + target_size)
    if do_classifier_free_guidance:
        add_neg_time_ids_list = list(original_size + crops_coords_top_left + target_size)
        # Create tensor for negative prompts and then for positive prompts
        add_neg_time_ids = torch.tensor([add_neg_time_ids_list], dtype=dtype, device=device)
        add_pos_time_ids = torch.tensor([add_time_ids_list], dtype=dtype, device=device)
        add_time_ids = torch.cat([add_neg_time_ids, add_pos_time_ids], dim=0)
    else:
        add_time_ids = torch.tensor([add_time_ids_list], dtype=dtype, device=device)

    return add_time_ids


    def run(self, prompt, negative_prompt, height, width, batch_size, batch_count, num_warmup_runs, **kwargs): # use_cuda_graph removed
        # Process prompt
        if not isinstance(prompt, list):
            raise ValueError(f"`prompt` must be of type `str` list, but is {type(prompt)}")
        prompt = prompt * batch_size

        if not isinstance(negative_prompt, list):
            raise ValueError(f"`--negative-prompt` must be of type `str` list, but is {type(negative_prompt)}")
        if len(negative_prompt) == 1:
            negative_prompt = negative_prompt * batch_size

        # num_warmup_runs logic simplified as use_cuda_graph is removed.
        # CUDA graph specific warmup (max(1, num_warmup_runs)) is no longer needed.
        if num_warmup_runs > 0:
            print("[I] Warming up ..")
            for _ in range(num_warmup_runs):
                self.infer(prompt, negative_prompt, height, width, warmup=True, **kwargs)

        for _ in range(batch_count):
            print("[I] Running StableDiffusionXL pipeline") # Changed message slightly
            if self.nvtx_profile:
                cudart.cudaProfilerStart()
            self.infer(prompt, negative_prompt, height, width, warmup=False, **kwargs)
            if self.nvtx_profile:
                cudart.cudaProfilerStop()