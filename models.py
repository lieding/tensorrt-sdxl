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

from diffusers.loaders import LoraLoaderMixin
from diffusers.models import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel
)
from diffusers.utils import convert_state_dict_to_diffusers
import json
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
import onnx_graphsurgeon as gs
import os
from polygraphy.backend.onnx.loader import fold_constants
import tempfile
import torch
import torch.nn.functional as F
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer
)

class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, 'model.onnx')
            onnx_inferred_path = os.path.join(temp_dir, 'inferred.onnx')
            onnx.save_model(onnx_graph,
                onnx_orig_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False)
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def clip_add_hidden_states(self, return_onnx=False):
        hidden_layers = -1
        onnx_graph = gs.export_onnx(self.graph)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                name = onnx_graph.graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                if onnx_graph.graph.node[i].output[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers-1):
                    onnx_graph.graph.node[i].output[j] = "hidden_states"
            for j in range(len(onnx_graph.graph.node[i].input)):
                if onnx_graph.graph.node[i].input[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers-1):
                    onnx_graph.graph.node[i].input[j] = "hidden_states"
        if return_onnx:
            return onnx_graph

# get_path function removed as model paths are now hardcoded for SDXL base.
# get_clip_embedding_dim function removed.
# get_clipwithproj_embedding_dim function removed.
# get_unet_embedding_dim function removed.

# FIXME after serialization support for torch.compile is added
def get_checkpoint_dir(framework_model_dir, subfolder):
    return os.path.join(framework_model_dir, subfolder)

torch_inference_modes = ['default', 'reduce-overhead', 'max-autotune']
# FIXME update callsites after serialization support for torch.compile is added
def optimize_checkpoint(model, torch_inference):
    if not torch_inference or torch_inference == 'eager':
        return model
    assert torch_inference in torch_inference_modes
    return torch.compile(model, mode=torch_inference, dynamic=False, fullgraph=False)

class LoraLoader(LoraLoaderMixin):
    def __init__(self,
        paths,
    ):
        self.paths = paths
        self.state_dict = dict()
        self.network_alphas = dict()

        for path in paths:
            state_dict, network_alphas = self.lora_state_dict(path)
            is_correct_format = all("lora" in key for key in state_dict.keys())
            if not is_correct_format:
                raise ValueError("Invalid LoRA checkpoint.")

            self.state_dict[path] = state_dict
            self.network_alphas[path] = network_alphas

    def get_dicts(self,
        prefix='unet',
        convert_to_diffusers=False,
    ):
        state_dict = dict()
        network_alphas = dict()

        for path in self.paths:
            keys = list(self.state_dict[path].keys())
            if all(key.startswith(('unet', 'text_encoder')) for key in keys):
                keys = [k for k in keys if k.startswith(prefix)]
                if keys:
                    print(f"Processing {prefix} LoRA: {path}")
                state_dict[path] = {k.replace(f"{prefix}.", ""): v for k, v in self.state_dict[path].items() if k in keys}

                if path in self.network_alphas:
                    alpha_keys = [k for k in self.network_alphas[path].keys() if k.startswith(prefix)]
                    network_alphas[path] = {
                        k.replace(f"{prefix}.", ""): v for k, v in self.network_alphas[path].items() if k in alpha_keys
                    }

            else:
                # Otherwise, we're dealing with the old format.
                warn_message = "You have saved the LoRA weights using the old format. To convert LoRA weights to the new format, first load them in a dictionary and then create a new dictionary as follows: `new_state_dict = {f'unet.{module_name}': params for module_name, params in old_state_dict.items()}`."
                print(warn_message)

        return state_dict, network_alphas


class BaseModel():
    def __init__(self,
        pipeline=None, # pipeline is PIPELINE_TYPE enum instance
        device='cuda',
        hf_token='',
        verbose=True,
        framework_model_dir='pytorch_model',
        fp16=False,
        max_batch_size=16,
        text_maxlen=77,
        embedding_dim=768, # Default, overridden by UNetXLModel
    ):
        """
        Base class for ONNX / TRT model wrapper classes in this demo.

        Args:
            pipeline (PIPELINE_TYPE): Enum representing the pipeline type (e.g., XL_BASE).
            device (str): PyTorch device.
            hf_token (str): HuggingFace API token.
            verbose (bool): Enable verbose logging.
            framework_model_dir (str): Directory for HF saved models.
            fp16 (bool): Use FP16 precision.
            max_batch_size (int): Max batch size for the model.
            text_maxlen (int): Max length for text inputs.
            embedding_dim (int): Embedding dimension for text models.
        """
        self.name = self.__class__.__name__
        self.pipeline = pipeline.name # pipeline is PIPELINE_TYPE enum instance
        self.sdxl_model_id = "stabilityai/stable-diffusion-xl-base-1.0" # Hardcoded for SDXL Base

        self.device = device
        self.hf_token = hf_token
        # self.hf_safetensor = not (pipeline.is_inpaint() and version in ("1.4", "1.5")) # Simplified, assume True for SDXL
        self.hf_safetensor = True
        self.verbose = verbose
        self.framework_model_dir = framework_model_dir

        self.fp16 = fp16

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256   # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.text_maxlen = text_maxlen
        self.embedding_dim = embedding_dim
        self.extra_output_names = []

        self.lora_dict = None

    def get_model(self, torch_inference=''):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.cleanup()
        opt.info(self.name + ': cleanup')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ': finished')
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width)


class CLIPModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size,
        embedding_dim,
        output_hidden_states=False,
        subfolder="text_encoder",
        lora_dict=None,
        lora_alphas=None,
    ):
        super(CLIPModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.subfolder = subfolder

        # Output the final hidden state
        if output_hidden_states:
            self.extra_output_names = ['hidden_states']

    def get_model(self, torch_inference=''):
        clip_model_dir = get_checkpoint_dir(self.framework_model_dir, self.subfolder)
        model = CLIPTextModel.from_pretrained(clip_model_dir).to(self.device)
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
       return ['text_embeddings']

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: 'B'},
            'text_embeddings': {0: 'B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim)
        }
        if 'hidden_states' in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)
        return output

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.select_outputs([0]) # delete graph output#1
        opt.cleanup()
        opt.info(self.name + ': remove output[1]')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        opt.select_outputs([0], names=['text_embeddings']) # rename network output
        opt.info(self.name + ': remove output[0]')
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        if 'hidden_states' in self.extra_output_names:
            opt_onnx_graph = opt.clip_add_hidden_states(return_onnx=True)
            opt.info(self.name + ': added hidden_states')
        opt.info(self.name + ': finished')
        return opt_onnx_graph


class CLIPWithProjModel(CLIPModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=16,
        output_hidden_states=False,
        subfolder="text_encoder_2",
        lora_dict=None,
        lora_alphas=None,
    ):

        super(CLIPWithProjModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, max_batch_size=max_batch_size, embedding_dim=get_clipwithproj_embedding_dim(version, pipeline), output_hidden_states=output_hidden_states)
        self.subfolder = subfolder

    def get_model(self, torch_inference=''):
        clip_model_dir = get_checkpoint_dir(self.framework_model_dir, self.subfolder)
        if not os.path.exists(clip_model_dir):
            model = CLIPTextModelWithProjection.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                use_auth_token=self.hf_token).to(self.device)
            model.save_pretrained(clip_model_dir)
        else:
            print(f"[I] Load CLIP pytorch model from: {clip_model_dir}")
            model = CLIPTextModelWithProjection.from_pretrained(clip_model_dir).to(self.device)
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.embedding_dim)
        }
        if 'hidden_states' in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)

        return output


class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet, controlnets) -> None:
        super().__init__()
        self.unet = unet
        self.controlnets = controlnets
        
    def forward(self, sample, timestep, encoder_hidden_states, images, controlnet_scales):
        for i, (image, conditioning_scale, controlnet) in enumerate(zip(images, controlnet_scales, self.controlnets)):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                return_dict=False,
            )

            down_samples = [
                    down_sample * conditioning_scale
                    for down_sample in down_samples
                ]
            mid_sample *= conditioning_scale
            
            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample
        
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        )
        return noise_pred


class UNetModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16 = False,
        max_batch_size = 16,
        text_maxlen = 77,
        controlnets = None,
        lora_scales = None,
        lora_dict = None,
        lora_alphas = None,
        do_classifier_free_guidance = False,
    ):

        super(UNetModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, max_batch_size=max_batch_size, text_maxlen=text_maxlen, embedding_dim=get_unet_embedding_dim(version, pipeline))
        self.subfolder = 'unet'
        self.controlnets = get_path(version, pipeline, controlnets) if controlnets else None
        self.unet_dim = (9 if pipeline.is_inpaint() else 4)
        self.lora_scales = lora_scales
        self.lora_dict = lora_dict
        self.lora_alphas = lora_alphas
        self.xB = 2 if do_classifier_free_guidance else 1 # batch multiplier

    def get_model(self, torch_inference=''):
        model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        if self.controlnets:
            unet_model = UNet2DConditionModel.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                use_auth_token=self.hf_token,
                **model_opts).to(self.device)
            cnet_model_opts = {'torch_dtype': torch.float16} if self.fp16 else {}
            controlnets = torch.nn.ModuleList([ControlNetModel.from_pretrained(path, **cnet_model_opts).to(self.device) for path in self.controlnets])
            # FIXME - cache UNet2DConditionControlNetModel
            model = UNet2DConditionControlNetModel(unet_model, controlnets)
        else:
            unet_model_dir = get_checkpoint_dir(self.framework_model_dir,self.subfolder)
            if not os.path.exists(unet_model_dir):
                model = UNet2DConditionModel.from_pretrained(self.path,
                    subfolder=self.subfolder,
                    use_safetensors=self.hf_safetensor,
                    use_auth_token=self.hf_token,
                    **model_opts).to(self.device)
                model.save_pretrained(unet_model_dir)
            else:
                print(f"[I] Load UNet pytorch model from: {unet_model_dir}")
                model = UNet2DConditionModel.from_pretrained(unet_model_dir).to(self.device)
            if torch_inference:
                model.to(memory_format=torch.channels_last)
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        if self.controlnets is None:
            return ['sample', 'timestep', 'encoder_hidden_states']
        else:    
            return ['sample', 'timestep', 'encoder_hidden_states', 'images', 'controlnet_scales']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        xB = '2B' if self.xB == 2 else 'B'
        if self.controlnets is None:
            return {
                'sample': {0: xB, 2: 'H', 3: 'W'},
                'encoder_hidden_states': {0: xB},
                'latent': {0: xB, 2: 'H', 3: 'W'}
            }
        else:
            return {
                'sample': {0: xB, 2: 'H', 3: 'W'},
                'encoder_hidden_states': {0: xB},
                'images': {1: xB, 3: '8H', 4: '8W'},
                'latent': {0: xB, 2: 'H', 3: 'W'}
            }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        if self.controlnets is None:
            return {
                'sample': [(self.xB*min_batch, self.unet_dim, min_latent_height, min_latent_width), (self.xB*batch_size, self.unet_dim, latent_height, latent_width), (self.xB*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
                'encoder_hidden_states': [(self.xB*min_batch, self.text_maxlen, self.embedding_dim), (self.xB*batch_size, self.text_maxlen, self.embedding_dim), (self.xB*max_batch, self.text_maxlen, self.embedding_dim)]
            }
        else:
            return {
                'sample': [(self.xB*min_batch, self.unet_dim, min_latent_height, min_latent_width),
                           (self.xB*batch_size, self.unet_dim, latent_height, latent_width),
                           (self.xB*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
                'encoder_hidden_states': [(self.xB*min_batch, self.text_maxlen, self.embedding_dim),
                                          (self.xB*batch_size, self.text_maxlen, self.embedding_dim),
                                          (self.xB*max_batch, self.text_maxlen, self.embedding_dim)],
                'images': [(len(self.controlnets), self.xB*min_batch, 3, min_image_height, min_image_width),
                          (len(self.controlnets), self.xB*batch_size, 3, image_height, image_width),
                          (len(self.controlnets), self.xB*max_batch, 3, max_image_height, max_image_width)]
            }


    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        if self.controlnets is None:
            return {
                'sample': (self.xB*batch_size, self.unet_dim, latent_height, latent_width),
                'encoder_hidden_states': (self.xB*batch_size, self.text_maxlen, self.embedding_dim),
                'latent': (self.xB*batch_size, 4, latent_height, latent_width)
            }
        else:
            return {
                'sample': (self.xB*batch_size, self.unet_dim, latent_height, latent_width),
                'encoder_hidden_states': (self.xB*batch_size, self.text_maxlen, self.embedding_dim),
                'images': (len(self.controlnets), self.xB*batch_size, 3, image_height, image_width),
                'latent': (self.xB*batch_size, 4, latent_height, latent_width)
                }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        if self.controlnets is None:
            return (
                torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
                torch.tensor([1.], dtype=torch.float32, device=self.device),
                torch.randn(batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device)
            )
        else:
            return (
                torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
                torch.tensor(999, dtype=torch.float32, device=self.device),
                torch.randn(batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
                torch.randn(len(self.controlnets), batch_size, 3, image_height, image_width, dtype=dtype, device=self.device),
                torch.randn(len(self.controlnets), dtype=dtype, device=self.device)
            )


class UNetXLModel(BaseModel):
    def __init__(self,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        fp16 = False,
        max_batch_size = 16,
        text_maxlen = 77,
        lora_scales = None,
        lora_dict = None,
        lora_alphas = None,
        do_classifier_free_guidance = False,
    ):
        # Version parameter removed from super call.
        # pipeline is PIPELINE_TYPE.XL_BASE
        # embedding_dim for SDXL UNet is 2048.
        super(UNetXLModel, self).__init__(pipeline=pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, fp16=fp16, max_batch_size=max_batch_size, text_maxlen=text_maxlen, embedding_dim=2048)
        self.subfolder = 'unet'
        self.unet_dim = 4 # For SDXL Base (not inpaint)
        self.time_dim = 6 # For SDXL Base (not refiner)
        self.lora_scales = lora_scales
        self.lora_dict = lora_dict
        self.lora_alphas = lora_alphas
        self.xB = 2 if do_classifier_free_guidance else 1 # batch multiplier

    def get_model(self, torch_inference=''):
        model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        # Use "xl-1.0" as a fixed version string for checkpoint directory naming.
        # self.pipeline is already the string name e.g. "XL_BASE"
        unet_model_dir = get_checkpoint_dir(self.framework_model_dir, self.subfolder)

        print(f"[I] Load UNet pytorch model from: {unet_model_dir}")
        model_load_opts = {'torch_dtype': torch.float16} if self.fp16 else {}
        model = UNet2DConditionModel.from_pretrained(unet_model_dir, **model_load_opts).to(self.device)
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['sample', 'timestep', 'encoder_hidden_states', 'text_embeds', 'time_ids']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        xB = '2B' if self.xB == 2 else 'B'
        return {
            'sample': {0: xB, 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: xB},
            'latent': {0: xB, 2: 'H', 3: 'W'},
            'text_embeds': {0: xB},
            'time_ids': {0: xB}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [(self.xB*min_batch, self.unet_dim, min_latent_height, min_latent_width), (self.xB*batch_size, self.unet_dim, latent_height, latent_width), (self.xB*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
            'encoder_hidden_states': [(self.xB*min_batch, self.text_maxlen, self.embedding_dim), (self.xB*batch_size, self.text_maxlen, self.embedding_dim), (self.xB*max_batch, self.text_maxlen, self.embedding_dim)],
            'text_embeds': [(self.xB*min_batch, 1280), (self.xB*batch_size, 1280), (self.xB*max_batch, 1280)],
            'time_ids': [(self.xB*min_batch, self.time_dim), (self.xB*batch_size, self.time_dim), (self.xB*max_batch, self.time_dim)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (self.xB*batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (self.xB*batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (self.xB*batch_size, 4, latent_height, latent_width),
            'text_embeds': (self.xB*batch_size, 1280),
            'time_ids': (self.xB*batch_size, self.time_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(self.xB*batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
            torch.tensor([1.], dtype=torch.float32, device=self.device),
            torch.randn(self.xB*batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            {
                'added_cond_kwargs': {
                    'text_embeds': torch.randn(self.xB*batch_size, 1280, dtype=dtype, device=self.device),
                    'time_ids' : torch.randn(self.xB*batch_size, self.time_dim, dtype=dtype, device=self.device)
                }
            }
        )


class VAEModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=16,
    ):
        super(VAEModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, max_batch_size=max_batch_size)
        self.subfolder = 'vae'

    def get_model(self, torch_inference=''):
        vae_decoder_model_path = get_checkpoint_dir(self.framework_model_dir, self.subfolder)
        if not os.path.exists(vae_decoder_model_path):
            model = AutoencoderKL.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                use_auth_token=self.hf_token).to(self.device)
            model.save_pretrained(vae_decoder_model_path)
        else:
            print(f"[I] Load VAE decoder pytorch model from: {vae_decoder_model_path}")
            model = AutoencoderKL.from_pretrained(vae_decoder_model_path).to(self.device)
        model.forward = model.decode
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
       return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(min_batch, 4, min_latent_height, min_latent_width), (batch_size, 4, latent_height, latent_width), (max_batch, 4, max_latent_height, max_latent_width)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, version, pipeline, hf_token, device, path, framework_model_dir, hf_safetensor=False):
        super().__init__()
        vae_encoder_model_dir = get_checkpoint_dir(framework_model_dir, 'vae_encoder')
        if not os.path.exists(vae_encoder_model_dir):
            self.vae_encoder = AutoencoderKL.from_pretrained(path,
                subfolder='vae',
                use_safetensors=hf_safetensor,
                use_auth_token=hf_token).to(device)
            self.vae_encoder.save_pretrained(vae_encoder_model_dir)
        else:
            print(f"[I] Load VAE encoder pytorch model from: {vae_encoder_model_dir}")
            self.vae_encoder = AutoencoderKL.from_pretrained(vae_encoder_model_dir).to(device)

    def forward(self, x):
        return self.vae_encoder.encode(x).latent_dist.sample()


class VAEEncoderModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=16,
    ):
        super(VAEEncoderModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, max_batch_size=max_batch_size)

    def get_model(self, torch_inference=''):
        vae_encoder = TorchVAEEncoder(self.version, self.pipeline, self.hf_token, self.device, self.path, self.framework_model_dir, hf_safetensor=self.hf_safetensor)
        return vae_encoder

    def get_input_names(self):
        return ['images']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'images': {0: 'B', 2: '8H', 3: '8W'},
            'latent': {0: 'B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, _, _, _, _ = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            'images': [(min_batch, 3, min_image_height, min_image_width), (batch_size, 3, image_height, image_width), (max_batch, 3, max_image_height, max_image_width)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'images': (batch_size, 3, image_height, image_width),
            'latent': (batch_size, 4, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)


# make_tokenizer function removed as tokenizers are loaded directly in StableDiffusionPipeline.loadEngines