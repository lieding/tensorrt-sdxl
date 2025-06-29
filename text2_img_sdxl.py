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

import argparse
# import inspect # No longer needed

from cuda import cudart

from stable_diffusion_pipeline import StableDiffusionPipeline
from utilities import PIPELINE_TYPE, TRT_LOGGER, add_arguments, process_pipeline_args

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion XL Txt2Img Demo", conflict_handler='resolve')
    parser = add_arguments(parser) # Adds common arguments.

    # Script-specific arguments or overrides for SDXL defaults if different from utilities.py
    # --version is fully removed.
    # Overriding defaults for height, width, and num-warmup-runs for SDXL.
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate for SDXL (must be multiple of 8).")
    parser.add_argument('--width', type=int, default=1024, help="Width of image to generate for SDXL (must be multiple of 8).")
    parser.add_argument('--num-warmup-runs', type=int, default=1, help="Number of warmup runs for SDXL demo.")

    # Refiner related arguments are removed:
    # --enable-refiner, --image-strength, --onnx-refiner-dir, --engine-refiner-dir
    # aesthetic_score and negative_aesthetic_score were handled by StableDiffusionPipeline.infer,
    # but since refiner is removed, these are no longer relevant for this script's direct CLI.

    return parser.parse_args()

# StableDiffusionXLPipeline class is removed as we will use StableDiffusionPipeline directly.

if __name__ == "__main__":
    print("[I] Initializing TensorRT accelerated StableDiffusionXL (Base) txt2img pipeline")
    args = parseArgs()

    # process_pipeline_args from utilities.py is used to parse common arguments
    # It will populate kwargs_init_pipeline, kwargs_load_engine, args_run_demo

    kwargs_init_pipeline, kwargs_load_engine, args_run_demo = process_pipeline_args(args)

    # Ensure pipeline_type is set to XL_BASE for the direct StableDiffusionPipeline instantiation
    # process_pipeline_args doesn't set pipeline_type, so we set it here.
    kwargs_init_pipeline['pipeline_type'] = PIPELINE_TYPE.XL_BASE
    # return_latents is False by default in add_arguments if not specified.
    # For a simple txt2img, False (meaning return images) is fine.
    # If 'return_latents' was intended to be a CLI arg, it should be added to add_arguments or parseArgs here.
    # Assuming default behavior is to return images.
    kwargs_init_pipeline.setdefault('return_latents', False)

    # Set SDXL specific vae_scaling_factor, if not already set by utilities.py (e.g. if it became a common arg)
    # The simplified StableDiffusionPipeline __init__ has a default for this.
    # We can override it here if parseArgs or process_pipeline_args specifically handled it for XL.
    # For now, relying on the default in StableDiffusionPipeline's __init__ (0.13025) or whatever process_pipeline_args might set.
    # To be safe, let's ensure it's the SDXL one if not otherwise specified by a more general mechanism.
    if 'vae_scaling_factor' not in kwargs_init_pipeline: # This default is now set in StableDiffusionPipeline.__init__
         kwargs_init_pipeline['vae_scaling_factor'] = 0.13025 # Ensure it if not set by process_pipeline_args

    # Remove 'version' from kwargs_init_pipeline if it exists, as StableDiffusionPipeline no longer accepts it.
    if 'version' in kwargs_init_pipeline:
        del kwargs_init_pipeline['version']

    # Initialize demo with the simplified StableDiffusionPipeline
    demo = StableDiffusionPipeline(**kwargs_init_pipeline)

    # Load TensorRT engines (only UNetXL) and PyTorch modules (CLIPs, VAE)
    demo.loadEngines(
        engine_dir=args.engine_dir, # For UNetXL TRT engine
        framework_model_dir=args.framework_model_dir, # For PyTorch model caching by diffusers/transformers
        onnx_dir=args.onnx_dir, # For UNetXL ONNX model
        **kwargs_load_engine # Contains opt_batch_size, image_height/width etc.
    )

    # Load resources
    max_mem = demo.calculateMaxDeviceMemory() # Calculates for UNetXL TRT

    shared_device_memory = None
    if max_mem > 0:
        _, shared_device_memory = cudart.cudaMalloc(max_mem)
    else:
        # This case might occur if UNetXL is not built/loaded (e.g. error in path or build)
        # or if calculateMaxDeviceMemory returns 0 incorrectly.
        # For PyTorch only components (if UNetXL was also PyTorch), max_mem would be 0.
        # Since UNetXL is TRT, max_mem should ideally be > 0.
        print("[W] Max device memory for TRT engines calculated as 0. Shared memory allocation skipped. This might be an issue if TRT UNetXL is expected.")

    demo.activateEngines(shared_device_memory) # Activates UNetXL TRT
    demo.loadResources(args.height, args.width, args.batch_size, args.seed) # Sets up generator, CUDA events, stream

    # Run inference
    # args_run_demo from process_pipeline_args typically includes:
    # (prompt, negative_prompt, height, width, batch_size, batch_count, num_warmup_runs, use_cuda_graph_flag)
    # The simplified demo.run expects:
    # (self, prompt, negative_prompt, height, width, batch_size, batch_count, num_warmup_runs, **kwargs)
    # We need to slice args_run_demo to exclude the use_cuda_graph flag if it's still there.

    # Assuming process_pipeline_args structure:
    # args_run_demo = (args.prompt, args.negative_prompt, args.height, args.width, args.batch_size, args.batch_count, args.num_warmup_runs, args.use_cuda_graph)
    # The simplified StableDiffusionPipeline.run does not take use_cuda_graph.

    # Check number of elements in args_run_demo to be safe, though process_pipeline_args should be consistent.
    if len(args_run_demo) == 8: # Original structure including use_cuda_graph
        run_args_tuple = args_run_demo[:-1]
    elif len(args_run_demo) == 7: # If process_pipeline_args was already updated to exclude use_cuda_graph
        run_args_tuple = args_run_demo
    else:
        raise ValueError(f"Unexpected number of arguments from process_pipeline_args: {len(args_run_demo)}")

    # The **kwargs for demo.run() would be for any extra parameters to StableDiffusionPipeline.infer()
    # The simplified infer() takes: prompt, negative_prompt, image_height, image_width, warmup, save_image
    # These are mostly covered by run_args_tuple or have defaults.
    # No specific infer_kwargs needed from here for the base SDXL run.
    demo.run(*run_args_tuple)

    demo.teardown()