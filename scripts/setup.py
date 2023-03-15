#!/usr/bin/env python


import os
import sys
import torch
from diffusers import StableDiffusionPipeline
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from transformers import CLIPTextModel, CLIPTokenizer


cache_dir = "stable-diffusion-v1-5-cache"
vae_cache_dir = "sd-vae-ft-mse-cache"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(vae_cache_dir, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    safety_checker=None
)

pipe.save_pretrained(cache_dir)
