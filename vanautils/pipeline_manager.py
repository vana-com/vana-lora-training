import os
import math
import gc
import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionAttendAndExcitePipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)

from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from lora_diffusion import LoRAManager

class PipelineManager:
    def __init__(self, **kwargs):
        self._pipe = None
        self._base_model = kwargs.get("base_model", None)
        if self._base_model is None:
            if os.path.exists("base"):
                self._base_model = "base"
            else:
                self._base_model = "neilzumwalde/init-v-base-model-1-5"
        self._pipe_type = kwargs.get("pipe_type", StableDiffusionPipeline)
        self._device = kwargs.get("device", "cuda")
        self._scheduler = kwargs.get("scheduler", None)
        self._torch_dtype = kwargs.get("torch_dtype", torch.float16)
        self._safety_checker = kwargs.get("safety_checker", True)
        try:
            import xformers

            self._xformers = kwargs.get("xformers", True)
        except:
            self._xformers = False
            print("xformers not installed, disabling")

        self._enable_attention_slicing = kwargs.get("enable_attention_slicing", True)
        self._enable_vae_slicing = kwargs.get("enable_vae_slicing", True)
        self._enable_vae_tiling = kwargs.get("enable_vae_tiling", True)
        self._lora_scale_base = kwargs.get("lora_scale_base", 0.5)
        self._lora_paths = kwargs.get("lora_paths", None)
        self._lora_scales = kwargs.get("lora_scales", None)
        self.lora_manager = None
        self._rebuild_pipe = True
        self._reapply_loras = True

    def save(self, path="base"):
        self.pipe.save_pretrained(path)

    @property
    def pipe(self):
        if self._rebuild_pipe:
            self._build_pipe()
            self._rebuild_pipe = False
            self._reapply_loras = True
        if self._reapply_loras:
            self._apply_loras()
            self._reapply_loras = False

        return self._pipe

    @property
    def base_model(self):
        return self._base_model

    @base_model.setter
    def base_model(self, base_model):
        if base_model != self.base_model:
            self._base_model = base_model
            self._rebuild_pipe = True

    @property
    def safety_checker(self):
        return self._safety_checker

    @safety_checker.setter
    def safety_checker(self, safety_checker):
        if safety_checker != self._safety_checker:
            self._safety_checker = safety_checker
            self._set_safety_checker()

    @property
    def pipe_type(self):
        return self._pipe_type

    @pipe_type.setter
    def pipe_type(self, pipe_type):
        if pipe_type != self._pipe_type:
            self._pipe_type = pipe_type
            self._rebuild_pipe = True

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        if scheduler != self.scheduler:
            self._scheduler = scheduler
            self._set_scheduler()

    @property
    def torch_dtype(self):
        return self._torch_dtype

    @torch_dtype.setter
    def torch_dtype(self, torch_dtype):
        if torch_dtype != self._torch_dtype:
            self._torch_dtype = torch_dtype
            self._rebuild_pipe = True

    @property
    def xformers(self):
        return self._xformers

    @xformers.setter
    def xformers(self, xformers):
        if xformers != self._xformers:
            self._xformers = xformers
            if self._pipe is not None:
                if self._xformers:
                    self._pipe.enable_xformers_memory_efficient_attention()
                else:
                    self._pipe.disable_xformers_memory_efficient_attention()

    @property
    def enable_attention_slicing(self):
        return self._enable_attention_slicing

    @enable_attention_slicing.setter
    def enable_attention_slicing(self, enable_attention_slicing):
        if enable_attention_slicing != self._enable_attention_slicing:
            self._enable_attention_slicing = enable_attention_slicing
            if self._pipe is not None:
                if self._enable_attention_slicing:
                    self._pipe.enable_attention_slicing()
                else:
                    self._pipe.disable_attention_slicing()

    @property
    def enable_vae_slicing(self):
        return self._enable_vae_slicing

    @enable_vae_slicing.setter
    def enable_vae_slicing(self, enable_vae_slicing):
        if enable_vae_slicing != self._enable_vae_slicing:
            self._enable_vae_slicing = enable_vae_slicing
            if self._pipe is not None:
                if self._enable_vae_slicing:
                    self._pipe.enable_vae_slicing()
                else:
                    self._pipe.disable_vae_slicing()

    @property
    def enable_vae_tiling(self):
        return self._enable_vae_tiling

    @enable_vae_tiling.setter
    def enable_vae_tiling(self, enable_vae_tiling):
        if enable_vae_tiling != self._enable_vae_tiling:
            self._enable_vae_tiling = enable_vae_tiling
            if self._pipe is not None:
                if self._enable_vae_tiling:
                    self._pipe.enable_vae_slicing()
                else:
                    self._pipe.disable_vae_slicing()

    @property
    def lora_scales(self):
        return self._lora_scales

    @lora_scales.setter
    def lora_scales(self, lora_scales):
        if lora_scales != self._lora_scales:
            self._lora_scales = lora_scales
            self._reapply_loras = True

    @property
    def lora_paths(self):
        return self._lora_paths

    @lora_paths.setter
    def lora_paths(self, lora_paths):
        if lora_paths != self._lora_paths:
            self._lora_paths = lora_paths
            self._reapply_loras = True

    @property
    def lora_base_scale(self):
        return self._lora_base_scale

    @lora_base_scale.setter
    def lora_base_scale(self, lora_base_scale):
        if lora_base_scale != self._lora_base_scale:
            self._lora_base_scale = lora_base_scale
            self._reapply_loras = True

    def _set_safety_checker(self):
        print("safety_checker", self.safety_checker)
        if self._pipe is None:
            return
        if self.safety_checker is None or self.safety_checker == False:
            self._pipe.safety_checker = None
        else:
            safety_checker_model = "CompVis/stable-diffusion-safety-checker"
            if isinstance(self.safety_checker, str):
                safety_checker_model = self.safety_checker
            if not (
                self.safety_checker == True and self._pipe.safety_checker is not None
            ):
                self._pipe.safety_checker = (
                    StableDiffusionSafetyChecker.from_pretrained(
                        safety_checker_model,
                        torch_dtype=self.torch_dtype,
                    ).to(self._device)
                )

    def _set_scheduler(self):
        if self._pipe is None:
            return
        if self._scheduler is not None:
            self._pipe.scheduler = {
                "PNDM": PNDMScheduler.from_config(self._pipe.scheduler.config),
                "KLMS": LMSDiscreteScheduler.from_config(self._pipe.scheduler.config),
                "DDIM": DDIMScheduler.from_config(self._pipe.scheduler.config),
                "K_EULER": EulerDiscreteScheduler.from_config(
                    self._pipe.scheduler.config
                ),
                "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(
                    self._pipe.scheduler.config
                ),
                "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(
                    self._pipe.scheduler.config
                ),
                "UniPC": UniPCMultistepScheduler.from_config(
                    self._pipe.scheduler.config
                ),
            }[self.scheduler]

    def _build_pipe(self):
        self._pipe = None
        gc.collect()
        torch.cuda.empty_cache()

        self._pipe = self.pipe_type.from_pretrained(
            self.base_model, torch_dtype=self.torch_dtype
        ).to(self._device)

        self._set_safety_checker()
        self._set_scheduler()

        if self.xformers:
            self._pipe.enable_xformers_memory_efficient_attention()

        if self.enable_attention_slicing:
            self._pipe.enable_attention_slicing()

        if self.enable_vae_slicing:
            self._pipe.enable_vae_slicing()

        if self.pipe_type == StableDiffusionPipeline:
            if self.enable_vae_tiling:
                self._pipe.enable_vae_tiling()

    def _apply_loras(self):

        if self.lora_paths is None or len(self.lora_paths) == 0:
            if self.lora_manager is not None:
                self.lora_manager = None
                # TODO: untune the unet?
            return

        if self.lora_scales is None or len(self.lora_scales) != len(self.lora_paths):
            lora_scale_base = self._lora_scale_base / math.sqrt(len(self.lora_paths))
            self.lora_scales = [lora_scale_base] * len(self.lora_paths)

        self.lora_manager = LoRAManager(self.lora_paths, self._pipe)
        self.lora_manager.tune(self.lora_scales)



    def generate(
        self, seed=None, number=1, regenerate_nsfw=True, subjects=None, **kwargs
    ):
        pipe = self.pipe
        prompt = kwargs.get("prompt", None)
        negative_prompt = kwargs.get("negative_prompt", None)

        if isinstance(prompt, str):
            prompt = [prompt] * number

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompt)

        if self.lora_manager is not None:
            prompt = [self.lora_manager.prompt(p) for p in prompt]

        kwargs["prompt"] = prompt
        kwargs["negative_prompt"] = negative_prompt

        if self._pipe_type == StableDiffusionAttendAndExcitePipeline and subjects:
            kwargs["max_iter_to_alter"] = kwargs.get("max_iter_to_alter", 20)
            kwargs["scale_factor"] = kwargs.get("scale_factor", 5)
            kwargs["attn_res"] = kwargs.get(
                "attn_res",
                int(kwargs.get("width", 512) / 64)
                + int(kwargs.get("height", 512) / 64),
            )
            kwargs["token_indices"] = []
            for p in prompt:
                indices = []
                p_tokens = self.pipe.get_indices(p)
                print(p_tokens)
                for subject in subjects:
                    indices.extend(
                        [
                            key
                            for key in p_tokens.keys()
                            if p_tokens[key] == f"{subject}</w>"
                        ]
                    )
                kwargs["token_indices"].append(indices)

        if seed is None or seed == -1:
            seed = int.from_bytes(os.urandom(2), "big")

        generator = torch.Generator(self._device).manual_seed(seed)

        output = pipe(**kwargs, generator=generator)
        out_images = []

        nsfw_detected = False
        for i, image in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                nsfw_detected = True
                if not regenerate_nsfw:
                    # If we aren't going to try again, Return the black image.
                    out_images.append(image)
                continue
            out_images.append(image)

        if regenerate_nsfw and nsfw_detected:
            new_kwargs = kwargs.copy()
            new_kwargs["prompt"] = []
            new_kwargs["negative_prompt"] = []

            for i, nsfw in enumerate(output.nsfw_content_detected):
                if nsfw:
                    new_kwargs["prompt"].append(prompt[i])
                    new_kwargs["negative_prompt"].append(negative_prompt[i])
            out_images.extend(
                self.generate(seed=-1, regenerate_nsfw=False, **new_kwargs)
            )

        return out_images
