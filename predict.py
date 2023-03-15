import gc
import torch
from cog import BasePredictor, Input, Path
from lora_diffusion.cli_lora_pti import train as lora_train
import json
import uuid
import re

from common import (
    clean_directories,
    extract_zip_and_flatten,
    extract_urls_and_flatten,
    get_output_filename,
)


class Predictor(BasePredictor):
    def predict(
        self,
        instance_data: Path = Input(
            description="A ZIP file containing your training images (JPG, PNG, etc. size not restricted). These images contain your 'subject' that you want the trained model to embed in the output domain for later generating customized scenes beyond the training images. For best results, use images without noise or unrelated objects in the background.",
            default=None
        ),
        instance_data_urls: str = Input(
            description="A list of URLs that can be used instead of a zip file. Do not use both.",
            default=None
        ),
        base_model: str = Input(
            description="A base_model",
            default="runwayml/stable-diffusion-v1-5"
        ),
        task: str = Input(
            default="face",
            choices=["face", "object", "style", "location"],
            description="Type of LoRA model you want to train: face, style, or object",
        ),
        resolution: int = Input(
            description="OPTIONAL: The resolution for input images. All the images in the train/validation dataset will be resized to this"
            " resolution.",
            default=512,
        ),
        max_train_steps_ti: int = Input(
            default=700,
            description="OPTIONAL: The maximum number of training steps for the TI.",
        ),
        max_train_steps_tuning: int = Input(
            default=700,
            description="OPTIONAL: The maximum number of training steps for the tuning.",
        ),
        learning_rate_text: float = Input(
            default=1e-5,
            description="OPTIONAL: The learning rate for the text encoder.",
        ),
        learning_rate_ti: float = Input(
            default=5e-4,
            description="OPTIONAL: The learning rate for the TI.",
        ),
        learning_rate_unet: float = Input(
            default=2e-4,
            description="OPTIONAL: The learning rate for the unet.",
        ),
        lr_scheduler: str = Input(
            description="OPTIONAL: The scheduler type to use",
            choices=[
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
            default="constant",
        ),
        lora_rank: int = Input(
            default=16,
            description="Rank of the LoRA. Larger it is, more likely to capture fidelity but less likely to be editable. Larger rank will make the end result larger.",
        ),
    ) -> Path:

        if instance_data is None and instance_data_urls is None:
            raise Exception('no instance data provided')

        seed = 0
        print(f"Using seed: {seed}")

        cog_instance_data = "cog_instance_data"
        cog_output_dir = "checkpoints"
        clean_directories([cog_instance_data, cog_output_dir])
        if instance_data is not None:
            extract_zip_and_flatten(instance_data, cog_instance_data)
        if instance_data_urls is not None:
            extract_urls_and_flatten(json.loads(
                instance_data_urls), cog_instance_data)

        params = {
            "save_steps": max_train_steps_tuning,
            "pretrained_model_name_or_path": "stable-diffusion-v1-5-cache",
            "instance_data_dir": cog_instance_data,
            "output_dir": cog_output_dir,
            "resolution": resolution,
            "seed": seed,
            "learning_rate_text": learning_rate_text,
            "learning_rate_ti": learning_rate_ti,
            "learning_rate_unet": learning_rate_unet,
            "lr_scheduler": lr_scheduler,
            "lr_scheduler_lora": lr_scheduler,
            "max_train_steps_ti": max_train_steps_ti,
            "max_train_steps_tuning": max_train_steps_tuning,
            "lora_rank": lora_rank,
            "use_template": task,
            "use_face_segmentation_condition": task == "face",
            "enable_xformers_memory_efficient_attention": False,
            "train_text_encoder": True,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": False,
            "scale_lr": True,
            "lr_warmup_steps": 0,
            "clip_ti_decay": True,
            "color_jitter": True,
            "continue_inversion": False,
            "continue_inversion_lr": 1e-4,
            "initializer_tokens": None,
            "lr_warmup_steps_lora": 0,
            "placeholder_token_at_data": None,
            "placeholder_tokens": "<s1>|<s2>",
            "weight_decay_lora": 0.001,
            "weight_decay_ti": 0,
        }

        lora_train(**params)

        gc.collect()
        torch.cuda.empty_cache()

        weights_path = Path(cog_output_dir) / \
            f"step_{max_train_steps_tuning}.safetensors"
        output_path = Path(cog_output_dir) / \
            get_output_filename(str(uuid.uuid4()))
        weights_path.rename(output_path)

        return output_path
