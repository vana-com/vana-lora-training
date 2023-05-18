import gc
import torch
from cog import BasePredictor, Input, Path
from lora_diffusion.cli_lora_pti import train as lora_train
import json
import uuid
import re
from vanautils import FileManager
import os
import shutil
from PIL import Image, ImageOps
import glob

from common import (
    clean_directories,
    extract_zip_and_flatten,
    extract_urls_and_flatten,
    get_output_filename,
)

IMAGE_DIR = "/tmp/vana-lora-training/images"
CHECKPOINT_DIR = "/tmp/vana-lora-training/checkpoints"

class Predictor(BasePredictor):
    def setup(self):
        self.file_manager = FileManager(download_dir=IMAGE_DIR)

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
        custom_prompts: str = Input(
            description='A list of custom training prompts. Use {} for the token to train. '
            'e.g. ["a cell phone photo of {}, on a carpeted floor, apartment interior, bright lighting"]',
            default=None
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

        clean_directories([IMAGE_DIR, CHECKPOINT_DIR])

        [
            self.file_manager.download_file(image_url) for image_url in json.loads(instance_data_urls)
        ] if instance_data_urls is not None else None

        if instance_data is not None:
            extract_zip_and_flatten(instance_data, IMAGE_DIR)

        for im in sorted(glob.glob(IMAGE_DIR + "/*")):
            imopened = Image.open(im)
            width, height = imopened.size
            if width > resolution or height > resolution: # if you're bigger than 1.5mb, resize to smaller so the training call does not time out
                imoppped = ImageOps.exif_transpose(imopened.convert('RGB'))
                imoppped.thumbnail([resolution,resolution], Image.LANCZOS)
                left = (width - resolution)/2
                top = (height - resolution)/2
                right = (width + resolution)/2
                bottom = (height + resolution)/2
                # Crop the center of the image
                imoppped = imoppped.crop((left, top, right, bottom))
                imoppped.save(im + f"_{resolution}.jpg", quality=100, optimize=True)
                os.remove(im)
                rezip = True
        if rezip:
            shutil.make_archive(IMAGE_DIR, "zip", IMAGE_DIR)
            instance_data = Path(IMAGE_DIR + ".zip")

        params = {
            "save_steps": max_train_steps_tuning,
            "pretrained_model_name_or_path": "stable-diffusion-v1-5-cache",
            "instance_data_dir": IMAGE_DIR,
            "output_dir": CHECKPOINT_DIR,
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
            "mixed_precision_tune":True
        }
        
        if custom_prompts is not None:
            params["custom_prompts"] = json.loads(custom_prompts)

        lora_train(**params)

        gc.collect()
        torch.cuda.empty_cache()

        weights_path = Path(CHECKPOINT_DIR) / \
            f"step_{max_train_steps_tuning}.safetensors"
        output_path = Path(CHECKPOINT_DIR) / \
            get_output_filename(str(uuid.uuid4()))
        weights_path.rename(output_path)

        return output_path
