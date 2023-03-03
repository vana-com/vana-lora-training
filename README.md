# vana-lora-training
The Vana Lora Training Service

[![Replicate](https://replicate.com/replicate/lora/badge)](https://replicate.com/vana-com/vana-lora-training)

# Model description

This model will create a Lora file.

The Lora file has information about the subject that you trained on, but doesn't save the base model, so it's a small file size.

The Lora could be a face, object, or style.

# Training a Model 

This is for the replicate frontend.

[![Replicate](https://replicate.com/replicate/lora/badge)](https://replicate.com/vana-com/vana-lora-training)

- Create a .zip file of some images (png or jpg) of the concept you want to train.
- Drop the .zip file in the instance_data field.
- Set the task to what you are training. (face, object, or style)
- Hit Submit
- Depending on the parameters and whether we need to "cold start" training will take 3-15 minutes.

## When Training is Done
### If you left the page
- click your avatar, you'll see a Dashboard icon, this will show you the jobs you've started
- find the training session you ran, and click on the id number (far left column)

### At the results page
- you should see all the settings you used, as well as a download button.
- Right click the button and "Copy Link Address", this will be used in vana-lora-inference

### If you want to make that Lora Available for others to use

Put it on your google drive

Share so anyone with the link can view it

Change the URL from https://drive.google.com/file/d/1LJuL5i4ihcNPY93-81uEUS2tOyY-25TN/view?usp=sharing to https://drive.google.com/uc?id=1LJuL5i4ihcNPY93-81uEUS2tOyY-25TN. Switching /file/d/ to uc?id= and take /view?usp=sharing off of the end of it.

### Optional Parameters

- resolution: Your images will be cropped or resized to a square of this resolution before being passed into the training pipeline. In portrait we're saving the preprocessed images as 512x512, so if you use that folder, it doesn't touch the images.
- max_train_steps_ti: TODO: Figure out what this is I don't understand why we're training the ti which I think stands for Textual Inverter?
- max_train_steps_tuning: This is how many training steps we're going to do. More steps means we capture more detail from the image, but it takes longer. If we do too many, we can overfit the model, which means that the AI will attempt to regenerate the training images exactly.

- learning_rates: These are how much oomfph to give each training step. Realistic ranges are 0.0000001 to 0.001. you'll usually see these in scientific notation (1e-6) since they're tiny little decimals. learning_rate_ti is for the Textual Inversion, learning_rate_unet is for the Unet, and learning_rate_text is for the text encoder.

# Relevant Background
Replicate Input and Output Data
When you run any replicate model via the UI, they save the inputs and outputs indefinitely. If you run this via the API, it deletes the inputs and outputs after 1 hour.

## Stable Diffusion Pipelines
Stable diffusion pipelines are actually several models working together, being coordinated by a Scheduler.
The models are called the Text Encoder, Variable Auto-Encoder (VAE), and UNet.

When we fine-tune a model, we're changing the mathematical weights in the UNet, and optionally in the Text Encoder. The VAE stays the same.

## Irrelevant Background: 
``` Most stable diffusion pipelines also include Safety Checker, Scheduler, Feature Extractors, and Tokenizers. The `Safety Checker` is also a model, but it's optional, if you want to check at the end for NSFW images. The other pieces of a stable diffusion pipeline are configured tools that aren't actually models. ```