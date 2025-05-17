import torch
from torch.nn import functional as F
import clip    #pip install git+https://github.com/openai/CLIP.git
from PIL import Image
import os
import inspect
from typing import List, Optional, Union
from diffusers.pipelines.unclip import UnCLIPTextProjModel
from diffusers import UNet2DConditionModel, UNet2DModel, UnCLIPScheduler, DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)


# This implementation uses the UnCLIPImageVariationPipeline from the
# Hugging Face Diffusers library.
class UnCLIPImageVariationPipeline(DiffusionPipeline):
    """
    Pipeline to generate variations from an input image using unCLIP
_
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `image_encoder`.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder. unCLIP Image Variation uses the vision portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_proj ([`UnCLIPTextProjModel`]):
            Utility class to prepare and combine the embeddings before they are passed to the decoder.
        decoder ([`UNet2DConditionModel`]):
            The decoder to invert the image embedding into an image.
        super_res_first ([`UNet2DModel`]):
            Super resolution unet. Used in all but the last step of the super resolution diffusion process.
        super_res_last ([`UNet2DModel`]):
            Super resolution unet. Used in the last step of the super resolution diffusion process.
        decoder_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the decoder denoising process. Just a modified DDPMScheduler.
        super_res_scheduler ([`UnCLIPScheduler`]):
            Scheduler used in the super resolution denoising process. Just a modified DDPMScheduler.

    """

    decoder: UNet2DConditionModel
    text_proj: UnCLIPTextProjModel
    text_encoder: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer
    feature_extractor: CLIPImageProcessor
    image_encoder: CLIPVisionModelWithProjection
    super_res_first: UNet2DModel
    super_res_last: UNet2DModel

    decoder_scheduler: UnCLIPScheduler
    super_res_scheduler: UnCLIPScheduler

    # Copied from diffusers.pipelines.unclip.pipeline_unclip_image_variation.UnCLIPImageVariationPipeline.__init__
    def __init__(
        self,
        decoder: UNet2DConditionModel,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_proj: UnCLIPTextProjModel,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        super_res_first: UNet2DModel,
        super_res_last: UNet2DModel,
        decoder_scheduler: UnCLIPScheduler,
        super_res_scheduler: UnCLIPScheduler,
    ):
        super().__init__()

        self.register_modules(
            decoder=decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_proj=text_proj,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            super_res_first=super_res_first,
            super_res_last=super_res_last,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool().to(device)
        text_encoder_output = self.text_encoder(text_input_ids.to(device))

        prompt_embeds = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(device)
            negative_prompt_embeds_text_encoder_output = self.text_encoder(uncond_input.input_ids.to(device))

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.text_embeds
            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_encoder_hidden_states, text_mask

    def _encode_image(self, image, device, num_images_per_prompt, image_embeddings: Optional[torch.Tensor] = None):
        dtype = next(self.image_encoder.parameters()).dtype

        if image_embeddings is None:
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

            image = image.to(device=device, dtype=dtype)
            image_embeddings = self.image_encoder(image).image_embeds

        image_embeddings = image_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        return image_embeddings

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Union[Image.Image, List[Image.Image], torch.Tensor]] = None,
        num_images_per_prompt: int = 1,
        decoder_num_inference_steps: int = 25,
        super_res_num_inference_steps: int = 7,
        generator: Optional[torch.Generator] = None,
        decoder_latents: Optional[torch.Tensor] = None,
        super_res_latents: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        decoder_guidance_scale: float = 8.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        """
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.Tensor`):
                `Image` or tensor representing an image batch to be used as the starting point. If you provide a
                tensor, it needs to be compatible with the [`CLIPImageProcessor`]
                [configuration](https://huggingface.co/fusing/karlo-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
                Can be left as `None` only when `image_embeddings` are passed.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            decoder_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps for the decoder. More denoising steps usually lead to a higher quality
                image at the expense of slower inference.
            super_res_num_inference_steps (`int`, *optional*, defaults to 7):
                The number of denoising steps for super resolution. More denoising steps usually lead to a higher
                quality image at the expense of slower inference.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            decoder_latents (`torch.Tensor` of shape (batch size, channels, height, width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            super_res_latents (`torch.Tensor` of shape (batch size, channels, super res height, super res width), *optional*):
                Pre-generated noisy latents to be used as inputs for the decoder.
            decoder_guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            image_embeddings (`torch.Tensor`, *optional*):
                Pre-defined image embeddings that can be derived from the image encoder. Pre-defined image embeddings
                can be passed for tasks like image interpolations. `image` can be left as `None`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
        if image is not None:
            if isinstance(image, Image.Image):
                batch_size = 1
            elif isinstance(image, list):
                batch_size = len(image)
            else:
                batch_size = image.shape[0]
        else:
            batch_size = image_embeddings.shape[0]

        prompt = [""] * batch_size

        device = self._execution_device

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = decoder_guidance_scale > 1.0

        prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance
        )

        image_embeddings = self._encode_image(image, device, num_images_per_prompt, image_embeddings)

        # decoder
        text_encoder_hidden_states, additive_clip_time_embeddings = self.text_proj(
            image_embeddings=image_embeddings,
            prompt_embeds=prompt_embeds,
            text_encoder_hidden_states=text_encoder_hidden_states,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        if device.type == "mps":
            # HACK: MPS: There is a panic when padding bool tensors,
            # so cast to int tensor for the pad and back to bool afterwards
            text_mask = text_mask.type(torch.int)
            decoder_text_mask = F.pad(text_mask, (self.text_proj.clip_extra_context_tokens, 0), value=1)
            decoder_text_mask = decoder_text_mask.type(torch.bool)
        else:
            decoder_text_mask = F.pad(text_mask, (self.text_proj.clip_extra_context_tokens, 0), value=True)

        self.decoder_scheduler.set_timesteps(decoder_num_inference_steps, device=device)
        decoder_timesteps_tensor = self.decoder_scheduler.timesteps

        num_channels_latents = self.decoder.config.in_channels
        height = self.decoder.config.sample_size
        width = self.decoder.config.sample_size

        if decoder_latents is None:
            decoder_latents = self.prepare_latents(
                (batch_size, num_channels_latents, height, width),
                text_encoder_hidden_states.dtype,
                device,
                generator,
                decoder_latents,
                self.decoder_scheduler,
            )

        for i, t in enumerate(self.progress_bar(decoder_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([decoder_latents] * 2) if do_classifier_free_guidance else decoder_latents

            noise_pred = self.decoder(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_encoder_hidden_states,
                class_labels=additive_clip_time_embeddings,
                attention_mask=decoder_text_mask,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input.shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(latent_model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + decoder_guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            if i + 1 == decoder_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = decoder_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            decoder_latents = self.decoder_scheduler.step(
                noise_pred, t, decoder_latents, prev_timestep=prev_timestep, generator=generator
            ).prev_sample

        decoder_latents = decoder_latents.clamp(-1, 1)

        image_small = decoder_latents

        # done decoder

        # super res

        self.super_res_scheduler.set_timesteps(super_res_num_inference_steps, device=device)
        super_res_timesteps_tensor = self.super_res_scheduler.timesteps

        channels = self.super_res_first.config.in_channels // 2
        height = self.super_res_first.config.sample_size
        width = self.super_res_first.config.sample_size

        if super_res_latents is None:
            super_res_latents = self.prepare_latents(
                (batch_size, channels, height, width),
                image_small.dtype,
                device,
                generator,
                super_res_latents,
                self.super_res_scheduler,
            )

        if device.type == "mps":
            # MPS does not support many interpolations
            image_upscaled = F.interpolate(image_small, size=[height, width])
        else:
            interpolate_antialias = {}
            if "antialias" in inspect.signature(F.interpolate).parameters:
                interpolate_antialias["antialias"] = True

            image_upscaled = F.interpolate(
                image_small, size=[height, width], mode="bicubic", align_corners=False, **interpolate_antialias
            )

        for i, t in enumerate(self.progress_bar(super_res_timesteps_tensor)):
            # no classifier free guidance

            if i == super_res_timesteps_tensor.shape[0] - 1:
                unet = self.super_res_last
            else:
                unet = self.super_res_first

            latent_model_input = torch.cat([super_res_latents, image_upscaled], dim=1)

            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
            ).sample

            if i + 1 == super_res_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = super_res_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            super_res_latents = self.super_res_scheduler.step(
                noise_pred, t, super_res_latents, prev_timestep=prev_timestep, generator=generator
            ).prev_sample

        image = super_res_latents

        # done super res
        self.maybe_free_model_hooks()

        # post processing

        image = image * 0.5 + 0.5
        image = image.clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """
    Perform Spherical Linear Interpolation (SLERP) between two vectors.

    Args:
        val (float): Interpolation factor.
        low (torch.Tensor): Starting vector.
        high (torch.Tensor): Ending vector.

    Returns:
        torch.Tensor: Interpolated vector.
    """
    # Normalize input vectors
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)

    # Compute angular distance
    omega = torch.acos(torch.dot(low_norm, high_norm))
    so = torch.sin(omega)

    # Compute SLERP interpolation
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high


# -----------------------------------
# Setup Device and Load CLIP Model
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and its preprocessing pipeline
model, preprocess = clip.load("ViT-L/14", device=device)

# -----------------------------------
# Load whitening matrix and mean feature vector
# -----------------------------------
w_path = 'w_mats/w_mat_image_L14.pt'
mean_path = 'w_mats/mean_image_L14.pt'

w_mat = torch.load(w_path, map_location=device, weights_only=False)
inv_w_mat = torch.inverse(w_mat)  # Compute inverse whitening matrix
mean_features = torch.load(mean_path, map_location=device, weights_only=False)

N = len(mean_features)  # Dimensionality of feature space

# -----------------------------------
# Configuration Parameters
# -----------------------------------
seed = 42      # Random seed for reproducibility
step = 5       # Step size (in degrees) for interpolation
wht = True     # Use W-CLIP interpolation

# -----------------------------------
# Create Output Directories
# -----------------------------------
results_dir_path = f'dog_elph_circle_step_{step}_wht'           # add path to output directory
images_dir_path = os.path.join(results_dir_path, 'images')

os.makedirs(results_dir_path, exist_ok=True)
os.makedirs(images_dir_path, exist_ok=True)

# -----------------------------------
# Load Source and Destination Images
# -----------------------------------
src_image_path = '../slerp_tests/images/000000297830.jpg' # add path to source image
dst_image_path = '../slerp_tests/images/000000133631.jpg' # add path to destination image

# Preprocess images using CLIP's pipeline
src_image = preprocess(Image.open(src_image_path).convert("RGB")).unsqueeze(0).to(device)
dst_image = preprocess(Image.open(dst_image_path).convert("RGB")).unsqueeze(0).to(device)

# Extract CLIP image features
src_image_features = model.encode_image(src_image).squeeze(0).to(device, dtype=torch.float32)
dst_image_features = model.encode_image(dst_image).squeeze(0).to(device, dtype=torch.float32)

# -----------------------------------
# Feature Transformation for Spherical Interpolation
# -----------------------------------
if wht:
    # Centering and transforming features
    src = torch.matmul((src_image_features - mean_features), w_mat)
    dst = torch.matmul((dst_image_features - mean_features), w_mat)
else:
    # Use raw CLIP features without transformation
    src = src_image_features
    dst = dst_image_features

# Normalize vectors
src_norm = src / torch.norm(src)
dst_norm = dst / torch.norm(dst)

# Compute angle between source and destination in degrees
omega_src_dst = torch.rad2deg(torch.acos(torch.dot(src_norm, dst_norm)))

# -----------------------------------
# Perform SLERP Interpolation
# -----------------------------------
circle_features = []
for idx, degree in enumerate(range(0, 361, step)):
    # Compute output image path
    step_image_path = os.path.join(images_dir_path, f'slerp_step_{idx}_(degree_{degree:.2f}).png')

    # Compute interpolation factor (t)
    t = degree / omega_src_dst

    # Perform SLERP interpolation
    step_features = slerp(t, src, dst)

    if wht:
        # Transform features back to raw space
        step_features = torch.matmul(step_features, inv_w_mat) + mean_features

    circle_features.append(step_features.detach().cpu())

    # -----------------------------------
    # Generate Image using UnCLIPImageVariationPipeline
    # -----------------------------------
    with torch.autocast("cuda"):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load the UnCLIP pipeline if not already loaded
        if 'pipe' not in globals():
            pipe = UnCLIPImageVariationPipeline.from_pretrained(
                "kakaobrain/karlo-v1-alpha-image-variations",
                torch_dtype=torch.float16
            ).to(device)

        # Generate image from feature embeddings
        step_image = pipe(image_embeddings=step_features.unsqueeze(0).to(device)).images[0]

        # Save the generated image
        step_image.save(step_image_path)

# -----------------------------------
# Save Interpolated Features
# -----------------------------------
torch.save(torch.stack(circle_features), os.path.join(results_dir_path, 'circle_features.pt'))
