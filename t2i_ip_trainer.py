import torch
from PIL import Image

from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad, image_to_tensor
from refiners.foundationals.latent_diffusion.lora import SDLoraManager
from refiners.foundationals.latent_diffusion.stable_diffusion_xl import StableDiffusion_XL, SDXLT2IAdapter
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.image_prompt import SDXLIPAdapter

# Load SDXL
sdxl = StableDiffusion_XL(device="cuda", dtype=torch.float16)
sdxl.clip_text_encoder.load_from_safetensors("DoubleCLIPTextEncoder.safetensors")
sdxl.unet.load_from_safetensors("sdxl-unet.safetensors")
sdxl.lda.load_from_safetensors("sdxl-lda.safetensors")

# Load LoRAs weights from disk and inject them into target
manager = SDLoraManager(sdxl)
scifi_lora_weights = load_from_safetensors("Sci-fi_Environments_sdxl.safetensors")
pixel_art_lora_weights = load_from_safetensors("pixel-art-xl-v1.1.safetensors")
manager.add_loras("scifi-lora", scifi_lora_weights, scale=1.5)
manager.add_loras("pixel-art-lora", pixel_art_lora_weights, scale=1.55)

# Load IP-Adapter
ip_adapter = SDXLIPAdapter(
    target=sdxl.unet,
    weights=load_from_safetensors("ip-adapter-plus_sdxl_vit-h.safetensors"),
    scale=1.0,
    fine_grained=True,  # Use fine-grained IP-Adapter (IP-Adapter Plus)
)
ip_adapter.clip_image_encoder.load_from_safetensors("CLIPImageEncoderH.safetensors")
ip_adapter.inject()

# Load T2I-Adapter
t2i_adapter = SDXLT2IAdapter(
    target=sdxl.unet, 
    name="zoe-depth", 
    weights=load_from_safetensors("t2i_depth_zoe_xl.safetensors"),
    scale=0.72,
).inject()

# Hyperparameters
prompt = "a futuristic castle surrounded by a forest, mountains in the background"
image_prompt = Image.open("german-castle.jpg")
image_depth_condition = Image.open("zoe-depth-map-german-castle.png")
seed = 42
sdxl.set_inference_steps(50, first_step=0)
sdxl.set_self_attention_guidance(
    enable=True, scale=0.75
)  # Enable self-attention guidance to enhance the quality of the generated images

with no_grad():
    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt + ", best quality, high quality",
        negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
    )
    time_ids = sdxl.default_time_ids

    clip_image_embedding = ip_adapter.compute_clip_image_embedding(ip_adapter.preprocess_image(image_prompt))
    ip_adapter.set_clip_image_embedding(clip_image_embedding)

    # Spatial dimensions should be divisible by default downscale factor (=16 for T2IAdapter ConditionEncoder)
    condition = image_to_tensor(image_depth_condition.convert("RGB").resize((1024, 1024)), device=sdxl.device, dtype=sdxl.dtype)
    t2i_adapter.set_condition_features(features=t2i_adapter.compute_condition_features(condition))

    manual_seed(seed=seed)
    x = sdxl.init_latents((1024, 1024)).to(sdxl.device, sdxl.dtype)

    # Diffusion process
    for step in sdxl.steps:
        if step % 10 == 0:
            print(f"Step {step}")
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
        )
    predicted_image = sdxl.lda.decode_latents(x)

predicted_image.save("scifi_pixel_IP_T2I_sdxl.png")