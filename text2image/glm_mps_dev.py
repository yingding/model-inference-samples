from applyllm.accelerators import (
    AcceleratorHelper, 
    DirectorySetting,
    TokenHelper,
)
dir_mode_map = {
    "kf_notebook": DirectorySetting(),
    "mac_local": DirectorySetting(
        home_dir="/Users/yingding/Code", # "/Users/yingding"
        transformers_cache_home="MODELS", 
        huggingface_token_file="MODELS/.huggingface_token"),
}
dir_setting = dir_mode_map["mac_local"]

# set up the torch mps environment and huggingface cache home, before importing datasets and transformers
AcceleratorHelper.init_torch_env(accelerator="mps", dir_setting=dir_setting)

from applyllm.utils import time_func
th = TokenHelper(dir_setting=dir_setting, prefix_list=["zai"])
token_kwargs = th.gen_token_kwargs(model_type="zai")
# remove trust_remote_code from the dict token_kwargs if exists
if "trust_remote_code" in token_kwargs:
    del token_kwargs["trust_remote_code"]

from time import time
import torch, os
from diffusers import DiffusionPipeline

# Check if CUDA is available, otherwise use CPU or MPS (for Mac)
# device_map = "mps"
if torch.cuda.is_available():
    device_map = "cuda"
elif torch.backends.mps.is_available():
   device_map = "mps"
    
# Note: The model "zai-org/GLM-Image" relies on specific implementations in diffusers and transformers.
# Ensure you have the latest versions installed from source as per instructions.

# Load the pipeline
# Using "auto" for device_map to handle different hardware configurations
# bfloat16 is recommended for newer GPUs, float16 or float32 might be needed elsewhere
dtype = torch.bfloat16 # Default to bfloat16 for CPU (Apple Silicon supports it)
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
elif device_map == "mps":
    # Use float32 for MPS to ensure stability and avoid "probability tensor contains either `inf`, `nan` or element < 0"
    dtype = torch.float32
    torch.set_float32_matmul_precision('high')


print(f"Loading model with dtype={dtype} and device_map={device_map}...")

# Note: GlmImagePipeline might not be directly importable as a class in older diffusers versions
# using the generic DiffusionPipeline.from_pretrained() is safer or importing specifically if available.
# The documentation uses: from diffusers.pipelines.glm_image import GlmImagePipeline
# But standard practice is often DiffusionPipeline if supported, however for new models specific pipelines are often needed.
# I will use the specific import as shown in the docs, but fallback or warn if it fails.

try:
    from diffusers.pipelines.glm_image import GlmImagePipeline
    PipelineClass = GlmImagePipeline
except ImportError:
    print("Could not import GlmImagePipeline. Falling back to DiffusionPipeline (might fail if not registered).")
    PipelineClass = DiffusionPipeline

# When using CPU, we cannot pass device_map="cpu". We must rely on the default behavior (which is CPU) 
# or manual .to("cpu") later. However, from_pretrained usually accepts device_map="auto" or specific logic.
# If device_map="cpu" fails (NotImplementedError), pass None or remove the argument.
load_kwargs = {
    "torch_dtype": dtype, 
    "trust_remote_code": True,
    **token_kwargs,
}
if device_map != "cpu":
    load_kwargs["device_map"] = device_map

pipe = PipelineClass.from_pretrained(
    "zai-org/GLM-Image", 
    **load_kwargs
)
# Explicitly move to CPU if intended
if device_map == "cpu":
    pipe.to("cpu")

# # Fix for black images on MPS (Mac):
# # 1. Enable VAE tiling to prevent OOM/Overflow during decoding of large images
# if hasattr(pipe, "enable_vae_tiling"):
#     pipe.enable_vae_tiling()
#     print("Enabled VAE tiling")
# if hasattr(pipe, "enable_vae_slicing"):
#     pipe.enable_vae_slicing()
#     print("Enabled VAE slicing")

# # 2. Ensure VAE is in float32 (even if strictly not needed, it helps stability)
# if hasattr(pipe, "vae") and pipe.vae is not None:
#     pipe.vae = pipe.vae.to(dtype=torch.float32)

prompt = "A beautifully designed modern food magazine style dessert recipe illustration, themed around a raspberry mousse cake. The overall layout is clean and bright, divided into four main areas: the top left features a bold black title 'Raspberry Mousse Cake Recipe Guide', with a soft-lit close-up photo of the finished cake on the right, showcasing a light pink cake adorned with fresh raspberries and mint leaves; the bottom left contains an ingredient list section, titled 'Ingredients' in a simple font, listing 'Flour 150g', 'Eggs 3', 'Sugar 120g', 'Raspberry puree 200g', 'Gelatin sheets 10g', 'Whipping cream 300ml', and 'Fresh raspberries', each accompanied by minimalist line icons (like a flour bag, eggs, sugar jar, etc.); the bottom right displays four equally sized step boxes, each containing high-definition macro photos and corresponding instructions."

print("Generating image...")

# NUM_INFERENCE_STEPS = 5
DEBUG_MODE = True
if DEBUG_MODE:
    NUM_INFERENCE_STEPS = 5
else:
    NUM_INFERENCE_STEPS = 50
    print("DEBUG_MODE is ON: Using reduced inference steps and additional logging.")

# runtime_kwargs = {
#     "num_inference_steps": NUM_INFERENCE_STEPS,
#     "output_type": "latent" if DEBUG_MODE else "pil",
# }

@time_func
def generate_image():
    # 1. Generate latents first to debug validity
    print("Step 1: Generating latents...")
    output = pipe(
        prompt=prompt,
        height=1024, # 32 * 32
        width=1152,  # 36 * 32
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=1.5,
        generator=torch.Generator(device=device_map).manual_seed(42),
        output_type="latent"
    )
    
    # The pipeline wrapper returns 'images' which contains the latents tensor when output_type="latent"
    latents = output.images if hasattr(output, "images") else output[0]
    
    # 2. Check for NaNs/Zeros
    print(f"Latents stats: Min={latents.min().item():.4f}, Max={latents.max().item():.4f}, Mean={latents.mean().item():.4f}, Std={latents.std().item():.4f}")
    
    if torch.isnan(latents).any():
        print("ERROR: Latents contain NaNs! The DiT model is unstable on MPS.")
        return None
        
    # 3. Manual VAE Decoding on CPU to avoid MPS floating point errors
    print("Step 2: Decoding VAE on CPU...")
    # Move strictly to CPU & float32 for decoding
    vae = pipe.vae.to("cpu", dtype=torch.float32)
    latents = latents.to("cpu", dtype=torch.float32)
    
    # Apply GLM-Image specific scaling logic (reverse engineered from pipeline code)
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.latent_channels, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.latent_channels, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std + latents_mean
    else:
        # Standard SD scaling fallback
        latents = latents / vae.config.scaling_factor
        
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    
    # Postprocess
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image

image = generate_image()

if image:
    # show the image
    # show the histogram of PIL image
    if DEBUG_MODE:
        gray_image = image.convert("L")
        hist = gray_image.histogram()
        # print(f"Histogram: {hist}")
        NUM_INFERENCE_STEPS = 5

        import matplotlib.pyplot as plt
        # Plot
        plt.plot(hist)
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel intensity (0–255)")
        plt.ylabel("Frequency")
        plt.show()

    # Save or display the image to the output folder
    # make the output folder if it doesn't exist
    folder = "output"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # add the utc timestamp in seconds to the output file name

    timestamp = int(time())
    output_file = f"{folder}/output_t2i_{timestamp}.png"

    image.save(output_file)
    print(f"Image saved to {output_file}")