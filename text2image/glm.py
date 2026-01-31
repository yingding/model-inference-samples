from applyllm.accelerators import (
    AcceleratorHelper, 
    DirectorySetting,
    TokenHelper,
)
from applyllm.utils import time_func 
from applyllm.pipelines import (
    KwargsBuilder
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


th = TokenHelper(dir_setting=dir_setting, prefix_list=["zai"])
token_kwargs = th.gen_token_kwargs(model_type="zai")

from time import time
import torch, os
from diffusers import DiffusionPipeline

# Check if CUDA is available, otherwise use CPU or MPS (for Mac)
device_map = "auto"
if torch.cuda.is_available():
    device_map = "cuda"
elif torch.backends.mps.is_available():
    device_map = "mps"

# Note: The model "zai-org/GLM-Image" relies on specific implementations in diffusers and transformers.
# Ensure you have the latest versions installed from source as per instructions.

# Load the pipeline
# Using "auto" for device_map to handle different hardware configurations
# bfloat16 is recommended for newer GPUs, float16 or float32 might be needed elsewhere
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

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

pipe = PipelineClass.from_pretrained(
    "zai-org/GLM-Image", 
    torch_dtype=dtype, 
    device_map=device_map,
    trust_remote_code=True, # Often needed for new models
    **token_kwargs,
)

prompt = "A beautifully designed modern food magazine style dessert recipe illustration, themed around a raspberry mousse cake. The overall layout is clean and bright, divided into four main areas: the top left features a bold black title 'Raspberry Mousse Cake Recipe Guide', with a soft-lit close-up photo of the finished cake on the right, showcasing a light pink cake adorned with fresh raspberries and mint leaves; the bottom left contains an ingredient list section, titled 'Ingredients' in a simple font, listing 'Flour 150g', 'Eggs 3', 'Sugar 120g', 'Raspberry puree 200g', 'Gelatin sheets 10g', 'Whipping cream 300ml', and 'Fresh raspberries', each accompanied by minimalist line icons (like a flour bag, eggs, sugar jar, etc.); the bottom right displays four equally sized step boxes, each containing high-definition macro photos and corresponding instructions."

print("Generating image...")
@time_func
def generate_image():
    return pipe(
        prompt=prompt,
        height=1024, # 32 * 32
        width=1152,  # 36 * 32
        num_inference_steps=50,
        guidance_scale=1.5,
        generator=torch.Generator(device=device_map).manual_seed(42) # Use device specific generator for reproducibility
    ).images[0]

image = generate_image()

# Save or display the image to the output folder
# make the output folder if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")
# add the utc timestamp in seconds to the output file name

timestamp = int(time.time())
output_file = f"output/output_t2i_{timestamp}.png"

image.save(output_file)
print(f"Image saved to {output_file}")
