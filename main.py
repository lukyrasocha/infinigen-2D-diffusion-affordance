import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

#image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
#mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")
image = load_image("bed.png")
mask = load_image("mask.jpeg")

pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")
image = pipe(
    prompt="a man with blue pants and white shirt sitting on a bed",
    image=image,
    mask_image=mask,
    height=800,
    width=800,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(50)
).images[0]
image.save(f"flux-fill-dev.png")
