import os
import numpy as np
import requests
import torch
import copy
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import gradio as gr


# Define the URL and the output path
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
output_dir = "models"
output_file = os.path.join(output_dir, "sam_vit_h_4b8939.pth")

# Create the models directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Download the file
response = requests.get(url)
if response.status_code == 200:
    with open(output_file, 'wb') as f:
        f.write(response.content)
    print(f"Model weights downloaded to: {output_file}")
else:
    print(f"Failed to download model weights. Status code: {response.status_code}")

# Configuration
target_width, target_height = 512, 512
model_dir = "stabilityai/stable-diffusion-2-inpainting"
sam_checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda"

# Load models
scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_dir,
    scheduler=scheduler,
    revision="fp16",
    torch_dtype=torch.float16
).to(device)
pipe.enable_xformers_memory_efficient_attention()

sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.999,
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

def show_anns(anns, original_image):
    if len(anns) == 0:
        return original_image
    sorted_anns = sorted(enumerate(anns), key=lambda x: x[1]['area'], reverse=True)
    ax = plt.gca()
    ax.imshow(original_image)

    for original_idx, ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]
        img = np.ones((*m.shape, 3))

        for i in range(3):
            img[:, :, i] = color_mask[i]

        ax.imshow(np.dstack((img, m * 0.35)))

        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                ax.text(cx, cy, str(original_idx), color='white', fontsize=16, ha='center', va='center', fontweight='bold')

    plt.axis('off')
    plt.savefig('static/masks.jpg')
    return 'static/masks.jpg'

def generate_images(image, prompts):
    source_image = Image.open(image).convert("RGB")
    seg = np.array(source_image)
    masks = mask_generator.generate(seg)
    
    # Display masks
    mask_image_path = show_anns(masks, seg)
    
    encoded_images = []
    for prompt in prompts:
        stable_diffusion_mask = Image.fromarray(masks[int(prompt)]['segmentation'])
        generator = torch.Generator(device="cuda").manual_seed(77)
        image_result = pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=60, generator=generator, image=source_image, mask_image=stable_diffusion_mask).images[0]
        encoded_images.append(image_result)
    
    return mask_image_path, encoded_images

def gradio_interface(image, prompts):
    mask_image_path, generated_images = generate_images(image, prompts)
    return mask_image_path, [img for img in generated_images]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Image Segmentation and Generation")
    
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Image")
        prompt_input = gr.Textbox(label="Enter Prompts (comma-separated)", placeholder="e.g., 'a cat', 'a dog'")
    
    submit_button = gr.Button("Generate")
    output_mask = gr.Image(label="Masks")
    output_images = gr.Gallery(label="Generated Images").style(grid=2)
    
    submit_button.click(gradio_interface, inputs=[image_input, prompt_input], outputs=[output_mask, output_images])

# Launch the Gradio app
demo.launch()
