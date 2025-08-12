import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionInpaintPipeline
from PIL import Image
import gradio as gr

# --- Model Setup ---
# Use cuda for Kaggle's NVIDIA GPUs
text2img_pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    use_safetensors=True
)
text2img_pipeline.to("cuda") # CHANGED FROM "mps"
text2img_pipeline.enable_attention_slicing()

inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
)
inpaint_pipeline.to("cuda") # CHANGED FROM "mps"


# --- Core Functions ---
def generate_image(prompt):
    m_prompt = f"white background, front facing, {prompt}"
    # The 'generator' argument helps ensure reproducibility if needed
    generator = torch.Generator(device="cuda").manual_seed(42)
    generated_image = text2img_pipeline(m_prompt, generator=generator).images[0]
    generated_image.save("generated_image.png")
    return generated_image

def modify_image(original_image, mask_image, modification_prompt):
    try:
        # The 'generator' argument helps ensure reproducibility
        generator = torch.Generator(device="cuda").manual_seed(42)
        output = inpaint_pipeline(
            prompt=modification_prompt,
            image=original_image,
            mask_image=mask_image,
            generator=generator
        ).images[0]

        output.save("modified_output.png")
        return output

    except Exception as e:
        return f"Error: {e}"


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## AI Image Generator & Modifier")

    with gr.Row():
        prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Describe the image you want...")
        generate_button = gr.Button("Generate Image")

    generated_image_output = gr.Image(label="Generated Image")

    gr.Markdown("### Upload images for modification")

    with gr.Row():
        original_image_input = gr.Image(label="Upload Original Image", type="pil")
        mask_image_input = gr.Image(label="Upload Mask Image", type="pil")

    with gr.Row():
        modification_prompt_input = gr.Textbox(label="Enter modification description", placeholder="Describe the changes...")
        modify_button = gr.Button("Apply Modification")

    modified_output = gr.Image(label="Modified Image")

    generate_button.click(generate_image, inputs=prompt_input, outputs=generated_image_output)
    modify_button.click(
        modify_image,
        inputs=[original_image_input, mask_image_input, modification_prompt_input],
        outputs=modified_output
    )

demo.launch(share=True)