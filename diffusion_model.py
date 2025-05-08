import os
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from latent_proccess import pil_to_latents, latents_to_pil, text_enc
from torch.cuda.amp import autocast

# Initialize scheduler and model
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)
scheduler.set_timesteps(50)

unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16
).to("cuda")

def prompt_2_img(prompts, g=7.5, seed=100, steps=100, dim=512, save_int=True):
    """
    Diffusion process to convert prompt to image
    """
    # Batch size
    bs = len(prompts)

    # Get embeddings
    text = text_enc(prompts)
    uncond = text_enc([""] * bs, text.shape[1])
    emb = torch.cat([uncond, text])

    # Set seed
    if seed:
        torch.manual_seed(seed)

    # Random noise latents
    latents = torch.randn((bs, unet.in_channels, dim // 8, dim // 8))
    latents = latents.to("cuda").half() * scheduler.init_noise_sigma

    # Prepare folder to save intermediate outputs
    save_dir = "tmp"
    if save_int:
        os.makedirs(save_dir, exist_ok=True)

    print("Processing text prompts:", prompts)
    print("Visualizing initial latents...")
    latents_norm = torch.norm(latents.view(latents.shape[0], -1), dim=1).mean().item()
    print(f"Initial Latents Norm: {latents_norm}")

    scheduler.set_timesteps(steps)

    for i, ts in enumerate(tqdm(scheduler.timesteps)):
        inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)

        with torch.no_grad():
            u, t = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)

        pred = u + g * (t - u)
        latents = scheduler.step(pred, ts, latents).prev_sample

        latents_norm = torch.norm(latents.view(latents.shape[0], -1), dim=1).mean().item()
        print(f"Step {i+1}/{steps} Latents Norm: {latents_norm}")

        if save_int and i % 20 == 0:
            img = latents_to_pil(latents)[0]
            image_path = os.path.join(save_dir, f"la_{i:04d}.jpeg")
            img.save(image_path)
            display(img)

    return latents_to_pil(latents)

def visualize_steps(folder='tmp'):
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist.")
        return

    files = os.listdir(folder)
    image_files = sorted([file for file in files if file.endswith('.jpeg')])
    num_steps = len(image_files)

    if num_steps == 0:
        print(f"No images found in '{folder}'")
        return

    fig, axs = plt.subplots(1, num_steps, figsize=(3 * num_steps, 5))
    if num_steps == 1:
        axs = [axs]

    for ax, img_file in zip(axs, image_files):
        img_path = os.path.join(folder, img_file)
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    images = prompt_2_img(["A dog wearing a hat"], save_int=True)
    for img in images:
        display(img)

    visualize_steps()