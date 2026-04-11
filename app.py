import argparse

from PIL import Image
import gradio as gr
from imagenet_en_cn import IMAGENET_1K_CLASSES

import torch

from pixelflow.utils.misc import seed_everything
from pixelflow.pipeline_pixelflow import PixelFlowPipeline


parser = argparse.ArgumentParser(description='Gradio Demo', add_help=False)
parser.add_argument('--checkpoint', type=str, help='checkpoint folder path')
parser.add_argument('--class_cond', action='store_true', help='use class conditional generation')
args = parser.parse_args()

local_rank = 0
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)

pipeline = PixelFlowPipeline.from_pretrained(
    args.checkpoint, local_files_only=True, text_cond=not args.class_cond, dtype=torch.bfloat16, device=device
)

if args.class_cond:
    resolution = 256
    NUM_EXAMPLES = 4
else:
    resolution = 1024
    NUM_EXAMPLES = 1

def infer(use_ode_dopri5, noise_shift, cfg_scale, class_label, seed, *num_steps_per_stage):
    seed_everything(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        samples = pipeline(
            prompt=[class_label] * NUM_EXAMPLES,
            height=resolution,
            width=resolution,
            num_inference_steps=list(num_steps_per_stage),
            guidance_scale=cfg_scale,         # The guidance for the first frame, set it to 7 for 384p variant
            device=device,
            shift=noise_shift,
            use_ode_dopri5=use_ode_dopri5,
        )
    samples = (samples * 255).round().astype("uint8")
    samples = [Image.fromarray(sample) for sample in samples]
    return samples


with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center'>PixelFlow: Pixel-Space Generative Models with Flow</h1>")

    with gr.Tabs():
        with gr.TabItem('Generate'):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        if args.class_cond:
                            user_input = gr.Dropdown(
                                list(IMAGENET_1K_CLASSES.values()),
                                value='daisy [雏菊]',
                                type="index", label='ImageNet-1K Class'
                            )
                        else:
                            # text input
                            user_input = gr.Textbox(label='Enter your prompt', show_label=False, max_lines=1, placeholder="Enter your prompt",)
                    ode_dopri5 = gr.Checkbox(label="Dopri5 ODE", info="Use Dopri5 ODE solver")
                    noise_shift = gr.Slider(minimum=1.0, maximum=100.0, step=1, value=1.0, label='Noise Shift')
                    cfg_scale = gr.Slider(minimum=1, maximum=25, step=0.1, value=4.0, label='Classifier-free Guidance Scale')
                    num_steps_per_stage = []
                    for stage_idx in range(pipeline.scheduler.num_stages):
                        num_steps = gr.Slider(minimum=1, maximum=100, step=1, value=10, label=f'Num Inference Steps (Stage {stage_idx})')
                        num_steps_per_stage.append(num_steps)
                    seed = gr.Slider(minimum=0, maximum=1000, step=1, value=42, label='Seed')
                    button = gr.Button("Generate", variant="primary")
                with gr.Column():
                    output = gr.Gallery(label='Generated Images', height=700)
                    button.click(infer, inputs=[ode_dopri5, noise_shift, cfg_scale, user_input, seed, *num_steps_per_stage], outputs=[output])
    demo.queue()
    demo.launch(share=False, debug=True)
