# Lumina-DiMOO
[Project](https://synbol.github.io/Lumina-DiMOO/) / [GitHub](https://github.com/Alpha-VLLM/Lumina-DiMOO/) / [Model](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO)

Lumina-DiMOO is a discrete-diffusion omni-modal foundation model unifying generation and understanding. This implementation integrates a Lumina-DiMOO switch for T2I, I2I editing, and MMU.

#### Key features

- **Unified Discrete Diffusion Architecture**: Employs a fully discrete diffusion framework to process inputs and outputs across diverse modalities.
- **Versatile Multimodal Capabilities**: Supports a wide range of multimodal tasks, including text-to-image generation (arbitrary and high-resolution), image-to-image generation (e.g., image editing, subject-driven generation, inpainting), and advanced image understanding.
- **Higher Sampling Efficiency**:  Outperforms previous autoregressive (AR) or hybrid AR-diffusion models with significantly faster sampling. A custom caching mechanism further boosts sampling speed by up to 2×.


### Example Usage

The Lumina-DiMOO pipeline provides three core functions — T2I, I2I, and MMU.
For detailed implementation examples and creative applications, please visit the [GitHub](https://github.com/Alpha-VLLM/Lumina-DiMOO)


#### Text-to-Image
**prompt**             |  **image** 
:-------------------------:|:-------------------------:
| "A striking photograph of a glass of orange juice on a wooden kitchen table, capturing a playful moment. The orange juice splashes out of the glass and forms the word \"Smile\" in a whimsical, swirling script just above the glass. The background is softly blurred, revealing a cozy, homely kitchen with warm lighting and a sense of comfort." | <img width="1536" height="768" alt="20251021-220001" src="https://github.com/user-attachments/assets/028ca447-b837-407b-8f4c-e546a1684cff" />


```python
import torch

from diffusers import VQModel, DiffusionPipeline
from transformers import AutoTokenizer

vqvae = VQModel.from_pretrained("Alpha-VLLM/Lumina-DiMOO", subfolder="vqvae").to(device='cuda', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Alpha-VLLM/Lumina-DiMOO", trust_remote_code=True)

pipe = DiffusionPipeline.from_pretrained(
    "Alpha-VLLM/Lumina-DiMOO",
    vqvae=vqvae,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    # use local custom pipeline until it’s merged upstream to diffusers
    custom_pipeline="path/to/diffusers/examples/community/lumina_dimoo.py",
)
pipe.to("cuda")

prompt = '''A striking photograph of a glass of orange juice on a wooden kitchen table, capturing a playful moment. The orange juice splashes out of the glass and forms the word \"Smile\" in a whimsical, swirling script just above the glass. The background is softly blurred, revealing a cozy, homely kitchen with warm lighting and a sense of comfort.'''

img = pipe(
    prompt=prompt,
    task="text_to_image",
    height=768,
    width=1536,
    num_inference_steps=64,
    cfg_scale=4.0,     
    use_cache=True,
    cache_ratio=0.9, 
    warmup_ratio=0.3,
    refresh_interval=5
).images[0]

img.save("t2i_test_output.png")
```

#### Image-to-Image
**prompt**             |  **image_before**   |  **image_after**  
:-------------------------:|:-------------------------:|:-------------------------:
| "A functional wooden printer stand.Nestled next to a brick wall in a bustling city street, it stands firm as pedestrians hustle by, illuminated by the warm glow of vintage street lamps." | ![20251021-215950](https://github.com/user-attachments/assets/cbf5a85d-e7c6-40d7-9557-0d05cc6ec67b) | <img width="512" height="512" alt="20251021-220007" src="https://github.com/user-attachments/assets/61ea3c43-dc98-4c85-b652-3f376ab4131b" /> |

```python
import torch

from diffusers import VQModel, DiffusionPipeline
from transformers import AutoTokenizer
from diffusers.utils import load_image

vqvae = VQModel.from_pretrained("Alpha-VLLM/Lumina-DiMOO", subfolder="vqvae").to(device='cuda', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Alpha-VLLM/Lumina-DiMOO", trust_remote_code=True)

pipe = DiffusionPipeline.from_pretrained(
    "Alpha-VLLM/Lumina-DiMOO",
    vqvae=vqvae,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    # use local custom pipeline until it’s merged upstream
    custom_pipeline="path/to/diffusers/examples/community/lumina_dimoo.py",
)
pipe.to("cuda")

input_image = load_image(
    "https://raw.githubusercontent.com/Alpha-VLLM/Lumina-DiMOO/main/examples/example_2.jpg"
).convert("RGB")

prompt = "A functional wooden printer stand.Nestled next to a brick wall in a bustling city street, it stands firm as pedestrians hustle by, illuminated by the warm glow of vintage street lamps."

img = pipe(
    prompt=prompt,
    image=input_image,
    task="image_to_image"
    edit_type="depth_control",
    num_inference_steps=64,
    temperature=1.0,
    cfg_scale=2.5,
    cfg_img=4.0,
).images[0]

img.save("i2i_test_output.png")

```


#### Multimodal Understanding
**question**       |   **image**      |   **answer** 
:-------------------------:|:-------------------------:|:-------------------------:
| "Please describe the image." | <img width="1820" height="1024" alt="20251021-220409" src="https://github.com/user-attachments/assets/e133c679-3261-400b-9c61-57f1f6e62ec0" /> | "The image shows a vibrant orange sports car parked in a showroom. The car has a sleek, aerodynamic design with a prominent front grille and side vents. The body is adorned with black and orange racing stripes, creating a striking contrast against the orange paint. The car is equipped with black alloy wheels and a low-profile body style. The background features a white wall with a large emblem that reads "BREITZEN" and includes a silhouette of a horse and text. The floor is tiled with dark tiles, and the showroom is well-lit, highlighting the car. The overall setting suggests a high-end, possibly luxury, automotive environment."|


```python
import torch

from diffusers import VQModel, DiffusionPipeline
from transformers import AutoTokenizer
from diffusers.utils import load_image

vqvae = VQModel.from_pretrained("Alpha-VLLM/Lumina-DiMOO", subfolder="vqvae").to(device='cuda', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Alpha-VLLM/Lumina-DiMOO", trust_remote_code=True)

pipe = DiffusionPipeline.from_pretrained(
    "Alpha-VLLM/Lumina-DiMOO",
    vqvae=vqvae,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    # use local custom pipeline until it’s merged upstream
    custom_pipeline="path/to/diffusers/examples/community/lumina_dimoo.py",
)
pipe.to("cuda")

question = "Please describe the image."

input_image = load_image(
    "https://raw.githubusercontent.com/Alpha-VLLM/Lumina-DiMOO/main/examples/example_8.png"
).convert("RGB")

out = pipe(
    prompt=question,
    image=input_image,
    task="multimodal_understanding",
    num_inference_steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
)

text = getattr(out, "text", out)
with open("mmu_answer.txt", "w", encoding="utf-8") as f:
    f.write(text.strip() + "\n")
```
