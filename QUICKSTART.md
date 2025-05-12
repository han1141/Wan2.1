# Wan2.1 快速启动指南

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.1">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">技术报告</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">博客</a> &nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">微信群</a>&nbsp&nbsp | &nbsp&nbsp 📖 <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>&nbsp&nbsp
<br>

-----

## 项目简介

**Wan2.1** 是一套全面开放的视频生成基础模型，具有以下核心特点：

- 🔥 **卓越性能**：在多项基准测试中，**Wan2.1** 持续超越现有开源模型和商业解决方案
- 🔥 **支持消费级GPU**：T2V-1.3B模型仅需8.19GB显存，兼容几乎所有消费级GPU，可在RTX 4090上生成5秒480P视频，用时约4分钟
- 🔥 **多任务支持**：擅长文本到视频、图像到视频、视频编辑、文本到图像和视频到音频等多种任务
- 🔥 **视觉文本生成**：首个能够生成中英文文本的视频模型，增强了实际应用场景
- 🔥 **强大的视频VAE**：**Wan-VAE** 提供卓越的效率和性能，可编码和解码任意长度的1080P视频

## 安装指南

### 环境要求

- Python 3.10或更高版本
- PyTorch 2.4.0或更高版本

### 方法一：使用pip安装

```bash
# 克隆仓库
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1

# 安装依赖
pip install -r requirements.txt
```

### 方法二：使用Poetry安装

确保您的系统上已安装[Poetry](https://python-poetry.org/docs/#installation)。

```bash
# 安装所有依赖
poetry install
```

#### 处理`flash-attn`安装问题

如果由于**PEP 517构建问题**导致`flash-attn`安装失败，可以尝试以下解决方法：

##### 无构建隔离安装（推荐）
```bash
poetry run pip install --upgrade pip setuptools wheel
poetry run pip install flash-attn --no-build-isolation
poetry install
```

##### 从Git安装（替代方案）
```bash
poetry run pip install git+https://github.com/Dao-AILab/flash-attention.git
```

## 模型下载

| 模型         | 下载链接                                                                                                                                       | 备注                      |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| T2V-14B      | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)      🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)         | 支持480P和720P |
| I2V-14B-720P | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P) | 支持720P |
| I2V-14B-480P | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) | 支持480P |
| T2V-1.3B     | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)     🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)        | 支持480P |
| FLF2V-14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P)     🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-FLF2V-14B-720P)      | 支持720P |

> 💡注意：
> * 1.3B模型能够生成720P分辨率的视频，但由于在此分辨率下的训练有限，结果通常比480P稳定性差。为获得最佳性能，建议使用480P分辨率。
> * 对于首帧末帧到视频生成，我们的模型主要在中文文本-视频对上训练，因此建议使用中文提示以获得更好的结果。

### 使用huggingface-cli下载模型：
```sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

### 使用modelscope-cli下载模型：
```sh
pip install modelscope
modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./Wan2.1-T2V-14B
```

## 使用指南

### 文本到视频生成 (Text-to-Video)

本项目支持两种文本到视频模型（1.3B和14B）和两种分辨率（480P和720P）。

#### 基本用法（不使用提示扩展）

- 单GPU推理

```sh
python generate.py --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

如果遇到内存不足(OOM)问题，可以使用`--offload_model True`和`--t5_cpu`选项减少GPU内存使用：

```sh
python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

> 💡注意：使用`T2V-1.3B`模型时，建议设置参数`--sample_guide_scale 6`。`--sample_shift`参数可以在8到12的范围内根据性能调整。

- 多GPU推理（使用FSDP + xDiT USP）

```sh
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=8 generate.py --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### 使用提示扩展

扩展提示可以有效丰富生成视频中的细节，进一步提高视频质量。我们提供以下两种提示扩展方法：

- 使用Dashscope API进行扩展

```sh
DASH_API_KEY=your_key python generate.py --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'dashscope' --prompt_extend_target_lang 'zh'
```

- 使用本地模型进行扩展

```sh
python generate.py --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" --use_prompt_extend --prompt_extend_method 'local_qwen' --prompt_extend_target_lang 'zh'
```

### 图像到视频生成 (Image-to-Video)

#### 基本用法（不使用提示扩展）

- 单GPU推理

```sh
python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression."
```

> 💡注意：对于图像到视频任务，`size`参数表示生成视频的面积，宽高比将遵循原始输入图像。

- 多GPU推理（使用FSDP + xDiT USP）

```sh
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=8 generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."
```

#### 使用提示扩展

- 使用本地模型进行扩展

```sh
python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG --use_prompt_extend --prompt_extend_model Qwen/Qwen2.5-VL-7B-Instruct --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."
```

- 使用Dashscope API进行扩展

```sh
DASH_API_KEY=your_key python generate.py --task i2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG --use_prompt_extend --prompt_extend_method 'dashscope' --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."
```

### 首帧末帧到视频生成 (First-Last-Frame-to-Video)

#### 基本用法（不使用提示扩展）

- 单GPU推理

```sh
python generate.py --task flf2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-FLF2V-14B-720P --first_frame examples/flf2v_input_first_frame.png --last_frame examples/flf2v_input_last_frame.png --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings."
```

- 多GPU推理（使用FSDP + xDiT USP）

```sh
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=8 generate.py --task flf2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-FLF2V-14B-720P --first_frame examples/flf2v_input_first_frame.png --last_frame examples/flf2v_input_last_frame.png --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings."
```

#### 使用提示扩展

- 使用本地模型进行扩展

```sh
python generate.py --task flf2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-FLF2V-14B-720P --first_frame examples/flf2v_input_first_frame.png --last_frame examples/flf2v_input_last_frame.png --use_prompt_extend --prompt_extend_model Qwen/Qwen2.5-VL-7B-Instruct --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings."
```

- 使用Dashscope API进行扩展

```sh
DASH_API_KEY=your_key python generate.py --task flf2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-FLF2V-14B-720P --first_frame examples/flf2v_input_first_frame.png --last_frame examples/flf2v_input_last_frame.png --use_prompt_extend --prompt_extend_method 'dashscope' --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings."
```

### 文本到图像生成 (Text-to-Image)

Wan2.1是一个统一的图像和视频生成模型，也可以生成图像：

```sh
python generate.py --task t2i-14B --size 1024*1024 --ckpt_dir ./Wan2.1-T2V-14B --prompt '一个朴素端庄的美人'
```

## 使用Gradio本地演示

```sh
cd gradio

# 文本到视频
DASH_API_KEY=your_key python t2v_14B_singleGPU.py --prompt_extend_method 'dashscope' --ckpt_dir ./Wan2.1-T2V-14B

# 图像到视频
DASH_API_KEY=your_key python i2v_14B_singleGPU.py --prompt_extend_method 'dashscope' --ckpt_dir_720p ./Wan2.1-I2V-14B-720P

# 首帧末帧到视频
DASH_API_KEY=your_key python flf2v_14B_singleGPU.py --prompt_extend_method 'dashscope' --ckpt_dir_720p ./Wan2.1-FLF2V-14B-720P
```

## 使用Diffusers

### 文本到视频

```python
import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# 可用模型: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
flow_shift = 5.0 # 720P用5.0, 480P用3.0
scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = scheduler
pipe.to("cuda")

prompt = "A cat and a dog baking a cake together in a kitchen."
negative_prompt = "Bright tones, overexposed, static, blurred details"

output = pipe(
     prompt=prompt,
     negative_prompt=negative_prompt,
     height=720,
     width=1280,
     num_frames=81,
     guidance_scale=5.0,
    ).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

### 图像到视频

```python
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

# 可用模型: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = load_image("examples/i2v_input.JPG")
max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))
prompt = "A white cat wearing sunglasses sits on a surfboard."
negative_prompt = "Bright tones, overexposed, static, blurred details"

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height, width=width,
    num_frames=81,
    guidance_scale=5.0
).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `--task` | 任务类型，可选：t2v-14B, t2v-1.3B, i2v-14B, flf2v-14B, t2i-14B |
| `--size` | 生成视频的尺寸，格式为宽*高 |
| `--frame_num` | 生成的帧数，默认为81（视频）或1（图像） |
| `--ckpt_dir` | 模型检查点目录 |
| `--offload_model` | 是否将模型卸载到CPU以减少GPU内存使用 |
| `--prompt` | 生成提示文本 |
| `--use_prompt_extend` | 是否使用提示扩展 |
| `--prompt_extend_method` | 提示扩展方法，可选：dashscope, local_qwen |
| `--prompt_extend_target_lang` | 提示扩展目标语言，可选：zh, en |
| `--image` | 图像到视频任务的输入图像路径 |
| `--first_frame` | 首帧末帧到视频任务的首帧图像路径 |
| `--last_frame` | 首帧末帧到视频任务的末帧图像路径 |
| `--sample_steps` | 采样步数，默认为文本到视频50步，图像到视频40步 |
| `--sample_guide_scale` | 采样引导比例，控制生成内容与提示的一致性 |

## 许可协议

本仓库中的模型基于Apache 2.0许可证授权。我们不对您生成的内容主张任何权利，您可以自由使用它们，但请确保您的使用符合本许可证的规定。您对模型的使用必须完全负责，不得涉及分享任何违反适用法律的内容，对个人或群体造成伤害，传播旨在伤害的个人信息，传播错误信息，或针对弱势群体。有关您权利的完整限制和详细信息，请参阅[许可证](LICENSE.txt)全文。

## 联系我们

如果您想向我们的研究或产品团队留言，欢迎加入我们的[Discord](https://discord.gg/AKNgpMK4Yj)或[微信群](https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg)！