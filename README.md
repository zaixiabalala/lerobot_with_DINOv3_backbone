# LeRobot with DINOv3 Backbone

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本仓库是基于 [LeRobot](https://github.com/huggingface/lerobot) 的修改版本，集成了 **DINOv3** 视觉backbone。目前只实现了act的backbone集成。

## 快速开始

### 1. DINOv3 预训练模型
下载 DINOv3 预训练模型（例如，可从 Hugging Face Hub 获取）。本仓库测试使用的是 `dinov3_vitb` 版本。你可以使用我们下载好的权重，或自行申请。

### 2. 配置环境
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg=7.1.1 -c conda-forge
cd lerobot
pip install .
pip install transformers # use dino
pip install peft # use lora
```
### 3. 训练

可以定制你的训练配置，例如可以通过更改config的方式选择你需要的backbone；config的案例在lerobot/train_config中

```bash
python -m lerobot.scripts.train --config_path='/home/student/workspace/ruogu/lerobot/train_config/64_100_2cam_dino.json'
```

本人比较喜欢用于训练的命令（建议ssh使用）
```bash
nohup python -m lerobot.scripts.train --config_path='/home/student/workspace/ruogu/lerobot/train_config/64_100_1cam_dino.json' > /home/student/workspace/ruogu/lerobot_outputs/train/train.out 2>&1 & disown
```

### 4. 部署

下载dinov3 backbone后 仿照以下指令部署：

```python 
import json
import numpy as np
import os

from lerobot.policies.act.modeling_act import ACTPolicy

DATA_PATH = "/home/student/new_data/2cam_org/record_20250917_143323"
PRETRAINED_PATH = "/home/student/workspace/ruogu/lerobot_outputs/train/act_100_2cam_ntem_dumbbell_0930/checkpoints/040000/pretrained_model"
CONFIG_PATH = os.path.join(PRETRAINED_PATH, "config.json")

# 下面实际上是修改了pretrained_model目录下面的config.json文件 是直接修改了文件本身
# 如果想要设置历史加权 也可以这么修改 比如n_action_steps=1 同时temporal_ensemble_coeff=0.01
# 本质上和手动直接改config.json是等价的
config = json.load(open(CONFIG_PATH))
config["dino_model_dir"] = "/home/student/workspace/ruogu/dinov3-vits-1/dinov3-vitb16-pretrain-lvd1689m"
json.dump(config, open(CONFIG_PATH, "w"))

policy = ACTPolicy.from_pretrained(PRETRAINED_PATH)
policy.eval()

print(policy.config)

# 请记得归一化之后再构建batch!
# batch = {
#             "observation.state": torch.from_numpy(angle).float().unsqueeze(0).to(device),
#             "observation.images.cam_0": torch.from_numpy(cam0_image).float().permute(2, 0, 1).unsqueeze(0).to(device),
#             "observation.images.cam_1": torch.from_numpy(cam1_image).float().permute(2, 0, 1).unsqueeze(0).to(device),
#         }
# action = policy.select_action(batch).cpu().numpy()
```

注意这里的lerobot是以包的形式被调用 并不需要拷贝代码

### 许可与归属（License & Attribution）

- 本仓库基于 LeRobot 项目进行修改与扩展。原 LeRobot 项目代码版权归其原作者所有，并遵循 MIT License 发布。

- 你可以在遵守 MIT 许可证条款的前提下，自由地使用、复制、修改、合并、发布、分发、再许可及销售本仓库的副本。

- 你的唯一义务是在本仓库的所有副本或重要部分中，包含上述版权声明和本许可声明。

- 本仓库中引入的修改与新增功能（如 DINOv3 集成、配置与脚本）由本仓库维护者提供。

- 第三方组件（如 DINOv3 预训练模型）受其各自许可协议的约束，请确保你遵守了相应的使用条款。

- 此许可是版权许可，并非商标许可。未经明确授权，不得使用原项目贡献者的名称、商标、服务标志或标识来推广你的衍生作品。
