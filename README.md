# 🚀 AGNOSTOS: Exploring the Limits of Vision-Language-Action Manipulations in Cross-task Generalization

[[🌐 Project Page]](https://jiaming-zhou.github.io/AGNOSTOS/)  |  [[📄 Paper]](https://arxiv.org/pdf/2505.15660)  |  [🤗 Huggingface Data](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS)  |  [🤗 Huggingface Model](https://huggingface.co/Jiaming2472/X-ICM)  |  [[📺 Video]](https://youtu.be/5MKlijK1gKI)

[Jiaming Zhou](https://jiaming-zhou.github.io/)<sup>1</sup>, [Ke Ye](https://yipko.com/about/)<sup>1</sup>, [Jiayi Liu](https://www.jiayi-liu.cn/)<sup>1</sup>, [Teli Ma](https://teleema.github.io/)<sup>1</sup>, [Zifan Wang](https://zifanw.notion.site/)<sup>1</sup>, [Ronghe Qiu](https://github.com/ConnerQiu)<sup>1</sup>, [Kun-Yu Lin](https://kunyulin.github.io/)<sup>2</sup>, [Zhilin Zhao](https://lawliet-zzl.github.io/)<sup>3</sup>, [Junwei Liang](https://junweiliang.me/)<sup>1,4</sup>

<sup>1</sup>HKUST (Guangzhou), <sup>2</sup>HKU, <sup>3</sup>SYSU, <sup>4</sup>HKUST


## 📝 Overview
The project introduces AGNOSTOS, a simulation manipulation benchmark designed to rigorously evaluate cross-task zero-shot generalization of Vision-Language-Action models, and proposes Cross-Task In-Context Manipulation (X-ICM), a method that significantly improves cross-task generalization capabilities.

<img src="docs/agnostos_benchmark.gif" width="100%">

## 🔧 Environment Setup

### 🐳 Option 1: Using Docker
Please refer to [INSTALL_docker.md](./docs/INSTALL_docker.md) to initialize your environment.

### ⚙️ Option 2: Local Installation

For simplified installation using modern package management, we recommend **Pixi**. 
Install it via the [official guide](https://pixi.sh/latest/#installation), and you can set up dependencies with minimal commands:

```bash
git clone https://github.com/jiaming-zhou/X-ICM.git && cd X-ICM
pixi shell  # Install dependencies and enter virtual environment
pixi run setup_env  # Install additional system dependencies (like xvfb, CoppeliaSim and flash-attention etc.)
```

Inside the Pixi shell, you can run additional tasks. For more options, run `pixi run --list`.


## 📊 AGNOSTOS Benchmark Data
The benchmark consists of two parts. To download the data, use the following Pixi tasks:

```bash
pixi run get_seen_tasks  # Downloads and extracts 18 seen tasks (140G)
pixi run get_unseen_tasks  # Downloads and extracts 23 unseen tasks (20.2GB)
```

Data will be placed in the `data/` directory. For manual download instructions, see [MANUAL_DATA_DOWNLOAD.md](./docs/MANUAL_DATA_DOWNLOAD.md).

## 🤖 X-ICM Method

### 1️⃣ Model Download
To download the pre-trained dynamics diffusion model, run:

```bash
pixi run get_model
```

The model will be extracted to `data/dynamics_diffusion/`. For manual download instructions, see [MANUAL_DATA_DOWNLOAD.md](./docs/MANUAL_DATA_DOWNLOAD.md).

### 2️⃣ Evaluation
Run the evaluation using Pixi task with the below parameters. (You can also run `bash eval_scripts/eval_XICM.sh` directly in the Pixi shell as an alternative to `pixi run eval_xicm`.)
```bash
### set seed numbers for different runs
seeds: [example: "0,99"]
### set the number of rollouts for each run
episodes: [example: 25]
### set the method of LLM
modelname: [example: Qwen2.5.7B.instruct]
### set the number of cross-task in-context samples
num_icls: [example: 18]
### set the gpu list
gpu_ids: [example: 0,1]
### set the in-context sample selection method
ranking_method: [example: "lang_vis.out"]
```

For dynamics-guided in-context manipulation, run:
```bash
pixi run eval_xicm "0,99" 25 Qwen2.5.7B.instruct 18 0,1 "lang_vis.out"
```
**Reminder**: During evaluation, you need to load the Stable-Diffusion model and Qwen-LLM models from huggingface.
You can manually download them from huggingface and load them from the local paths in [`load_weight func`](./main.py#L132) and [`model_path`](./rlbench_inference_dynamics_diffusion.py#L136), accordingly.

For random selection of cross-task samples, run:
```bash
pixi run eval_xicm "0,99" 25 Qwen2.5.7B.instruct 18 0,1 "random"
```

After testing, you can use [`gather_score.py`](./gather_score.py) to collect and analyze the results.

## 🎯 Benchmarking Results over all 23 unseen tasks
We provide the testing results of our `X-ICM (7B)` and `X-ICM (72B)` models under the sub-folder `logs/`.
- X-ICM (7B) achieves 23.5% average success rate and X-ICM (72B) achieves 30.1% average success rate, both versions outperform all existing VLA models;
- X-ICM (7B) fails on only two tasks, while X-ICM (72B) succeeds on all tasks;

## 🔬 Benchmarking Your VLA Model on AGNOSTOS

### 1️⃣ Fine-tuning
Due to the embodiment gap, existing VLA models need to be fine-tuned on RLBench data. 

Please follow your VLA model's fine-tuning guidelines to fine-tune your models on our 18 seen tasks, and then test the models on our 23 unseen tasks.

### 2️⃣ Testing Fine-tuned VLA Models
Modify the [`custom_agent.py`](./custom_agent.py) file:
1. Load your VLA model in the [`load_weights`](./custom_agent.py#L116) function;
2. Implement VLA model inference in the [`_inference`](./custom_agent.py#L21) function, including input construction and output format conversion;
3. Run the evaluation:
    ```bash
    bash eval_CustomModel.sh seeds episodes gpu_ids
    ```

    Example:
    ```bash
    bash eval_scripts/eval_CustomModel.sh "0,99" 25 0,1
    ```

💡 Note: Different VLA models may require different input image sizes (default is 256x256). 

💡 Please modify [`IMAGE_SIZE`](./main_custom.py#L32) in [`main_custom.py`](./main_custom.py) accordingly.



## 🙏 Acknowledgments
This repository is built upon the [RoboPrompt](https://github.com/davidyyd/roboprompt). Some resources from [RVT](https://github.com/NVlabs/RVT) and [RLBench](https://github.com/stepjam/RLBench) are used in this work.

## 📄 Citation
If you find our work helpful to your research, please kindly give us a star and cite our paper.
```bibtex
@article{zhou2025exploring,
    title={Exploring the Limits of Vision-Language-Action Manipulations in Cross-task Generalization},
    author={Zhou, Jiaming and Ye, Ke and Liu, Jiayi and Ma, Teli and Wang, Zifang and Qiu, Ronghe and Lin, Kun-Yu and Zhao, Zhilin and Liang, Junwei},
    journal={arXiv preprint arXiv:2505.15660},
    year={2025}
}
```
