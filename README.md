# Fine-tuning Qwen 2.5 for Chain of Thought Reasoning

This project focuses on fine-tuning [Qwen/Qwen2.5](https://huggingface.co/Qwen) using Chain of Thought (CoT) prompting for both classification and multiple-choice reasoning tasks.  
For solving math problems, we can use pre-trained models like **Qwen2.5-7B-Instruct** or **Qwen/Qwen2.5-Math-7B-Instruct**.  
For multiple-choice and Yes/No questions, we fine-tune **Qwen2.5-1.5B-Instruct**.

## Requirements

Set up the environment with **conda**:

```bash
conda create -n qwen-cot python=3.10
conda activate qwen-cot

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft trl accelerate numpy
```

The code will automatically download models from Hugging Face (no auth-token needed),  
or you can manually download from [huggingface.co/Qwen](https://huggingface.co/Qwen).

## Project Structure

| File                         | Description                                        |
|-------------------------------|----------------------------------------------------|
| `train_v1.json`               | Dataset in TRNS-AI challenage                      |
| `solver-routing.py`           | Inference engine that selects models and generates chain-of-thought output. |
| `finetune-classify.py`        | Fine-tuning script for Yes/No classification.       |
| `finetune-multiple-choice.py` | Fine-tuning script for multiple-choice solving.     |
| `/model/qwen-classify/`       | Fine-tuned classification model directory.          |
| `/model/qwen-multiple-choice/`| Fine-tuned multiple-choice model directory.         |
| `/output`                     | Output directory for inference.                     |


## Pre-trained Models Used

**Base Model**: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (for fine-tuning)

Optional models for math tasks:

- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen/Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct/tree/main)


## Fine-tuning Instructions

We use **LoRA (Low-Rank Adaptation)** for efficient fine-tuning.

### 1. Yes/No Classification Fine-tuning with CoT Reasoning
Output directory: `./model/qwen-classify/`

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune-classify.py
```

### 2. Multiple-Choice Fine-tuning with CoT Reasoning
Output directory: `./model/qwen-multiple-choice/`

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune-multiple-choice.py
```


## Chain of Thought (CoT)

Improves logical thinking.
Provides interpretable and verifiable reasoning.
Reduces hallucination on complex questions.

**Format**

```text
<reasoning>
Step-by-step detailed reasoning here...
</reasoning>
<answer>
Final answer (Yes, No, A, B, C, etc.)
</answer>
```


## Inference

After fine-tuning, run the inference script:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python solver-routing.py
```

Automatically select the correct model (classification or multiple-choice or math solving),
Generate chain-of-thought reasoning and final answers save to `/output`.
**Inputs** for inferencing one sample are **a list of premises** and **a list of questions** to answer based on the provided premises.

## Notes

- Fine-tuning uses **LoRA** with alpha=16: only a small number of parameters are updated.
- Supports multi-GPU training (minor setup change needed).
- Preprocessing automatically formats input prompts into Chain of Thought style.


## Credits

- Pretrained models from [Qwen](https://huggingface.co/Qwen).
- Chain of Thought prompting idea from [Wei et al., 2022](https://arxiv.org/abs/2201.11903).
- LoRA fine-tuning method based on [Hu et al., 2021](https://arxiv.org/abs/2106.09685).

