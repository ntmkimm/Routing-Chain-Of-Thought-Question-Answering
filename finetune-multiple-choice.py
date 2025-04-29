import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import os
os.environ["CUDA_VISIBLE_DEVICES"]="7,6"

# Initialize the distributed environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12351'  # Use a free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32768)

# ==== FREE ORIGIN WEIGHTS

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float16)

# model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float16)
model.lm_head = CastOutputToFloat(model.lm_head)

# ====== SET UP  LoRA Adapter

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=16, #alpha scaling
    target_modules=[
          "q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj",
      ],
    # target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # modules_to_save=["lm_head", "embed_token"], 
    # lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM", # SEQ_CLS, SEQ_2_SEQ_LM, CAUSAL_LM, TOKEN_CLS, QUESTION_ANS, FEATURE_EXTRACTION,
    fan_in_fan_out=False,
)

model = get_peft_model(model, config)

# ===== data processing
import json
from datasets import Dataset, DatasetDict
data_file = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/TRNS-AI/data/train_v1.json"

# data = load_dataset("json", data_files=data_file, split="train")
# Open the file and load the JSON data
with open(data_file, 'r') as f:
    data = json.load(f)

# Convert to dictionary format (column names as keys, values as lists)
data_dict = {
    "premises-NL": [item["premises-NL"] for item in data],
    "premises-FOL": [item["premises-FOL"] for item in data],
    "questions": [item["questions"] for item in data],
    "answers": [item["answers"] for item in data],
    "idx": [item["idx"] for item in data],
    "explanation": [item["explanation"] for item in data]
}

flattened = []
num_examples = len(data_dict["questions"])

for i in range(num_examples):
    premises    = data_dict["premises-NL"][i]
    questions   = data_dict["questions"][i]
    answers     = data_dict["answers"][i]
    explanations= data_dict["explanation"][i]

    for q, a, e in zip(questions, answers, explanations):
        # strip leading/trailing whitespace, then look at first char
        if a and a.strip()[0] in {"A", "B", "C", "D"}:
            flattened.append({
                "premises-NL": premises,
                "question":    q,
                "answer":      a,
                "explanation": e
            })

# Now, convert the data_dict to a Dataset object
from datasets import Dataset

dataset = Dataset.from_list(flattened)


system_prompt = (
    "You are a logician expert with advanced knowledge in reasoning, diagnostics, "
    "and treatment planning. You first think through the reasoning process step‐by‐step "
    "in your mind and then provide the user with the answer."
)

def format_sample(sample):
    # 1) Split the raw question into lines:
    #    line[0] = the stem
    #    line[1:] = the choice texts in order A, B, C, (maybe D)
    lines = sample["question"].split("\n")
    premises = str(sample["premises-NL"])
    stem    = lines[0].strip()
    choices = [line.strip() for line in lines[1:] if line.strip()]
    explanation = str(sample["explanation"])
    answer = sample["answer"]

    # 2) Map them to letters A, B, C, (D)
    letters = ["A", "B", "C", "D"]
    opts = { letters[i]: choices[i] for i in range(len(choices)) }

    # 3) Build a dynamic choices block
    choices_block = "\n".join(f"{opts[ltr]}" for ltr in opts)

    # 4) Build your user prompt text
    user_prompt = (
        f"### Context ###\n {premises}\n"
        "Carefully read the contexts and strictly follow to the rules: select the best answer to the question based on the provided contexts.\n\n"
        "### Question ###\n"
        f"{stem}\n\n"
        "### Choices ###\n"
        f"{choices_block}\n\n"
        "Show your reasoning in <reasoning>…</reasoning> tags,\n"
        "and your final selected option letter in <answer>…</answer> tags.\n"
    )


    # 5) Apply your chat template
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
            {"role": "assistant", "content": f"Let me think step by step.\n<reasoning>\n{str(explanation)}\n</reasoning>\n<answer>\n{answer}\n</answer>\n"}
        ],
        tokenize=False,
        continue_final_message=True
    )
    print(prompt)
    return {"prompt": prompt}

dataset=dataset.map(format_sample)


optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

lr_scheduler = transformers.get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)

training_args = SFTConfig(
    do_eval=True,
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=8,
    warmup_steps=100, 
    max_steps=1000, 
    learning_rate=1e-5, 
    fp16=True,
    logging_steps=1, 
    seed=42,
    save_strategy="steps",  # Save checkpoints
    save_steps=200,  # Adjust based on trainings
    output_dir="./model/qwen-multiple-choice",
    dataset_text_field="prompt",
)


trainer = SFTTrainer(
    model=model, 
    train_dataset=dataset,
    # eval_dataset=val_dataset,
    args=training_args,
    # formatting_func=formatting_prompts_func,
    optimizers=(optimizer, lr_scheduler),
    # data_collator=collator, # adding for question answering task
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

save_directory = "./model/qwen-multiple-choice"  # Choose your directory

# Save model and tokenizer locally
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory, safe_serialization=True)

print(f"Model saved locally at: {save_directory}")