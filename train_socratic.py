import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"

# ══════════════════════════════════════════
# 1. TOKENIZER Y MODELO
# ══════════════════════════════════════════

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

print("Cargando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)

# ══════════════════════════════════════════
# 2. LoRA
# ══════════════════════════════════════════

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ══════════════════════════════════════════
# 3. DATASET
# ══════════════════════════════════════════

dataset = load_dataset(
    "json",
    data_files="dataset_socratico.jsonl",
    split="train"
)

def format_example(example):
    text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_example)

# ══════════════════════════════════════════
# 4. TRAINING ARGUMENTS
# ══════════════════════════════════════════

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    output_dir="./checkpoints_socratic",
    report_to="none",
    fp16=False,
)

# ══════════════════════════════════════════
# 5. TRAINER (TRL 0.8.6)
# ══════════════════════════════════════════

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=1024,
    args=training_args,
)

print("Iniciando entrenamiento...")
trainer.train()

# ══════════════════════════════════════════
# 6. GUARDAR
# ══════════════════════════════════════════

model.save_pretrained("./qwen-math-socratic-lora")
tokenizer.save_pretrained("./qwen-math-socratic-lora")

print("Modelo guardado en ./qwen-math-socratic-lora")