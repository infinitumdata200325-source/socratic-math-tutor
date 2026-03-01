# train_socratic.py — Qwen2.5-Math-7B → Tutor Socrático
# Stack: HuggingFace + PEFT + TRL (sin Unsloth, compatible CPU)

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch

MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"

# ══════════════════════════════════════════
# 1. CARGAR MODELO Y TOKENIZER
# ══════════════════════════════════════════
print("Cargando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # CPU requiere float32
    load_in_4bit=True,           # QLoRA
    device_map="cpu",
    trust_remote_code=True,
)

# ══════════════════════════════════════════
# 2. CONFIGURAR LORA
# ══════════════════════════════════════════
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ══════════════════════════════════════════
# 3. CARGAR DATASET
# ══════════════════════════════════════════
def format_example(example):
    msgs = example["conversations"]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = load_dataset("json",
    data_files="dataset_socratico.jsonl", split="train")
dataset = dataset.map(format_example)

# ══════════════════════════════════════════
# 4. ENTRENAR
# ══════════════════════════════════════════
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=20,
        optim="adamw_torch",
        save_steps=100,
        save_total_limit=2,
        output_dir="./checkpoints_socratic",
        report_to="none",
    ),
)

print("Iniciando entrenamiento...")
trainer.train()

# ══════════════════════════════════════════
# 5. GUARDAR
# ══════════════════════════════════════════
model.save_pretrained("./qwen-math-socratic-lora")
tokenizer.save_pretrained("./qwen-math-socratic-lora")
print("Modelo guardado en ./qwen-math-socratic-lora")