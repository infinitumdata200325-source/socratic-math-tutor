import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"

# Cuantización 4-bit (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",          # ahora sí usa la RTX 5050
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)  # prepara el modelo cuantizado para LoRA

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Dataset (igual que antes)
dataset = load_dataset("json", data_files="dataset_socratico.jsonl", split="train")
def format_example(example):
    text = tokenizer.apply_chat_template(example["conversations"], tokenize=False, add_generation_prompt=False)
    return {"text": text}
dataset = dataset.map(format_example)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,   # CLAVE: reduce VRAM de activaciones
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,                     # ahora sí, la RTX 5050 lo soporta
    logging_steps=10,
    save_steps=100, save_total_limit=2,
    output_dir="./checkpoints_socratic_7b",
    report_to="none",
)

trainer = SFTTrainer(
    model=model, train_dataset=dataset, tokenizer=tokenizer,
    dataset_text_field="text", max_seq_length=1024, args=training_args,
)
trainer.train()
model.save_pretrained("./qwen-math-7b-socratic-lora")
tokenizer.save_pretrained("./qwen-math-7b-socratic-lora")