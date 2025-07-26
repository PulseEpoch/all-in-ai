import torch
import pandas as pd
from transformers import GPT2LMHeadModel, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
import os

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 加载中文GPT2模型和分词器
model_name = "ckiplab/gpt2-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 设置填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载中文诗歌数据集
with open('chinese_poetry.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]
print(f'清洗后数据集大小: {len(lines)} 样本')

# 数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 自定义数据集类
class PoetryDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )
        encoding['labels'] = encoding['input_ids'].copy()
        return encoding

dataset = PoetryDataset(lines, tokenizer)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results_poetry_lora",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=500,
    logging_dir="./logs_poetry_lora",
    logging_steps=100,
    learning_rate=3e-4,
    weight_decay=0.001,
    gradient_accumulation_steps=2,
    report_to="none",
    fp16=False,
    dataloader_drop_last=True,
    dataloader_num_workers=0,
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 开始训练
print("开始使用LoRA微调中文诗歌模型...")
trainer.train()

# 保存LoRA模型
model.save_pretrained("./fine_tuned_poetry_lora_model")
tokenizer.save_pretrained("./fine_tuned_poetry_lora_model")
print("中文诗歌LoRA模型微调完成并保存!")

# 诗歌生成函数
def generate_poetry(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试诗歌生成
if __name__ == "__main__":
    test_prompts = ["床前明月光，", "春眠不觉晓，", "大漠孤烟直，"]
    for prompt in test_prompts:
        poetry = generate_poetry(prompt)
        print(f"输入提示: {prompt}")
        print(f"生成诗歌: {poetry}\n")