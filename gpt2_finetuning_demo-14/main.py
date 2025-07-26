import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from tqdm import tqdm
import time

# 设置随机种子确保可复现性
torch.manual_seed(42)

# 检测并设置设备（支持MPS）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # 设置填充标记

# 加载原始GPT2模型（微调前）
model_before = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# 加载数据集（使用小型莎士比亚数据集作为示例）
dataset = load_dataset('tiny_shakespeare')

def preprocess_function(examples):
    # 对文本进行分词处理
    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

# 预处理数据集
tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT2是自回归模型，不需要掩码语言模型
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # 如果有GPU则使用混合精度训练
)

# 加载要微调的模型
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(1)),  # 仅使用可用的1个样本
    eval_dataset=tokenized_dataset["validation"].shuffle(seed=42).select(range(1)),
    data_collator=data_collator,
)

# 微调前生成文本函数
def generate_text(model, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 微调前的文本生成
prompt = "He is a poor boy, and worrying about the future."
output = generate_text(model_before, prompt)


# 开始微调
print("\n开始微调模型...")
start_time = time.time()
trainer.train()
end_time = time.time()
print(f"微调完成，耗时: {end_time - start_time:.2f}秒")

# 保存微调后的模型
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# 加载微调后的模型
model_after = GPT2LMHeadModel.from_pretrained("./fine_tuned_model").to(device)

# 对比总结
print("\n=== 模型对比总结 ===")
print("微调前模型: 使用原始GPT2参数，生成通用文本")
print("微调后模型: 在莎士比亚数据集上微调，生成文本更接近莎士比亚风格")
print("=== 输入提示 ===")
print(prompt)
print("\n=== 微调前模型生成结果 ===")
print(output)
# 微调后的文本生成
print("\n=== 微调后模型生成结果 ===")
print(generate_text(model_after, prompt))