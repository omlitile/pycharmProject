# 导入核心模块
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import torch

# 1. 加载IMDB数据集
dataset = load_dataset("imdb")

# 2. 查看数据结构
print(f"训练样本数: {len(dataset['train'])}")
print(f"测试样本数: {len(dataset['test'])}")

# 3. 数据预处理函数
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",  # 自动填充到最长序列
        truncation=True,       # 超长文本截断
        max_length=512         # 最大输入长度
    )

# 4. 应用预处理
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,  # 批量处理加速
    remove_columns=["text", "label"]  # 移除原始列
)

# 加载预训练模型并调整输出层
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,  # 二分类任务
    trust_remote_code=True  # 兼容新版本模型
)

# 检查设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

training_args = TrainingArguments(
    output_dir="./results",          # 模型保存路径
    eval_strategy="epoch",     # 每个epoch评估
    save_strategy="epoch",    # 保存策略（必须与评估策略一致）
    learning_rate=2e-5,              # 学习率
    per_device_train_batch_size=8,  # 单设备训练批次
    per_device_eval_batch_size=32,   # 单设备评估批次
    num_train_epochs=3,              # 训练轮次
    weight_decay=0.01,               # 权重衰减
    logging_dir="./logs",            # 日志路径
    load_best_model_at_end=True      # 训练后加载最佳模型
)

# 自定义评估函数（关键）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {"accuracy": (predictions == labels).mean()}

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,  # 用于推理时的数据处理
    compute_metrics = compute_metrics  # 必须传递
)

# 开始训练
trainer.train()

# 保存最终模型
trainer.save_model("./fine_tuned_model")

# 评估指标计算
results = trainer.evaluate()

print(f"测试集准确率: {results['eval_accuracy']:.4f}")
print(f"测试集损失值: {results['eval_loss']:.4f}")