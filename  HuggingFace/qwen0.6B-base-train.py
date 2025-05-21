from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate

##加载数据集
dataset = load_dataset("yelp_review_full")
print(dataset["train"][100])
##使用分词器把文字转换成机器模型能够处理的数字
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
##生成小数据集
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))
##指定微调模型
model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-0.6B-Base", num_labels=5)


##设置超参数
##此外，TrainingArguments 提供了许多其他参数，你可以根据具体的训练需求来设置，比如：
##per_device_train_batch_size：每个设备上的训练批次大小。
##learning_rate：设置学习率。
##num_train_epochs：训练的迭代次数。
##evaluation_strategy：定义何时进行评估（例如 "steps", "epoch" 等）。
##save_steps：在训练过程中保存模型的频率。
##logging_dir：用于存放 TensorBoard 日志文件的目录。
training_args = TrainingArguments(output_dir="test_trainer")

##训练评估和计算预测的准确性
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


##创建Trainer对象执行微调任务
trainer = Trainer(
    model=model,  # 定义的模型（ BERT 分类模型）
    args=training_args,  # 训练的参数，如学习率、批次大小、训练周期等
    train_dataset=small_train_dataset,  # 用于训练的小数据集
    eval_dataset=small_eval_dataset,  # 用于评估的小数据集
    compute_metrics=compute_metrics,  # 之前定义的，验证的准确性等指标的函数
)
trainer.train() # 开始训练模型
trainer.save_model() # 保存训练后的模型