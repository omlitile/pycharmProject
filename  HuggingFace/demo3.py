from sentence_transformers import SentenceTransformer, util
import torch

# 初始化模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 输入数据
documents = [
    "人工智能技术正在改变医疗诊断方式",
    "机器学习算法在图像识别领域取得突破",
    "量子计算将颠覆传统密码学体系",
    "深度学习模型需要大量标注数据",
    "自然语言处理技术提升机器翻译质量"
]

queries = [
    "AI如何影响现代医疗",
    "最新的机器学习进展有哪些",
    "量子计算机与传统计算机的区别"
]

# 生成Embedding
query_emb = model.encode(queries, convert_to_tensor=True)
doc_emb = model.encode(documents, convert_to_tensor=True)

# 计算相似度
similarity = util.cos_sim(query_emb, doc_emb)  # (3,5)

# 获取降序排序索引
sorted_indices = similarity.argsort(dim=1, descending=True)

# 生成结果
results = []
for i in range(len(queries)):
    top3 = sorted_indices[i][:3]  # 取前三位
    for idx in top3:
        score = similarity[i, idx].item()
        if score > 0.25:
            results.append(f"{queries[i]} → {documents[idx]} ({score:.2f})")

# 输出结果
print("\n匹配结果：")
print("\n".join(results))