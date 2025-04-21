from sentence_transformers import SentenceTransformer, util

# 加载多语言句子嵌入模型
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# 待对比文本
text1 = "今天天气晴朗，适合户外活动。"
text2 = "阳光明媚，正是出游的好天气。"

# 编码文本
embedding1 = model.encode(text1, convert_to_tensor=True)
embedding2 = model.encode(text2, convert_to_tensor=True)

# 计算余弦相似度
cos_sim = util.cos_sim(embedding1, embedding2)
print(f"文本相似度：{cos_sim.item():.4f} (阈值>0.75建议判定为相似)")