chunks = []
for chunk in model.stream("天空是什么颜色？"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)