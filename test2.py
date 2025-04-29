from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import env
import os
from langchain_community.document_loaders import TextLoader


# 初始化组件
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.1,  # 严格遵循提示要求
    top_p=0.85,  # 保留高概率差异点
    max_tokens=500,  # 限制单次输出长度
    frequency_penalty=0.8,  # 防止重复报告相同差异
    presence_penalty=0.4,  # 避免遗漏新条款
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
# 格式化输出
parser = StrOutputParser()

# 构建模板（注意：输入参数通过字符串插值）
temp = ("作为专业文本分析师，请完成以下任务"
        "1. 对比两段文本的差异（版本A vs 版本B）"
        "2. 识别以下差异类型：- 新增内容（标记为+）- 删除内容（标记为-）修改内容（标记为*）"
        "3. 标注差异位置（字符位置和语义变化）"
        "4. 评估差异重要性等级（高/中/低）"
        "5.用Json格式输出结果")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", temp),
    ("human", "文本A：{text_a}\n文本B：{text_b}")
])

# 读取文本
file_a = "lps/demo1.txt"
file_b = "lps/demo2.txt"
loaderA = TextLoader(file_a, autodetect_encoding=False, encoding="utf-8")
text_a = loaderA.load()[0].page_content
loaderB = TextLoader(file_b, autodetect_encoding=False, encoding="utf-8")
text_b = loaderB.load()[0].page_content
input_data = {
    "text_a": text_a,
    "text_b": text_b
}
diff_chain = prompt_template | llm | parser
file = open("result.txt", "a",encoding="utf-8")

for num in range(1, 11):
    print(num)
    result = diff_chain.invoke(input_data)
    file.write(result)
    file.write("==========================================================================")

file.close()
