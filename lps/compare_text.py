from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import difflib
import pandas as pd
import env
import os
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# 初始化OpenAI模型
def init_model():
    return ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0.1,  # 严格遵循提示要求
        top_p=0.85,  # 保留高概率差异点
        max_tokens=500,  # 限制单次输出长度
        frequency_penalty=0.8,  # 防止重复报告相同差异
        presence_penalty = 0.4,  # 避免遗漏新条款
        api_key=os.getenv("OPENAI_API_KEY")
    )


# 智能分块策略
def smart_chunking(text, chunk_size=512, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


# 差异检测Prompt模板
DIFF_PROMPT_TEMPLATE = """作为专业文本分析师，请完成以下任务：
1. 对比两段文本的差异（版本A vs 版本B）
2. 识别以下差异类型：
   - 新增内容（标记为+）
   - 删除内容（标记为-）
   - 修改内容（标记为*）
3. 标注差异位置（字符位置和语义变化）
4. 评估差异重要性等级（高/中/低）

文本A：{text_a}
文本B：{text_b}

请按以下JSON格式输出结果：
[{
    "type": "+|-|*",
    "position": [start,end],
    "text_a": "原内容",
    "text_b": "新内容",
    "reason": "变化原因描述",
    "importance": "等级"
}]"""

diff_parser = StrOutputParser()
# 创建差异检测链
def create_diff_chain(llm):
    # prompt_template = PromptTemplate(
    #     input_variables=["text_a", "text_b"],
    #     template="比较以下两段文本的差异：{text_a} 和 {text_b}"
    # )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "比较以下两段文本的差异"),
            ("human", "文本A：{text_a}\n文本B：{text_b}")
        ]
    )
    return RunnableSequence(
        llm | prompt_template | RunnableLambda(lambda x: pd.DataFrame(x["result"]))
    )


# 差异聚合与可视化
def aggregate_diffs(all_diffs):
    diff_df = pd.concat(all_diffs)
    diff_df['position'] = diff_df['position'].apply(lambda x: f"{x[0]}-{x[1]}")

    # 生成HTML可视化报告
    html_report = diff_df.style \
        .set_properties(**{'background-color': '#FFCCCC', 'color': '#990000'}, subset=['type', 'importance']) \
        .set_caption('文本差异分析报告') \
        .to_html()

    return html_report


# 主处理流程
def compare_large_texts(file_a, file_b, chunk_size=1024):
    # 加载文本
    loaderA = TextLoader(file_a, autodetect_encoding=False, encoding="utf-8")
    text_a = loaderA.load()[0].page_content
    loaderB = TextLoader(file_b, autodetect_encoding=False, encoding="utf-8")
    text_b = loaderB.load()[0].page_content

    # 分块处理
    chunks_a = smart_chunking(text_a, chunk_size)
    chunks_b = smart_chunking(text_b, chunk_size)

    # 初始化模型和链
    llm = init_model()
    diff_chain = create_diff_chain(llm)

    # 并行处理差异检测
    all_diffs = []
    for chunk_a, chunk_b in zip(chunks_a, chunks_b):
        input_data = {"text_a": chunk_a, "text_b": chunk_b}
        print(input_data)
        result = diff_chain.invoke(input_data)
        all_diffs.append(result)

    # 聚合结果
    return aggregate_diffs(all_diffs)


# 示例调用
if __name__ == "__main__":
    print(os.getenv("OPENAI_API_KEY"))
    file_a = "demo1.txt"
    file_b = "demo2.txt"
    report = compare_large_texts(file_a, file_b)

    # 保存报告
    with open("diff_report.html", "w", encoding="utf-8") as f:
        f.write(report)
    print("差异报告已生成：diff_report.html")