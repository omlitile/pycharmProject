from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
import env
from langchain_community.chat_models import ChatOpenAI

from langchain_openai import OpenAI
from langchain.chat_models import init_chat_model

# 一行代码初始化（自动处理依赖和版本兼容）
llm = init_chat_model(
    model="gpt-4-turbo",  # 支持gpt-4/4-turbo/3.5-turbo等
    temperature=0.3,           # 生成随机性控制（0-2）
    max_tokens=4000,           # 最大生成长度
    openai_api_key=os.getenv("OPENAI_API_KEY"),  # API密钥
    model_provider="openai"    # 显式指定提供商（可选）
)

# 调用示例
response = llm.invoke("用100字解释量子纠缠")
print(response.content)