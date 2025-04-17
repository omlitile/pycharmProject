from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# prompt_template = PromptTemplate.from_template(
#     "给我讲一个关于{content}的{adjective}笑话"
# )
# result = prompt_template.format(adjective="冷",content="猴子")
# print(result)


chat_template = ChatPromptTemplate.from_messages([
    ("system","你是一位人工智能助手，你的名字是{name}"),
    ("human","你好"),
    ("ai","我很好，谢谢"),
    ("human","{user_input}")
])
messages = chat_template.format_messages(name="Bob",user_input="你的名字叫什么？")
print(messages)