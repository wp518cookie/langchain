from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chat_model = init_chat_model(
    "deepseek:deepseek-chat",  # 改成 deepseek 的模型标识
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
    # 需要额外提供 DeepSeek 的 API 配置
    api_key="sk-2040b27da07d4d4d90a5e67bfa090d65",      # 你的 DeepSeek API Key
    api_base="https://api.deepseek.com",  # DeepSeek 的 API 地址
)

template = ChatPromptTemplate.from_messages([
    ("system", "你是一位善解人意的朋友，记得对话历史。"),
    ("placeholder", "{history}"),  # 历史消息会注入到这里
    ("user", "{input}")
])

chain = template | chat_model | StrOutputParser()

with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 第一轮对话
resp1 = with_history.invoke(
    {"input": "我叫小明，喜欢踢足球。"},
    config={"configurable": {"session_id": "user1"}}
)
print(resp1)

# 第二轮对话，模型应记得“小明”
resp2 = with_history.invoke(
    {"input": "请问我喜欢什么运动？"},
    config={"configurable": {"session_id": "user1"}}
)
print(resp2)
