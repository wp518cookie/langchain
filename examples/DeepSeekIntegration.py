from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

from examples.fetch_text_from_url import fetch_text_from_url

model = init_chat_model(
    "deepseek:deepseek-chat",  # 改成 deepseek 的模型标识
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
    # 需要额外提供 DeepSeek 的 API 配置
    api_key="sk-2040b27da07d4d4d90a5e67bfa090d65",      # 你的 DeepSeek API Key
    api_base="https://api.deepseek.com",  # DeepSeek 的 API 地址
)

SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

content = f"what have on the page: www.baidu.com"

agent_result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "great-gatsby-lc"}},
)
deep_agent_result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "great-gatsby-da"}},
)
print(agent_result["messages"][-1].content_blocks)
print("\n")
print(deep_agent_result["messages"][-1].content_blocks)