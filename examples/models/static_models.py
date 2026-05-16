from langchain.chat_models import init_chat_model

model = init_chat_model(
    "deepseek:deepseek-chat",  # 改成 deepseek 的模型标识
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
    # 需要额外提供 DeepSeek 的 API 配置
    api_key="sk-2040b27da07d4d4d90a5e67bfa090d65",      # 你的 DeepSeek API Key
    api_base="https://api.deepseek.com",  # DeepSeek 的 API 地址
)

response = model.invoke("hello world")
print(response)