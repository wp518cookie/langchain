from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model(
    "deepseek:deepseek-chat",  # 改成 deepseek 的模型标识
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
    # 需要额外提供 DeepSeek 的 API 配置
    api_key="sk-2040b27da07d4d4d90a5e67bfa090d65",      # 你的 DeepSeek API Key
    api_base="https://api.deepseek.com",  # DeepSeek 的 API 地址
)

class commandLineChatbot:
    def __init__(self):
        self.chat = model
        self.message = [SystemMessage(content="你是一个花卉行家。")]

    def chat_loop(self):
        print("chatbot 已启动！输入'exit'来退出程序。")
        while True:
            user_input = input("你：")
            if user_input == "exit":
                print("再见！")
                break
            self.message.append(HumanMessage(content=user_input))
            response = self.chat.invoke(self.message)
            print("chatbot：", response.content)

if __name__ == "__main__":
    bot = commandLineChatbot()
    bot.chat_loop()