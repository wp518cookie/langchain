from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

chat_model = init_chat_model(
    "deepseek:deepseek-chat",  # 改成 deepseek 的模型标识
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
    # 需要额外提供 DeepSeek 的 API 配置
    api_key="sk-2040b27da07d4d4d90a5e67bfa090d65",      # 你的 DeepSeek API Key
    api_base="https://api.deepseek.com",  # DeepSeek 的 API 地址
)

class Joke(BaseModel):
    setup: str = Field(description="笑话的开头铺垫")
    punchline: str = Field(description="笑话的包袱/笑点")

parser = PydanticOutputParser(pydantic_object=Joke)

template = ChatPromptTemplate.from_messages([
    ("system", "你是一个笑话机器人。\n{format_instructions}"),
    ("user", "讲一个关于{topic}的短笑话")
])

template = template.partial(format_instructions = parser.get_format_instructions())

chain = template | chat_model | parser

joke_obj = chain.invoke({"topic": "人工智能"})

print(joke_obj)            # Joke(setup='...', punchline='...')
print(joke_obj.punchline)