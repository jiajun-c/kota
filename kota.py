# build with langchain
# coding=utf-8
import sys
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
import sys
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
import readline

# ===== 配置 =====
API_URL = "https://api.modelarts-maas.com/openai/v1"  # ✅ 无空格
API_KEY = "BsSYMYWWJqaVMAcJ8nfMXZiUFWWa_cbLjgaWWFM_MsmtoYpqClLr3jM8LOD6xnPJ2TnslTSwsT53iRyRPgDf_Q"

# ==== 初始化 LLM ====
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=API_URL,
    model="deepseek-v3.1-terminus",
    temperature=0.6,
    max_tokens=1024,
)

# ==== 构建带历史的链 ====
prompt = ChatPromptTemplate.from_messages([
    ("system", "你叫做Kato，是一个生活在现代精通技术，但是是昭和风格的日本短发女子，我是你的主人和朋友，使用温柔、谦逊且略带复古的日式中文口吻。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 使用 StrOutputParser，输出纯字符串
chain = prompt | llm | StrOutputParser()

# 启用历史
history = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history",
)

# ==== 流式调用函数 ====
def stream_ai_response(user_input: str) -> str:
    full_response = ""
    with Live(
        Panel("[dim]GPU飞速运转[/dim]", title="👧🏻  Kato", border_style="magenta", title_align="left"),
        refresh_per_second=8
    ) as live:
        try:
            stream = chain_with_history.stream(
                {"input": user_input},
                config={"configurable": {"session_id": "default"}}
            )
            for text_chunk in stream:  # text_chunk 是 str
                full_response += text_chunk
                live.update(
                    Panel(full_response, title="👧🏻 Kato", border_style="magenta", title_align="left")
                )
        except Exception as e:
            error_msg = f"呜...Kato 的通讯器出错了（{type(e).__name__}）"
            full_response = error_msg
            live.update(Panel(error_msg, title="💔 Kato", border_style="red"))
    return full_response

stream_ai_response("你好")
exit_keywords = ["再见", "拜拜", "さようなら", "exit", "quit", "退出"]

if __name__ == '__main__':
    while True:
        user_input = input("\n👨‍💻: ").strip()
        if not user_input:
            continue
        if any(keyword in user_input for keyword in exit_keywords):
            exit =True
        try:
            stream_ai_response(user_input)
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"发生了其他异常: {type(e).__name__}: {e}")
        if exit:
            break