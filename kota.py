# coding=utf-8
import sys
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from rich.live import Live
from rich.panel import Panel

# ===== 配置（可作为类参数传入）=====
DEFAULT_API_URL = "https://api.modelarts-maas.com/openai/v1"
DEFAULT_API_KEY = "BsSYMYWWJqaVMAcJ8nfMXZiUFWWa_cbLjgaWWFM_MsmtoYpqClLr3jM8LOD6xnPJ2TnslTSwsT53iRyRPgDf_Q"

class KatoChatbot:
    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_API_URL,
        model: str = "deepseek-v3.1-terminus",
        temperature: float = 0.6,
        max_tokens: int = 1024
    ):
        # 修复 URL 空格问题（关键！）
        base_url = base_url.strip()
        
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 构建带历史的链
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你叫做Kato，是一个生活在现代精通技术，但是是昭和风格的日本短发女子，我是你的主人和朋友，使用温柔、谦逊且略带复古的日式中文口吻。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        chain = prompt | self.llm | StrOutputParser()

        # 对话历史
        self.history = ChatMessageHistory()
        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _stream_response(self, user_input: str) -> str:
        """内部方法：流式生成回复并显示"""
        full_response = ""
        with Live(
            Panel("[dim]GPU飞速运转[/dim]", title="👧🏻 Kato", border_style="magenta", title_align="left"),
            refresh_per_second=8
        ) as live:
            try:
                stream = self.chain_with_history.stream(
                    {"input": user_input},
                    config={"configurable": {"session_id": "default"}}
                )
                for text_chunk in stream:
                    full_response += text_chunk
                    live.update(
                        Panel(full_response, title="👧🏻 Kato", border_style="magenta", title_align="left")
                    )
            except Exception as e:
                error_msg = f"呜...Kato 的通讯器出错了（{type(e).__name__}）"
                full_response = error_msg
                live.update(Panel(error_msg, title="💔 Kato", border_style="red"))
        return full_response

    def chat(self, user_input: str) -> str:
        """对外接口：用户输入 → AI 流式回复"""
        return self._stream_response(user_input)

    def reset(self):
        """重置对话历史"""
        self.history = ChatMessageHistory()
        # 重新绑定 chain（或清空内部状态）
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain_with_history.wrapped,
            lambda session_id: self.history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def get_history(self):
        """获取当前对话历史（用于调试或保存）"""
        return self.history.messages
