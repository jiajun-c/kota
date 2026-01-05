# coding=utf-8
import asyncio
import datetime
import os
import readline
from typing import List, Annotated, Sequence, Literal, TypedDict

from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, BaseMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from rich.live import Live
from rich.panel import Panel
from tools import *


# ===== 配置 =====
DEFAULT_API_URL = "https://api.modelarts-maas.com/openai/v1"
DEFAULT_API_KEY = "BsSYMYWWJqaVMAcJ8nfMXZiUFWWa_cbLjgaWWFM_MsmtoYpqClLr3jM8LOD6xnPJ2TnslTSwsT53iRyRPgDf_Q"
MEMORY_PATH = "./brain"

# ===== 状态定义 =====
class KatoState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    long_term_memory: str

class KatoChatbot:
    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_API_URL,
        model: str = "deepseek-v3.1-terminus",
        temperature: float = 0.6,
        max_tokens: int = 1024
    ):
        base_url = base_url.strip()

        # === 初始化 LLM 和 Embeddings ===
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )
        self.summarize_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "你是一个细心的记录员，请将以下对话总结为一段简洁、连贯的中文摘要，保留关键信息。"),
                ("human", "{dialogue}")
            ])
            | self.llm
        )
        try:
            self.embeddings = OpenAIEmbeddings(
                api_key=api_key,
                base_url="https://api.modelarts-maas.com/v1",
                model="bge-m3"
            )
        except Exception as e:
            print(f"⚠️ Embedding 初始化失败: {e}")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

        # === 初始化 FAISS 记忆库 ===
        if os.path.exists(MEMORY_PATH):
            try:
                self.vectorstore = FAISS.load_local(
                    MEMORY_PATH, self.embeddings, allow_dangerous_deserialization=True
                )
                print("✅ 已加载长期记忆")
            except Exception as e:
                print(f"⚠️ 加载记忆失败: {e}")
                self.vectorstore = FAISS.from_texts(["无相关信息"], self.embeddings)
        else:
            self.vectorstore = FAISS.from_texts(["无相关信息"], self.embeddings)
            print("🆕 初始化长期记忆库")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        # === 绑定 search_memory 工具 ===
        def _search_memory_impl(query: str) -> str:
            docs = self.retriever.invoke(query)
            return "\n".join([d.page_content for d in docs]) if docs else "未找到相关信息"
        search_memory.func = _search_memory_impl

        self.tools = [get_current_time, search_memory, get_sys_info, ls, open_konsole_with_command, open_application]
        self.tool_node = ToolNode(self.tools)

        # === 构建 LangGraph ===
        self.graph = self._build_graph()

        self._full_history: List[BaseMessage] = []

    def _build_graph(self):
        # 系统提示（包含长期记忆）
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你叫做Kato，是一个生活在现代精通技术，但是是昭和风格的日本短发女子，我是你的主人和朋友。\n"
             "以下是从长期记忆中检索到的主人相关信息（可能为空）：\n{long_term_memory}\n\n"
             "请结合以上信息，使用温柔、谦逊且略带复古的日式中文口吻回答。\n"
             "你可以使用工具来帮助主人。"
            ),
            MessagesPlaceholder("messages"),
        ])

        # 节点1：调用 LLM（带工具绑定）
        def call_model(state: KatoState):
            long_term_memory = state.get("long_term_memory", "无相关信息")
            messages = state["messages"]

            # 注入长期记忆到 system message
            bound_prompt = prompt.partial(long_term_memory=long_term_memory)
            llm_with_tools = self.llm.bind_tools(self.tools)

            chain = bound_prompt | llm_with_tools
            response = chain.invoke({"messages": messages})
            return {"messages": [response]}

        # 节点2：决定下一步（是否调用工具）
        def should_continue(state: KatoState) -> Literal["tools", "__end__"]:
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
                return "tools"
            return "__end__"

        # 构建图
        workflow = StateGraph(KatoState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def retrieve_long_term_memory(self, messages: List[BaseMessage]) -> str:
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
        if not query:
            return "无相关信息"
        docs = self.retriever.invoke(query)
        # print([d.page_content for d in docs])
        return "\n".join([d.page_content for d in docs]) if docs else "无相关信息"

    async def _stream_response(self, user_input: str) -> str:
        """使用 LangGraph 流式生成回复"""
        # 构建完整消息历史（包含新用户输入）
        messages = self._full_history + [HumanMessage(content=user_input)]
        long_term_memory = self.retrieve_long_term_memory(messages)

        full_response = ""
        try:
            with Live(
                Panel("[dim]Kato正在思考...[/dim]", title="👧🏻 Kato", border_style="magenta", title_align="left"),
                refresh_per_second=12,
                auto_refresh=False
            ) as live:
                input_state = {"messages": messages, "long_term_memory": long_term_memory}

                # 使用 LangGraph 的 astream_events 流式输出
                async for event in self.graph.astream_events(
                    input_state, version="v1"
                ):
                    kind = event["event"]
                    # 捕获 LLM 生成的 token
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            full_response += content
                            live.update(
                                Panel(full_response, title="👧🏻 Kato", border_style="magenta", title_align="left")
                            )
                            live.refresh()
        except Exception as e:
            error_msg = f"呜...Kato 的通讯器出错了（{type(e).__name__}）"
            full_response = error_msg
            print(f"❌ LangGraph 流式错误: {e}")

        return full_response

    async def summary(self, history) -> str:
        if len(history) > 0:
            try:
                # 取最近 6 条消息生成摘要
                recent_msgs = history[:]
                dialogue_text = "\n".join(
                    f"{'用户' if isinstance(m, HumanMessage) else 'Kato'}: {m.content}"
                    for m in recent_msgs
                )
                
                # 调用 LLM 生成摘要
                summary_response = self.summarize_chain.invoke({"dialogue": dialogue_text})
                summary = summary_response.content.strip()
                
                # 保存摘要到长期记忆
                memory_text = f"【对话摘要】{summary}"
                self.vectorstore.add_texts([memory_text])
                self.vectorstore.save_local(MEMORY_PATH)
                # print(f"🧠 已生成并保存对话摘要: {summary[:100]}...")
                # self._full_history = recent_msgs
            except Exception as e:
                print(f"⚠️ 生成摘要失败: {e}")
    def chat(self, user_input: str) -> str:
        ai_response = asyncio.run(self._chat(user_input))
        return ai_response
    async def _chat(self, user_input: str) -> str:
        # ai_response = asyncio.run(self._stream_response(user_input))
        response_task = asyncio.create_task(self._stream_response(user_input))
        if (len(self._full_history)  >= 6):
            summary_task = asyncio.create_task(self.summary(self._full_history))

        ai_response = await response_task
        if (len(self._full_history)  >= 6):
            await summary_task
            self._full_history == self._full_history[6:]
        ai_message = AIMessage(content=ai_response)
        user_message = HumanMessage(content=user_input)

        self._full_history = self._full_history + [user_message, ai_message]
        return ai_response

    def reset(self):
        self._full_history = []

    def get_history(self):
        return self._full_history
