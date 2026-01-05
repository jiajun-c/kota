# coding=utf-8
import asyncio
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS  # ✅ 修复拼写错误
from rich.live import Live
from rich.panel import Panel
import os
import readline

# ===== 配置 =====
DEFAULT_API_URL = "https://api.modelarts-maas.com/openai/v1"  # ✅ 移除多余空格
DEFAULT_API_KEY = "BsSYMYWWJqaVMAcJ8nfMXZiUFWWa_cbLjgaWWFM_MsmtoYpqClLr3jM8LOD6xnPJ2TnslTSwsT53iRyRPgDf_Q"
MEMORY_PATH = "./brain"

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
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )

        # ✅ 尝试初始化 Embedding（带错误处理）
        try:
            self.embeddings = OpenAIEmbeddings(
                api_key=api_key,
                base_url="https://api.modelarts-maas.com/v1",
                model="bge-m3"  # ✅ 常见的 ModelArts embedding 模型
            )
        except Exception as e:
            print(f"⚠️ 初始化 OpenAI Embeddings 失败: {e}")
            print("🔄 回退到本地 HuggingFace Embeddings...")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5"
            )

        # 初始化长期记忆（FAISS）
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

        # 构建带长期记忆的 prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你叫做Kato，是一个生活在现代精通技术，但是是昭和风格的日本短发女子，我是你的主人和朋友。\n"
             "以下是从长期记忆中检索到的主人相关信息（可能为空）：\n{long_term_memory}\n\n"
             "请结合以上信息，使用温柔、谦逊且略带复古的日式中文口吻回答。"
            ),
            ("placeholder", "{messages}"),
        ])

        # 构建 chain
        def retrieve_long_term_memory(messages: List[BaseMessage]) -> str:
            # print("mess")
            query = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
            if not query:
                return "无相关信息"
            print(query)
            docs = self.retriever.invoke(query)
            print("docs: ", [doc.page_content for doc in docs])
            return "\n".join([doc.page_content for doc in docs]) if docs else "无相关信息"

        self.chain = (
            {
                "long_term_memory": lambda x: retrieve_long_term_memory(x["messages"]),
                "messages": lambda x: x["messages"]
            }
            | self.prompt
            | self.llm
        )

        self._full_history: List[BaseMessage] = []  # 完整对话历史

    async def _stream_response_with_history(self, messages: List[BaseMessage]) -> str:
        """流式生成回复（传入完整历史）"""
        full_response = ""
        
        # ✅ 直接调用 chain 而不是用 LangGraph（简化架构）
        try:
            with Live(
                Panel("[dim]GPU飞速运转[/dim]", title="👧🏻 Kato", border_style="magenta", title_align="left"),
                refresh_per_second=12,
                auto_refresh=False
            ) as live:
                # ✅ 直接流式调用 chain
                async for chunk in self.chain.astream({"messages": messages}):
                    if hasattr(chunk, 'content') and chunk.content:
                        full_response += chunk.content
                        live.update(
                            Panel(full_response, title="👧🏻 Kato", border_style="magenta", title_align="left")
                        )
                        live.refresh()
        except Exception as e:
            error_msg = f"呜...Kato 的通讯器出错了（{type(e).__name__}）"
            full_response = error_msg
            print(f"❌ 流式响应错误: {e}")
        
        return full_response

    def chat(self, user_input: str) -> str:
        """对外接口：处理用户输入并返回回复"""
        user_message = HumanMessage(content=user_input)
        
        # 1. 构建完整上下文（短期记忆）
        current_context = self._full_history + [user_message]
        
        # 2. 获取 AI 回复（✅ 统一使用 asyncio.run 处理异步）
        try:
            ai_response = asyncio.run(
                self._stream_response_with_history(current_context)
            )
        except Exception as e:
            print(f"⚠️ 异步调用失败: {e}")
            # 降级为同步调用
            response = self.chain.invoke({"messages": current_context})
            ai_response = response.content if hasattr(response, 'content') else str(response)
        
        ai_message = AIMessage(content=ai_response)
        
        # 3. 保存到长期记忆
        if len(user_input.strip()) > 2 and "无相关信息" not in ai_response:
            memory_text = f"用户说：{user_input}"
            try:
                self.vectorstore.add_texts([memory_text])
                self.vectorstore.save_local(MEMORY_PATH)
                print(f"💾 已保存记忆: {memory_text[:100]}...")
            except Exception as e:
                print(f"⚠️ 保存长期记忆失败: {e}")
        
        # 4. 更新短期历史
        self._full_history.extend([user_message, ai_message])
        return ai_response

    def reset(self):
        """重置对话历史"""
        self._full_history = []

    def get_history(self):
        """获取完整对话历史"""
        return self._full_history