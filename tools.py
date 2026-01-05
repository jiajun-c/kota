from langchain_core.tools import tool
import datetime
import os
import requests

@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")

@tool
def get_sys_info() -> str:
    """获取当前系统信息"""
    return f"当前系统{os.uname()}"

@tool
def search_memory(query: str) -> str:
    """从长期记忆中搜索相关信息（实际逻辑在 Chatbot 类中绑定）"""
    return "未绑定检索器"  # 占位