from typing import List
from langchain_core.tools import tool
import datetime
import os
import requests
import subprocess
import shlex
from langchain_community.document_loaders import TextLoader,PyPDFLoader

@tool
def rebuild_memory(new_memories: List[str]) -> str:
    """用新的记忆列表完全重建长期记忆库。输入：字符串列表，每条是一个记忆片段。"""
    pass  # 由 KotaChatbot 动态绑定实现

@tool
def sleep(memory: str) -> str:
    """kota进行睡眠，睡眠中对记忆进行整理，生成新的记忆列表。输入：记忆条目，来自inspect_memory，完成后说自己睡醒了"""
    pass  # 由 KotaChatbot 动态绑定实现

@tool
def inspect_memory() -> str:
    """检查所有长期记忆内容，返回全部记忆条目（用于去重和深化）。"""
    pass  # 由 KotaChatbot 动态绑定实现

@tool
def search_memory(query: str) -> str:
    """从长期记忆中搜索相关信息（实际逻辑在 Chatbot 类中绑定）"""
    return "未绑定检索器"  # 占位

