# 获取环境感知，如时间，地点，人，物
from typing import List
from langchain_core.tools import tool
import datetime
import os
import requests
import subprocess
import shlex
import inspect
from langchain_community.document_loaders import TextLoader,PyPDFLoader

@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")


# members = inspect.getmembers(memory, inspect.isfunction)  
  
# # 遍历所有函数对象并打印  
# for name, func in members:  
#     print(f"Function Name: {name}")  
#     print(f"Function Object: {func}")  
#     # 如果需要，你还可以打印函数的定义  
#     # 注意：这将显示函数定义的源代码，如果源代码可用的话  
#     print(f"Function Definition:\n{inspect.getsource(func)}\n")