from langchain_core.tools import tool
import datetime
import os
import requests
import subprocess
import shlex
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

@tool
def ls(path: str = ".") -> str:
    """列出指定目录的文件名（不调用 shell，更安全）"""
    try:
        if not os.path.isdir(path):
            return f"错误: '{path}' 不是有效目录"
        files = os.listdir(path)
        return "\n".join(sorted(files))
    except PermissionError:
        return "错误: 权限不足"
    except Exception as e:
        return f"错误: {str(e)}"
    
import shlex

def open_konsole_with_command(command: str, stay_open: bool = True):
    """
    在 Konsole 中执行命令。
    
    :param command: 要执行的 shell 命令（如 "ls -l && pwd"）
    :param stay_open: 是否在命令结束后保持窗口打开（方便查看输出）
    """
    wrapped_cmd = f'source ~/.zshrc 2>/dev/null; {command}'
    
    if stay_open:
        full_cmd = f'zsh -c "{wrapped_cmd}; exec zsh -i"'
    else:
        full_cmd = f'zsh -c "{wrapped_cmd}"'
    print(full_cmd)
    try:
        subprocess.Popen([
            "konsole",
            "-e", "/bin/zsh", "-c", full_cmd
        ])
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def open_application(app_name: str):
    """
    在 Konsole 中打开应用程序,app名称为{name}，对应的
    google: "com.google.Chrome"
    firefox: "firefox"
    
    :param app_name: 要打开的应用程序的名称（如 "firefox"）
    """
    syspath = "/usr/share/applications/"
    fullpath = os.path.join(syspath, f"{app_name}.desktop")
    try:
        subprocess.Popen([
            "kioclient",
            "exec", fullpath
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"❌ 启动失败: {e}")