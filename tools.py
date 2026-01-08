from typing import List
from langchain_core.tools import tool
import datetime
import os
import requests
import subprocess
import shlex
from langchain_community.document_loaders import TextLoader,PyPDFLoader

@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")

@tool
def get_sys_info() -> str:
    """获取当前系统信息"""
    print("获取系统信息...")
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
@tool
def grep(content: str, grepstr: str):
    """
    在给定的字符串内容中搜索包含 grepstr 的行。
    
    参数:
        content (str): 要搜索的完整文本内容（多行字符串）。
        grepstr (str): 要搜索的子字符串。
    
    返回:
        list of dict: 每个匹配行的信息，包含行号（从1开始）和内容。
                      例如: [{"line_number": 3, "content": "foo bar\\n"}, ...]
    """
    matches = []
    lines = content.splitlines(keepends=True)
    for line_num, line in enumerate(lines, start=1):
        if grepstr in line:
            matches.append({
                "line_number": line_num,
                "content": line
            })
    return matches

@tool
def execute_command(command: str, timeout: int = 30) -> str:
    """
    在后台执行 shell 命令并返回输出结果（不打开任何终端窗口）。
    
    :param command: 要执行的 shell 命令（如 "ls -l && pwd"）
    :param timeout: 命令超时时间（秒），防止卡死
    :return: 命令的标准输出 + 标准错误（若失败），或成功结果
    """
    try:
        # 加载 shell 配置（可选，根据你的环境需求调整）
        shell_env = os.environ.copy()
        # 可选：显式指定 PATH 或加载 .zshrc（但注意非交互式 shell 可能不加载）
        # 这里用 zsh -l 保证加载 login shell 环境
        result = subprocess.run(
            ["/bin/zsh", "-l", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=shell_env,
            cwd=os.getcwd()
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output else "命令执行成功，无输出。"
        else:
            error_msg = result.stderr.strip() or f"命令退出码: {result.returncode}"
            return f"❌ 执行失败:\n{error_msg}"
    except subprocess.TimeoutExpired:
        return f"⏰ 命令执行超时（>{timeout}秒），已终止。"
    except Exception as e:
        return f"💥 执行异常: {type(e).__name__}: {e}"

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

@tool
def inspect_memory() -> str:
    """检查所有长期记忆内容，返回全部记忆条目（用于去重和深化）。"""
    pass  # 由 KotaChatbot 动态绑定实现

@tool
def rebuild_memory(new_memories: List[str]) -> str:
    """用新的记忆列表完全重建长期记忆库。输入：字符串列表，每条是一个记忆片段。"""
    pass  # 由 KotaChatbot 动态绑定实现


@tool
def sleep(memory: str) -> str:
    """kota进行睡眠，睡眠中对记忆进行整理，生成新的记忆列表。输入：记忆条目，来自inspect_memory，完成后说自己睡醒了"""
    pass  # 由 KotaChatbot 动态绑定实现

@tool
def readfile(path: str) -> str:
    """
    读取文本文件内容
    :param path: 文件路径
    """
    return TextLoader(path).load()[0].page_content

# print(TextLoader("/home/star/.zshrc").load()[0].page_content)

def readpdffile(path: str) -> str:
    """
    读取PDF文件所有页面的内容并合并为单个字符串
    :param path: PDF文件路径
    :return: 所有页面的文本内容（按页拼接）
    """
    try:
        # 创建加载器
        loader = PyPDFLoader(path)
        
        # 加载所有页面（返回 Document 对象列表）
        pages = loader.load()
        
        # 提取所有页面的文本内容并合并
        full_text = "\n".join(page.page_content for page in pages)
        
        return full_text
    
    except Exception as e:
        return f"❌ PDF读取失败: {str(e)}"
    
# loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
# pages = loader.load_and_split()
def request_file_upload_via_kdialog(path: str = "/home/star", filesuffix="") -> str:
    """
    打开文件管理器用于选择文件。
    仅适用于 KDE 桌面环境。
    :param path: 默认打开的目录
    :param filesuffix: 文件后缀，如*.png
    """
    print(f"\n📎 Kota 请求上传文件:")
    # print("正在启动 KDE 文件选择器...")

    try:
        # 构造 kdialog 命令
        cmd = [
            "kdialog",
            "--getopenfilename",
            path,  # 默认打开目录
            filesuffix,
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60  # 最多等待 60 秒
        )

        if result.returncode == 0 and result.stdout.strip():
            file_path = result.stdout.strip()
            return f"用户选择了文件: {file_path}"
        else:
            return "用户取消了文件选择或 kdialog 未响应。"

    except FileNotFoundError:
        return "❌ kdialog 未安装（仅支持 KDE 桌面）。请使用其他方式上传。"
    except subprocess.TimeoutExpired:
        return "❌ 文件选择超时（60秒未操作）。"
    except Exception as e:
        return f"❌ 调用 kdialog 失败: {type(e).__name__}: {e}"
