from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint

console = Console()

# 1. 显示一个“输入提示”面板
input_prompt = Panel(
    "在此输入您的消息 👇",
    title="💬 输入框",
    border_style="blue",
    padding=(1, 2),
    expand=False  # 不占满宽度
)

# 2. 渲染面板
console.print(input_prompt)

# 3. 获取用户输入（实际输入在面板下方）
user_input = Prompt.ask("[bold]>>>[/bold]")

# 4. 显示用户输入内容（可选）
rprint(Panel(
    user_input,
    title="✅ 您输入了",
    border_style="green",
    padding=(0, 1)
))
