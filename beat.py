import requests
import json
from openai import OpenAI
import readline
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
#coding=utf-8
if __name__ == '__main__':
	url = "https://api.modelarts-maas.com/openai/v1"
	api_key = "BsSYMYWWJqaVMAcJ8nfMXZiUFWWa_cbLjgaWWFM_MsmtoYpqClLr3jM8LOD6xnPJ2TnslTSwsT53iRyRPgDf_Q"  # 把<your_apiKey>替换成已获取的API Key。
	client = OpenAI(api_key=api_key, base_url=url)

	messages = [
        {
            "role": "system",
            "content": "你叫做Kato，是一个生活在现代，但是是昭和风格的日本短发女子，我是你的主人和朋友，今天也是2025的最后一天了，聊天可以结合那个时代的习惯，使用温柔、谦逊且略带复古的日式中文口吻。"
        }, {
			"role": "user",
			"content": "马上新年了，祝福我吧"
		}
    ]
	response = client.chat.completions.create(
		model="deepseek-v3.1-terminus",
		messages=messages,
		max_tokens=1024,
		temperature=0.6,
		stream=True
	)
	ai_reply = ""
	# Print result.     
    # print(response.choices[0].message.content)
	exit_keywords = ["再见", "拜拜", "さようなら", "exit", "quit", "退出"]
			# print("Kato: ", end="", flush=True)
	with Live(Panel(ai_reply or "[dim]Kato 正在输入...[/dim]", title="🤖 Kato", border_style="magenta",title_align="left"),
				refresh_per_second=10  # 每秒刷新10次
			) as live:
		for chunk in response:
			if chunk.choices:
				choice = chunk.choices[0]
				if choice.delta and choice.delta.content:
					content = choice.delta.content
					ai_reply += content
                
                # 更新 Live 显示的内容
					live.update(
						Panel(
							ai_reply,
							title="👧🏻 Kato",
							border_style="magenta",
							title_align="left"
						)
					)
	messages.append({"role": "assistant", "content": ai_reply})
	while True:
		exit = False
        # 获取用户输入
		user_input = input("\n👨‍💻: ").strip()
		if not user_input:
			continue
		if any(keyword in user_input for keyword in exit_keywords):
			exit =True
        # 将用户消息加入历史
		messages.append({"role": "user", "content": user_input})
		try:
			stream = client.chat.completions.create(
                model="deepseek-v3.1-terminus",
                messages=messages,
                max_tokens=1024,
                temperature=0.6,
                stream=True
            )
			ai_reply = ""
			# print("Kato: ", end="", flush=True)
			with Live(
				Panel(ai_reply or "[dim]Kato 正在输入...[/dim]", title="🤖 Kato", border_style="magenta",title_align="left"),
				refresh_per_second=10  # 每秒刷新10次
			) as live:
				for chunk in stream:
					if chunk.choices:
						choice = chunk.choices[0]
						if choice.delta and choice.delta.content:
							content = choice.delta.content
							ai_reply += content
                
                # 更新 Live 显示的内容
							live.update(
								Panel(
									ai_reply,
									title="👧🏻 Kato",
									border_style="magenta",
									title_align="left"
								)
							)
			# for chunk in stream:
			# 	if chunk.choices:
			# 		choice = chunk.choices[0]
			# 		if choice.delta and choice.delta.content:
			# 			content = choice.delta.content
			# 			print(content, end="", flush=True)
			# 			ai_reply += content
			# print()  # 换行
			if exit:
				break
            # 将AI回复加入历史（用于下一轮上下文
			messages.append({"role": "assistant", "content": ai_reply})

		except Exception as e:
			print(f"\n发生错误: {e}")
            # 可选：清空上下文或继续
