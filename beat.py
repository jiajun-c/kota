from kota import KotaChatbot


# ===== 主程序 =====
def main():
    exit_keywords = ["再见", "拜拜", "bye", "さようなら", "exit", "quit", "退出"]
    
    # 初始化 kota
    kota = KotaChatbot()

    # 打个招呼
    kota.chat("你好")

    while True:
        exit = False
        try:
            user_input = input("\n👨‍💻: ").strip()
            if not user_input:
                continue

            if any(keyword in user_input for keyword in exit_keywords):
                exit = True

            # 调用 kota 回复
            kota.chat(user_input)
            if exit:
                break
        except KeyboardInterrupt:
            print("\n\n👧🏻 kota: 啊啦...主人要离开了吗？请保重身体，明年再见哦。")
            break
        except Exception as e:
            print(f"\n⚠️  发生了意外错误: {type(e).__name__}: {e}")
            
if __name__ == '__main__':
   main()  # ✅ 启动异步主循环