from langchain_core.tools import tool
import datetime
import requests

@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    return datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")

@tool
def get_weather(location: str) -> str:
    """获取指定城市的天气（模拟）"""
    # 实际项目可接入和风天气、OpenWeather 等 API
    weather_map = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "东京": "小雨，22°C"
    }
    return weather_map.get(location, f"未知城市：{location}")