import asyncio
import sys
import re
import json
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

def format_tools_for_llm(tool: Tool) -> str:
    """Format tool schema for LLM prompt"""
    args_desc = []
    if 'properties' in tool.inputSchema:
        for param_name, param_info in tool.inputSchema["properties"].items():
            desc = param_info.get('description', 'No description')
            arg_desc = f"- {param_name}: {desc}"
            if param_name in tool.inputSchema.get("required", []):
                arg_desc += " (required)"
            args_desc.append(arg_desc)
    return (
        f"Tool: {tool.name}\n"
        f"Description: {tool.description}\n"
        f"Arguments:\n{chr(10).join(args_desc)}"
    )

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction from potentially messy LLM responses
    Returns first valid JSON object found or None
    """
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Look for JSON-like structures in text
    patterns = [
        r'\{[\s\S]*\}',  # Match outermost {...}
        r'```json\s*({[\s\S]*})\s*```',  # Extract from JSON code blocks
        r'({[\s\S]*})'   # Any JSON object
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, IndexError):
                continue
    return None

def is_valid_tool_call(data: Any) -> bool:
    """Validate tool call structure"""
    return (
        isinstance(data, dict) and
        "tool" in data and isinstance(data["tool"], str) and
        "arguments" in data and isinstance(data["arguments"], dict)
    )

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools: List[Tool] = []  # Cache available tools
        self.client = AsyncOpenAI(
            # FIXED: Removed trailing spaces in base_url
            base_url="https://api.modelarts-maas.com/openai/v1",
            api_key="BsSYMYWWJqaVMAcJ8nfMXZiUFWWa_cbLjgaWWFM_MsmtoYpqClLr3jM8LOD6xnPJ2TnslTSwsT53iRyRPgDf_Q",
        )
        self.model = "deepseek-v3.1-terminus"
        self.messages = []
 
    async def connect_to_server(self, server_script_path: str):
        """Connect to MCP server and cache available tools"""
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )
 
        stdio, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()
 
        # Cache available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        print("\n服务器中可用的工具：", [tool.name for tool in self.tools])
 
        tools_description = "\n".join([format_tools_for_llm(tool) for tool in self.tools])
        system_prompt = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{tools_description}\n\n"
            "Choose the appropriate tool based on the user's question. "
            "If no tool is needed, reply directly.\n\n"
            "IMPORTANT: When you need to use a tool, you must ONLY respond with "
            "the exact JSON object format below, nothing else:\n"
            "{\n"
            '    "tool": "tool-name",\n'
            '    "arguments": {\n'
            '        "argument-name": "value"\n'
            "    }\n"
            "}\n\n"
            "### CRITICAL INSTRUCTIONS ###\n"
            "1. When you receive a message starting with 'TOOL_RESULT:',\n"
            "   - This is the result from a previous tool call\n"
            "   - DO NOT call any tools\n"
            "   - Generate a natural language response based on this data\n"
            "2. Format your response as a normal conversational reply\n"
            "3. Do not include JSON or tool call format in your response\n\n"
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n\n"
            "Please use only the tools that are explicitly defined above."
        )
        self.messages.append({"role": "system", "content": system_prompt})
 
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM with given messages (does NOT modify history)"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
 
    async def execute_tool(self, llm_response: str) -> str:
        """Process LLM response and execute tools if valid"""
        tool_call = extract_json(llm_response)
        
        # If not valid tool call format, return original response
        if not tool_call or not is_valid_tool_call(tool_call):
            return llm_response
        
        tool_name = tool_call["tool"]
        arguments = tool_call["arguments"]
        
        # Check if tool exists in cached tools
        available_tools = {tool.name: tool for tool in self.tools}
        if tool_name not in available_tools:
            return f"Error: Tool '{tool_name}' not available. Available tools: {list(available_tools.keys())}"
        
        try:
            print(f"[执行工具]: {tool_name} with args: {arguments}")
            result = await self.session.call_tool(tool_name, arguments)
            
            # Handle progress updates if present
            if isinstance(result, dict) and "progress" in result:
                progress = result["progress"]
                total = result.get("total", 100)
                percentage = (progress / total) * 100 if total else 0
                print(f"Progress: {progress}/{total} ({percentage:.1f}%)")
                
            print(f"[工具结果]: {result}")
            return f"Tool execution result: {result}"
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            print(f"[错误] {error_msg}")
            return error_msg

    async def chat_loop(self):
        """Interactive chat loop with proper message history management"""
        print("\nMCP 客户端启动 (输入 /bye 退出)")
        print("可用工具:", [tool.name for tool in self.tools])
        print("-" * 50)

        while True:
            user_input = input("\n>>> ").strip()
            if user_input.lower() in ['/bye', '/exit', '/quit']:
                break
            
            # Add user message to history
            self.messages.append({"role": "user", "content": user_input})
            
            # Get LLM response
            llm_response = await self.chat(self.messages)
            print(f"\n[LLM 原始响应]:\n{llm_response}\n")
            
            # Check if this is a tool call
            tool_result = await self.execute_tool(llm_response)
            
            if tool_result != llm_response:
                # Tool was executed - add tool result as system message
                self.messages.append({
                    "role": "user", 
                    "content": f"Tool result: {tool_result}"
                })
                print(self.messages)
                # Get final response from LLM
                final_response = await self.chat(self.messages)
                print(f"\n[最终回复]:\n{final_response}\n")
                
                # Add final response to history
                self.messages.append({
                    "role": "assistant", 
                    "content": final_response
                })
            else:
                # Direct response - add to history
                print(f"\n[回复]:\n{llm_response}\n")
                self.messages.append({
                    "role": "assistant", 
                    "content": llm_response
                })
            
            print("-" * 50)

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        print("Example: python client.py ./mcp_server.py")
        sys.exit(1)
 
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        # Ensure proper cleanup
        await client.exit_stack.aclose()
        print("\n已断开连接，资源已释放")

if __name__ == "__main__":
    asyncio.run(main())