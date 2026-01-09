![kota](./img/kota.png)

An AI agent for kde desktop

## Architecture

reAct framework with LangChain, LangGraph and Faiss

<img width="1124" height="578" alt="image" src="https://github.com/user-attachments/assets/d312c823-18b5-4ffa-8308-a8c35821e769" />

## feature

- [x] Long-Term Memory with FAISS
- [x] Automatic Memory Management:
  - Summarization: After every 6 messages, the conversation is summarized and saved as a memory entry.
  - Nightly Maintenance: Supports a sleep tool to deduplicate, refine, and clean memory entries (e.g., remove noise, resolve contradictions, infer higher-level insights).
- [x] Rich Tool Ecosystem:
  - File operations (readfile, ls, grep, readpdffile)
  - System interaction (get_sys_info, execute_command, open_application, open_konsole_with_command)
  - Memory inspection & rebuilding (inspect_memory, rebuild_memory, search_memory)
  - Time awareness (get_current_time)
  - GUI file selection via KDE’s kdialog (request_file_upload_via_kdialog)
- [x] Streaming Response UI
- [x] LangGraph-Powered Agent Workflow:
- [x] Safe Tool Execution:
Implements a tool call limit (MAX_TOOL_CALLS = 10) to prevent infinite loops or excessive tool usage.
- [x] Tools Router: use a router node to dynamically add tools to the agent.
- [ ] RAG
- [ ] version + voice
- [ ] Self-awareness
- [ ] Agentic rl
- [ ] mcp/skills

## video

https://github.com/user-attachments/assets/99eeab3e-cff3-4aff-bcf3-0e282d05d573

## usage
replace the api key in kota.py the `python3 beats.py` to make this heart beats.
> some function in the tools only work for the linux and kde desktop. Because I use manjaro(:
