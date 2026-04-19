import json
from py.get_setting import HOST,PORT
from openai import AsyncOpenAI
async def get_agent_tool(settings):
    tool_agent_list = []
    for agent_id,agent_config in settings['agents'].items():
        if agent_config['enabled']:
            tool_agent_list.append({"agent_id": agent_id, "agent_skill": agent_config["system_prompt"]})
    if len(tool_agent_list) > 0:
        tool_agent_list = json.dumps(tool_agent_list, ensure_ascii=False, indent=4)
        agent_tool = {
            "type": "function",
            "function": {
                "name": "agent_tool_call",
                "description": f"根据Agent给出的agent_skill调用指定Agent工具，返回结果。当前可用的Agent工具ID以及Agent工具的agent_skill有：{tool_agent_list}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "需要调用的Agent工具ID",
                        },
                        "query": {
                            "type": "string",
                            "description": "需要向Agent工具发送的问题",
                        }
                    },
                    "required": ["agent_id", "query"]
                }
            }
        }
    else:
        agent_tool = None
    return agent_tool

async def agent_tool_call(agent_id, query):
    try:
        client = AsyncOpenAI(
            api_key="super-secret-key",
            base_url=f"http://{HOST}:{PORT}/v1"
        )
        response = await client.chat.completions.create(
            model=agent_id,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        res = response.choices[0].message.content
        return str(res)
    except Exception as e:
        print(f"Error: {e}")
        return str(e)

