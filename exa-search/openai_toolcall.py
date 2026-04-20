import json
from openai import OpenAI
from exa_py import Exa

openai = OpenAI()
exa = Exa()

tools = [
    {
        "type": "function",
        "function": {
            "name": "exa_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    }
]


def exa_search(query: str) -> str:
    results = exa.search_and_contents(
        query, type="auto", num_results=10, highlights={"max_characters": 4000}
    )
    return "\n".join([f"{r.title}: {r.url}" for r in results.results])


messages = [{"role": "user", "content": "What's the latest in AI safety?"}]
response = openai.chat.completions.create(
    model="gpt-4o", messages=messages, tools=tools
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    search_results = exa_search(json.loads(tool_call.function.arguments)["query"])
    messages.append(response.choices[0].message)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": search_results}
    )
    final = openai.chat.completions.create(model="gpt-4o", messages=messages)
    print(final.choices[0].message.content)
