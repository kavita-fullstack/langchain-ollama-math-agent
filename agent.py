"""A tiny LangChain agent with one multiplication tool.

Install the Python integration with: python -m pip install langchain-ollama
Install Ollama, then download a model with: ollama pull qwen2.5:3b
Run the agent with: python agent.py
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


model = ChatOllama(model="qwen2.5:3b", temperature=0)
agent = create_agent(
    model=model,
    tools=[multiply],
    system_prompt=(
        "You are a helpful math assistant. Use the multiply tool for "
        "multiplication questions, then explain the result briefly."
    ),
)

print("Ask a multiplication question (or type 'quit' to stop).")
while True:
    question = input("You: ").strip()
    if question.lower() in {"quit", "exit"}:
        break
    if not question:
        continue

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("Agent:", result["messages"][-1].content)
