from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from IPython.display import display, Image
import os

# 开启 LangSmith 跟踪，便于调试和查看详细执行信息
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph ChatBot"

message_key = "my_messages"
chat_model = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    my_messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {message_key: [chat_model.invoke(state[message_key])]}


def main():
    # 创建状态图对象
    graph = StateGraph(State)
    # 添加聊天节点
    graph.add_node("chatbot", chatbot)

    # 添加对话流程
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    # 编译图
    graph = graph.compile()

    # 流程可视化
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except  Exception:
        pass

    # 运行图
    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        for event in graph.stream({message_key: [user_input]}):
            for value in event.values():
                print(f"Assistant: {value[message_key][-1].content}")


if __name__ == "__main__":
    main()
