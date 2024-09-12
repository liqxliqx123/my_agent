import json
import os
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

os.getenv("TAVILY_API_KEY")
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, BaseMessage

# 定义工具
tools = [TavilySearchResults(max_result=2)]
# 初始化LLM并绑定工具
llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

my_web_messages = "my_web_messages"


# 定义State 字典
class State(TypedDict):
    my_web_messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {my_web_messages: [llm_with_tools.invoke(state[my_web_messages])]}


# 工具调用方法
class BasicToolNode:
    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messags := inputs.get(my_web_messages):
            message = messags[-1]
        else:
            raise ValueError("No messages found in inputs")
        # 工具的输出结果
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(content=json.dumps(tool_result), name=tool_call["name"], tool_call_id=tool_call["id"]))
        return {my_web_messages: outputs}


# 路由函数
def route_tools(state: State) -> Literal["tools", "__end__"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif msg := state.get(my_web_messages, []):
        ai_message = msg[-1]
    else:
        raise ValueError("No messages found in state")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


def main():
    # 初始化状态图
    graph_builder = StateGraph(State)
    # 添加起点边
    graph_builder.add_edge(START, "chatbot")
    # 添加聊天节点
    graph_builder.add_node("chatbot", chatbot)
    # 添加工具调用节点
    graph_builder.add_node("tools", BasicToolNode(tools=tools))
    # 添加条件边
    graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", "__end__": END})
    # 添加工具调用完成边
    graph_builder.add_edge("tools", "chatbot")

    # 编译图
    graph = graph_builder.compile()
    # 生成可视化图
    from IPython.display import Image, display
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception as e:
        print("Failed to generate graph image:", e)

    # 运行图
    while True:
        # 获取用户输入
        user_input = input("User: ")

        # 如果用户输入 "quit"、"exit" 或 "q"，则退出循环，结束对话
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")  # 打印告别语
            break  # 退出循环

        # 使用 graph.stream 处理用户输入，并生成机器人的回复
        # "messages" 列表中包含用户的输入，传递给对话系统
        for event in graph.stream({my_web_messages: [("user", user_input)]}):

            # 遍历 event 的所有值，检查是否是 BaseMessage 类型的消息
            for value in event.values():
                if isinstance(value[my_web_messages][-1], BaseMessage):
                    # 如果消息是 BaseMessage 类型，则打印机器人的回复
                    print("Assistant:", value[my_web_messages][-1].content)


if __name__ == "__main__":
    main()
