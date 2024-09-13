import json
import os
from langchain_openai import ChatOpenAI
import sqlite3

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

os.getenv("TAVILY_API_KEY")
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, BaseMessage

import pandas as pd

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

    # 创建memery saver 检查点
    memory = MemorySaver()
    # 创建sqlite检查点
    # conn = sqlite3.connect("checkpoints.sqlite")
    # memory = SqliteSaver(conn)
    # SQLite objects created in a thread can only be used in that same thread

    # 添加断点
    interrupt_nodes = ["tools"]
    # interrupt_nodes = None
    # 编译图
    graph = graph_builder.compile(
        checkpointer=memory,
        interrupt_before=interrupt_nodes  # 在工具调用之前中断，执行手动操作
    )
    # 生成可视化图
    # from IPython.display import Image, display
    # try:
    #     display(Image(graph.get_graph().draw_mermaid_png()))
    # except Exception as e:
    #     print("Failed to generate graph image:", e)

    # 配置当前会话
    config = {"configurable": {"thread_id": 1}}
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
        events = graph.stream(
            {my_web_messages: [("user", user_input)]},
            config=config,  # 用于指定线程配置， 包括线程id
            stream_mode="values"  # 返回流式数据的值
        )
        # 遍历 event 的所有值，检查是否是 BaseMessage 类型的消息
        for event in events:
            # 如果消息是 BaseMessage 类型，则打印机器人的回复
            print("Assistant:", event[my_web_messages][-1])

        # 查看断点
        if len(interrupt_nodes) > 0:
            # 获取当前会话的状态
            snapshot = graph.get_state(config)
            print("Next node:", snapshot.next)
            if "tools" in snapshot.next:
                current_msg = snapshot.values[my_web_messages][-1]
                print("Current tool-calls message:", current_msg.tool_calls)

            # 恢复执行图
            # interrupt_events = graph.stream(None, config, stream_mode="values")
            # for interrupt_event in interrupt_events:
            #     if my_web_messages in interrupt_event:
            #         interrupt_event[my_web_messages][-1].pretty_print()

            # 手动生成新tool message
            tool_call_id = snapshot.values[my_web_messages][-1].tool_calls[0]["id"]
            print("tool_call_id: ", tool_call_id)
            tool_message = ToolMessage(
                content="这是人类干预生成的信息: sorry I do not known",  # 工具调用返回的内容
                tool_call_id=tool_call_id  # 关联工具调用的 ID
            )
            # 更新会话状态，加入工具调用结果
            graph.update_state(config, {my_web_messages: [tool_message]})
            events = graph.stream(None, config, stream_mode="values")
            for event in events:
                if my_web_messages in event:
                    event[my_web_messages][-1].pretty_print()
    # 查看当前会话的状态
    # snapshot = graph.get_state(config)
    # msgs = snapshot.values[my_web_messages]
    # df = pd.DataFrame([{
    #     "content": msg.content,
    #     "msg_id": msg.id,
    #     "type": type(msg).__name__,
    #     "token_usage": msg.response_metadata.get("token_usage") if hasattr(msg, "response_metadata") else None
    # }] for msg in msgs)
    # print(df)

    # 查看当前会话的历史
    # 遍历历史记录，打印每个状态中的所有消息
    history = graph.get_state_history(config)
    # 使用集合存储已处理过的消息 ID
    seen_message_ids = set()
    for state in history:
        # 获取状态中的消息列表
        messages = state.values.get(my_web_messages, [])
        # 检查是否存在至少一条未处理的 BaseMessage 类型的消息
        valid_messages = [msg for msg in messages if isinstance(msg, BaseMessage) and msg.id not in seen_message_ids]
        if valid_messages:
            print("=== 对话历史 ===")
            # 遍历每个状态中的消息记录
            for message in valid_messages:
                # 根据消息类型区分用户与机器人
                if message.content:
                    print(message.content)
        else:
            print("=== 对话历史为空 ===")


if __name__ == "__main__":
    main()
