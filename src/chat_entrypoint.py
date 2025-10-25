from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

from controller_agent import get_controller_agent

_check_pointer = InMemorySaver()


@task
def _generate_response(messages: list[BaseMessage]) -> AIMessage:

    agent = get_controller_agent()

    result = agent.invoke({"messages": messages})

    return result["messages"][-1]


def clear_thread(thread_id: str):
    _check_pointer.delete_thread(thread_id)


@entrypoint(checkpointer=_check_pointer)
def get_history(
    _: str,
    *,
    previous: list[BaseMessage] | None = None
) -> list[BaseMessage]:

    history = previous or []

    return entrypoint.final(value=history, save=history)


@entrypoint(checkpointer=_check_pointer)
def chat(
    input: HumanMessage,
    *,
    previous: list[BaseMessage] | None = None,
) -> entrypoint.final[AIMessage, list[BaseMessage]]:

    messages = add_messages(previous or [], input)

    ai_message = _generate_response(messages).result()

    messages = add_messages(messages, ai_message)

    return entrypoint.final(value=ai_message, save=messages)
