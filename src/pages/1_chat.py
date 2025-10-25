from typing import cast
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
import streamlit as st

from chat_entrypoint import chat, clear_thread, get_history
from utils import to_ui_msg


st.title("💬 ChatNFB - NutriFoodBot")
st.caption(
    "🚀 Chat with AI and ask anything about food."
)

thread_id = cast(str, st.session_state["thread_id"])
user_id = cast(str, st.session_state["user_id"])

if st.sidebar.button("🧹 Clear Chat History"):
    clear_thread(thread_id)
    if "messages" in st.session_state:
        del st.session_state.messages
    st.rerun()

config = {
    "configurable": {
        "thread_id": thread_id,
        "user_id": user_id,
    },
}

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hello! I'm NutriFoodBot. "
            "How can I assist you with food today?"
        ),
    }]

    if history := get_history.invoke("dummy", config=config):
        st.session_state.messages += [to_ui_msg(m) for m in history]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle quick prompts from sidebar buttons
if "quick_prompt" in st.session_state and st.session_state.quick_prompt:
    prompt = st.session_state.quick_prompt
    st.session_state.quick_prompt = None  # Clear the quick prompt

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()  # Prevents ghost of prior chat output

        with st.spinner("Thinking..."):
            ai_message = cast(
                AIMessage,
                chat.invoke(
                    HumanMessage(content=prompt),
                    config=cast(RunnableConfig, config),
                ),
            )

        st.session_state.messages.append(to_ui_msg(ai_message))
        placeholder.write(ai_message.content)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()  # Prevents ghost of prior chat output

        with st.spinner("Thinking..."):
            ai_message = cast(
                AIMessage,
                chat.invoke(
                    HumanMessage(content=prompt),
                    config=cast(RunnableConfig, config),
                ),
            )

        print(ai_message)
        st.session_state.messages.append(to_ui_msg(ai_message))
        placeholder.write(ai_message.content)
