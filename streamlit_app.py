import streamlit as st
from src.agent.f1_agent import F1Agent

st.set_page_config(page_title="F1 Assistant", page_icon="🏎️")
st.title("🏎️ F1 Conversational Assistant")

@st.cache_resource
def get_agent():
    return F1Agent()

agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about Formula 1..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.handle_query(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
