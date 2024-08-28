from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

# ------- set up session state for question --------#
st.session_state.question_button = None

# ---- set up chat history ----#
news_chat_msgs = StreamlitChatMessageHistory(key="news_key")
news_chat_history_size = 3

financial_chat_msgs = StreamlitChatMessageHistory(key="financial_key")
financial_chat_history_size = 3

weather_chat_msgs = StreamlitChatMessageHistory(key="weather_key")
weather_chat_history_size = 3

schedule_chat_msgs = StreamlitChatMessageHistory(key="schedule_key")
schedule_chat_history_size = 3

# ---- set up creative chat history ----#
creative_chat_msgs = StreamlitChatMessageHistory(key="creative_key")
creative_chat_history_size = 5
