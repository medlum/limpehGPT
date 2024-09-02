from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

# ------- set up session state for question --------#
if 'question_button' not in st.session_state:
    st.session_state.question_button = None


if 'selection' not in st.session_state:
    st.session_state.selection = None


def chat_msg_change():
    st.session_state.chat_msg_change = True


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

patient_data = {
    "name": "serlina eng",
    "gender": "female",
    "weight": "49.8kg",
    "height": "163cm",
    "bmi": "18.7 kg per square metre",
    "systolic": "119 mmHg",
    "diastolic": "67 mmHg",
    "waist": "65cm",
    "hip": "86cm",
    "total cholesterol": "212 mg/dL",
    "HDL": "121 mg/dL",
    "Direct LDL": "89 mg/dL",
    "triglyceride": "44 md/dL",
    "Glucose": "5.1 mmol/L"
}
