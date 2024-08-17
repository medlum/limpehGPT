import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain

from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage
)
import time
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Cosmo the QuizDog",
                   layout="wide", page_icon="🐶")

model_mistral8B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llama3_70B = "meta-llama/Meta-Llama-3-70B-Instruct"
llama3p1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
quant = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"

llm = HuggingFaceEndpoint(
    repo_id=llama3p1_70B,
    # task="text-generation",
    max_new_tokens=500,
    do_sample=False,
    temperature=0.01,
    repetition_penalty=1.1,
    return_full_text=False,
    top_p=0.2,
    top_k=40,
    huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
)


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
           You are a chatbot who specialized in give quiz questions on Python programming language for beginners.
           Keep track of the number of right and wrong answers.
           Always randomize your questions.
           """
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)
chat_msgs = StreamlitChatMessageHistory(key="special_app_key")
chat_history_size = 10

memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=chat_msgs,
    k=chat_history_size,
    return_messages=True
)

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)


# ---- set up lottie icon ---- #

url = "https://lottie.host/afd755b7-2ead-4ac6-a75e-02b05054871e/SKQzuvxmW2.json"
url = requests.get(url)


url_json = dict()
if url.status_code == 200:
    url_json = url.json()
else:
    print("Error in the URL")

st_lottie(url_json,
          reverse=True,  # change the direction
          height=130,  # height and width
          width=130,
          speed=1,  # speed
          loop=True,  # run forever like a gif
          quality='high',  # options include "low" and "medium"
          key='bot'  # Uniquely identify the animation
          )

st.markdown("<p style='text-align: left; font-size:2rem'>The Python's Quiz Show</p>",
            unsafe_allow_html=True)

for msg in chat_msgs.messages:
    st.chat_message(msg.type).write(
        msg.content.replace('<|eot_id|>', ''))


if question := st.chat_input("Your Answer..."):
    with st.spinner("Grrrr..."):
        response = chat_llm_chain.predict(human_input=question+"<|eot_id|>")
        st.chat_message("ai").write(response)
