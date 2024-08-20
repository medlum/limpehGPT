import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    SystemMessage,
)
import requests
from streamlit_lottie import st_lottie
import re

st.set_page_config(page_title="The Quiz Bot",
                   layout="wide", page_icon="🤖")

# HuggingFace Modelcard
model_mistral8B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llama3_70B = "meta-llama/Meta-Llama-3-70B-Instruct"
llama3p1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"


llm = HuggingFaceEndpoint(
    repo_id=llama3p1_70B,
    # task="text-generation",
    max_new_tokens=1000,
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.1,
    return_full_text=False,
    top_p=0.4,
    top_k=100,
    huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
)

# You are a chatbot who specialized in quiz questions on Python programming language for beginners.
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            - You are a chatbot that specialized in quiz questions on General knowledge, Python programming, Microsoft Excel, Statistics and Economics for beginners. 
            - Always begin your conversation by asking which topic the user would like to quiz.
            - The quiz should contain different levels of difficulty.
            - Keep track of the number of right and wrong answers.
            - Review the strength and weakness at the end of the quiz.
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


# ---- set up lottie icon ---- #
# python icon
url = "https://lottie.host/afd755b7-2ead-4ac6-a75e-02b05054871e/SKQzuvxmW2.json"
# quiz icon
url = "https://lottie.host/1897011a-cf70-491f-8618-82c16d5c2fa2/d4LAW696Ly.json"
url = requests.get(url)
url_json = dict()

if url.status_code == 200:
    url_json = url.json()
else:
    print("Error in the URL")

st_lottie(url_json,
          reverse=True,  # change the direction
          height=230,  # height and width
          width=230,
          speed=1,  # speed
          loop=True,  # run forever like a gif
          quality='high',  # options include "low" and "medium"
          key='bot'  # Uniquely identify the animation
          )

# ----- page header -----#
# st.markdown("<p style='text-align: left; font-size:2rem'>The Quiz Show</p>",
#            unsafe_allow_html=True)

# ----- set up chat history to retain memory-----#

# set up chat history in streamlit as session state does not work
chat_msgs = StreamlitChatMessageHistory(key="special_app_key")
# set history size
chat_history_size = 10

# if len of chat history is 0, clear and add first message
if len(chat_msgs.messages) == 0:
    chat_msgs.clear()
    chat_msgs.add_ai_message(
        """Hi there! What is your name?""")

# handle LLMA3.1 prompt format: 'human', 'assistant'
# refer to https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/

# strings are located in msg.content
for msg in chat_msgs.messages:
    # remove assistant role
    msg.content = msg.content.replace('assistant', '')
    # regex on human to remove Humam
    human = re.search(r"Human:.*|human:.*", msg.content)
    if human is not None:
        # human.start() is index position 9
        msg.content = msg.content[:human.start()]
        # remove <|eot_id|> before writing to chat history
        st.chat_message(msg.type).write(msg.content.replace(
            '<|eot_id|>', ''))
    else:
        st.chat_message(msg.type).write(msg.content.replace(
            '<|eot_id|>', ''))

# Langchain ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=chat_msgs,
    k=chat_history_size,
    return_messages=True
)

# ------ set up llm chain -----#
chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

# ------ generate questions and responses ------#

if question := st.chat_input("Your Answer..."):
    with st.spinner("Grrrr..."):
        # question needs to include <|eot_id|> to end interaction
        # refer to llama3.1 prompt format
        question = f"{question} <|eot_id|>"
        # chain llm to the question
        response = chat_llm_chain.predict(
            human_input=question)
        # exclude 'assistant' from response
        response = response[9:]

        # regex on human to remove Humam
        human = re.search(r"Human:.*|human:.*", response)

        if human is not None:
            # exclude "Human:" located at end of string
            response = response[:human.start()]
            st.chat_message("ai").write(response)
        else:
            st.chat_message("ai").write(response)
