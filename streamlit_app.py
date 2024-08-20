import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from utils import *
import time
from huggingface_hub.utils._errors import HfHubHTTPError
from huggingface_hub.errors import OverloadedError
from langchain_huggingface import HuggingFaceEndpoint
import requests
from streamlit_lottie import st_lottie
from st_copy_to_clipboard import st_copy_to_clipboard
from duckduckgo_search.exceptions import RatelimitException
from huggingface_hub.inference._common import ValidationError
from streamlit_option_menu import option_menu
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain

# ---------set up page config -------------#
st.set_page_config(page_title="Cosmo-Chat-Dog",
                   layout="wide", page_icon="🐶")

# ---- set up session state for factual and creative button ---- #
if 'factual_mode' not in st.session_state:
    st.session_state.factual_mode = False

if 'creative_mode' not in st.session_state:
    st.session_state.creative_mode = False


def factual_mode_button():
    st.session_state.factual_mode = True  # factual memory on
    st.session_state.creative_mode = False  # creative memory off


def creative_mode_button():
    st.session_state.creative_mode = True  # creative memory on
    st.session_state.factual_mode = False  # factual memory off


# ---- set up factual chat history ----#
factual_chat_msgs = StreamlitChatMessageHistory(key="factual_key")
factual_chat_history_size = 10

if len(factual_chat_msgs.messages) == 0:
    factual_chat_msgs.clear()


# ---- set up creative chat history ----#
creative_chat_msgs = StreamlitChatMessageHistory(key="creative_key")
creative_chat_history_size = 10

if len(creative_chat_msgs.messages) == 0:
    creative_chat_msgs.clear()


# ---- App Header: Curiosity Starts Here ---- #

st.markdown("<p style='text-align: center; font-size:1.3rem; color:#2ecbf2'>Curiosity Starts Here</p>",
            unsafe_allow_html=True)

# ---- set up lottie icon ---- #
url = "https://lottie.host/ec0907dc-d6ac-4ecf-b267-e98d2b1c558d/eGhP7jwBj3.json"
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

# ----- Create creative and factual ------#

col1, col2 = st.columns(2)

creative_mode = col1.button(
    'Be Creative', use_container_width=True, key='creative',  on_click=creative_mode_button)

factual_mode = col2.button(
    'Be Factual', use_container_width=True, key='search', on_click=factual_mode_button)

# --------- llm model and question button ---------#
llama3p1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
st.session_state.question_button = None

# --------- factual button clicked -----------#
if st.session_state.factual_mode:

    # clear chat messages from creative mode
    creative_chat_msgs.clear()

    with st.sidebar:
        st.markdown(":blue[Select a factual prompt]")

        # create button for sample questions in factual_options
        with st.container(height=400):
            for q in factual_options:
                if st.button(q):
                    st.session_state.question_button = q

    # Set up LLM for factual mode
    llm = HuggingFaceEndpoint(
        repo_id=llama3p1_70B,
        max_new_tokens=1000,
        do_sample=False,
        temperature=0.01,
        repetition_penalty=1.2,
        return_full_text=False,
        top_p=0.2,
        top_k=40,
        huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
    )

    factual_conversational_memory = ConversationBufferMemory(
        memory_key='chat_history',
        chat_memory=factual_chat_msgs,
        k=factual_chat_history_size,
        return_messages=True)

    react_agent = create_react_agent(llm, tools, PROMPT)

    executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        memory=factual_conversational_memory,
        max_iterations=15,
        handle_parsing_errors=True,
        verbose=True,
        agent_kwargs=agent_kwargs,
    )

    if st.session_state.question_button:
        prompt = st.session_state.question_button
        # enable st.chat_input() at the bottom
        # when options from factual_options are selected
        st.session_state.question_button = st.chat_input(
            "Woof! Cosmo is in factual mode!", key='factual_prompt')

    else:
        prompt = st.chat_input(
            "Woof! Cosmo is in factual mode!", key='factual_prompt')

    if prompt:
        st.markdown(f":blue[{prompt.upper()}]")

        try:
            with st.spinner("Grrrr..."):
                response = executor.invoke({'input': f'{prompt}<|eot_id|>'})
                response = str(
                    response['output'].replace('<|eot_id|>', '').replace('<|eom_id|>', ''))

            def stream_data():
                for word in response.split(" "):
                    yield word + " "
                    time.sleep(0.04)

            with st.container(border=True, height=400):

                st.write_stream(stream_data)
                st_copy_to_clipboard(response)

        except HfHubHTTPError as error:
            st.write(endpoint_error_message)

        except OverloadedError as error:
            st.write(model_error_message)

        except RatelimitException as error:
            st.write(
                "Woof! I've reached rate limit for using DuckDuckGo to perform online search. Come back later...")
        except ValidationError as error:
            st.write(
                "Woof! I can't handle too much information, try again by reducing your request.")


# ----- creative button is clicked ------#
if st.session_state.creative_mode:
    # clear chat messages from factual mode
    factual_chat_msgs.clear()
    # create button for sample questions in factual_options
    with st.sidebar:
        st.markdown(":blue[Select a creative prompt]")
        # with st.container(height=400):

        for q in creative_options:
            if st.button(q):
                st.session_state.question_button = q

    # Set up LLM for creative mode
    llm = HuggingFaceEndpoint(
        repo_id=llama3p1_70B,
        task="text-generation",
        max_new_tokens=1000,
        do_sample=False,
        temperature=1.4,
        repetition_penalty=1.3,
        return_full_text=False,
        top_p=0.2,
        top_k=100,
        huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
    )

    # strings are located in msg.content
    for msg in creative_chat_msgs.messages:
        # remove assistant role
        msg.content = msg.content.replace('assistant', '')
        # regex on human to remove Humam
        human = re.search(r"Human:.*|human:.*", msg.content)
        if human is not None:
            # human.start() is index position 9
            msg.content = msg.content[:human.start()]

    # Langchain ConversationBufferMemory
    creative_conversational_memory = ConversationBufferMemory(
        memory_key='chat_history',
        chat_memory=creative_chat_msgs,
        k=creative_chat_history_size,
        return_messages=True
    )

    chatPrompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""
            You're a Cosmo a friendly chatdog. Always be helpful and thorough with your answers.
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

    # ------ set up llm chain -----#
    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=chatPrompt,
        verbose=True,
        memory=creative_conversational_memory,
    )

    if st.session_state.question_button:
        prompt = st.session_state.question_button
        # enable st.chat_input() at the bottom
        # when options from factual_options are selected
        st.session_state.question_button = st.chat_input(
            "Woof! Cosmo is in creative mode!", key='creative_prompt')

    else:
        prompt = st.chat_input(
            "Woof! Cosmo is in creative mode!", key='creative_prompt')

    if prompt:
        st.markdown(f":blue[{prompt.upper()}]")

        with st.spinner("Grrrr..."):
            prompt = f"{prompt} <|eot_id|>"
            # chain llm to the question
            response = chat_llm_chain.predict(
                human_input=prompt)
            # exclude 'assistant' from response
            response = response[9:]

            # regex on human to remove Humam
            human = re.search(r"Human:.*|human:.*", response)

            def stream_data():
                for word in response.split(" "):
                    yield word + " "
                    time.sleep(0.04)

            with st.container(border=True, height=400):

                if human is not None:
                    # exclude "Human:" located at end of string
                    response = response[:human.start()]
                    st.write_stream(stream_data)
                    st_copy_to_clipboard(response)

                else:
                    st.write_stream(stream_data)
                    st_copy_to_clipboard(response)


st.sidebar.write(footer_html, unsafe_allow_html=True)
