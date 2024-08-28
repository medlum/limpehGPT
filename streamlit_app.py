import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from utils import *
import time
from huggingface_hub.utils._errors import HfHubHTTPError
from huggingface_hub.errors import OverloadedError
from langchain_huggingface import HuggingFaceEndpoint
from st_copy_to_clipboard import st_copy_to_clipboard
from huggingface_hub.inference._common import ValidationError
from langchain.schema import (
    SystemMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain
import streamlit_antd_components as sac
import datetime
from comp_ticker_btn_options import *
from comp_state_chathistory import *
from comp_sidebar_schedule import *
# ---------set up page config -------------#
st.set_page_config(page_title="Cosmo-Chat-Dog",
                   layout="wide", page_icon="🐶")

# --- session state and chat history are initialize in component_session_state ----#
if len(creative_chat_msgs.messages) == 0:
    creative_chat_msgs.clear()

# ---- App Header: INTELLIGENCE Starts Here ---- #

# st.markdown("<p style='text-align: center; font-size:1.4rem; color:#2ecbf2'>INTELLIGENCE STARTS HERE</p>",
#            unsafe_allow_html=True)

# -----set up news ticker ------#
news = CNAheadlines("news")  # utils
breakingnews(news)  # component_sidebar

# ----- set up mode button -----#
btn = mode_button()  # component_sidebar

# --------- llm model and question button ---------#
llama3p1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# -------- reset selectbox selection ---------#


def reset_selectbox():
    # key = 'selection' is found in st.selectbox
    # which is the session_state variable
    st.session_state.selection = None

# --------- set up for questions, chat messages, agent tool when a mode button: btn is clicked -----------#


if btn in ["news", "weather", "finance", "schedule"]:
    # if st.session_state.factual_mode:

    # clear chat messages from creative mode
    creative_chat_msgs.clear()

    if btn == "schedule":
        questions = schedule_options
        chat_msg = schedule_chat_msgs
        chat_history_size = schedule_chat_history_size
        agent_tools = tools_for_schedule
        creative_chat_msgs.clear()
        financial_chat_msgs.clear()
        weather_chat_msgs.clear()
        news_chat_msgs.clear()

    if btn == "news":
        questions = news_options
        chat_msg = news_chat_msgs
        chat_history_size = news_chat_history_size
        agent_tools = tools_for_news
        creative_chat_msgs.clear()
        financial_chat_msgs.clear()
        weather_chat_msgs.clear()
        schedule_chat_msgs.clear()

    elif btn == "finance":
        questions = financial_options
        chat_msg = financial_chat_msgs
        chat_history_size = financial_chat_history_size
        agent_tools = tools_for_stock
        creative_chat_msgs.clear()
        news_chat_msgs.clear()
        weather_chat_msgs.clear()
        schedule_chat_msgs.clear()

    elif btn == "weather":
        questions = weather_options
        chat_msg = weather_chat_msgs
        chat_history_size = weather_chat_history_size
        agent_tools = tools_for_weather
        creative_chat_msgs.clear()
        financial_chat_msgs.clear()
        news_chat_msgs.clear()
        schedule_chat_msgs.clear()

    chat_msg.clear()

    with st.sidebar:
        st.session_state.question_button = st.selectbox(label="",
                                                        options=questions,
                                                        placeholder=f"Try a question related to {btn}...",
                                                        key="selection",
                                                        index=None,
                                                        )
        # setup schedule widgets
        if btn == "schedule":

            schedule_widgets()

    # Set up LLM for factual mode
    llm_factual = HuggingFaceEndpoint(
        repo_id=llama3p1_70B,
        max_new_tokens=1500,
        do_sample=False,
        temperature=0.01,
        repetition_penalty=1.1,
        return_full_text=False,
        top_p=0.2,
        top_k=40,
        huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
    )

    conversational_memory = ConversationBufferMemory(
        memory_key='chat_history',
        chat_memory=chat_msg,
        k=chat_history_size,
        return_messages=True)

    react_agent = create_react_agent(llm_factual, agent_tools, PROMPT)

    executor = AgentExecutor(
        agent=react_agent,
        tools=agent_tools,
        memory=conversational_memory,
        max_iterations=10,
        handle_parsing_errors=True,
        verbose=True,
        agent_kwargs=agent_kwargs,
    )

    if st.session_state.question_button:
        prompt = st.session_state.question_button
        # enable st.chat_input() at the bottom
        # when options from factual_options are selected
        st.session_state.question_button = st.chat_input(
            f"Ask a question in {btn} mode", key='factual_prompt', on_submit=reset_selectbox)

    else:
        prompt = st.chat_input(
            f"Ask a question in {btn} mode", key='factual_prompt')

    with st.container(border=True, height=200):
        if prompt:
            st.markdown(f":red[{prompt.upper()}]")
            try:
                with st.spinner("Grrrr..."):
                    response = executor.invoke(
                        {'input': f'{prompt}<|eot_id|>'})
                    response = str(
                        response['output'].replace('<|eot_id|>', ''))
                    st.markdown(response, unsafe_allow_html=True)

                    print(response)


#                    def stream_data():
#                        for word in response.split(" "):
#                            yield word + " "
#                            time.sleep(0.02)
#
#                    st.write_stream(stream_data)

                    st_copy_to_clipboard(response)

            except HfHubHTTPError as error:
                st.write(endpoint_error_message)

            except OverloadedError as error:
                st.write(model_error_message)

            except ValidationError as error:
                st.write(
                    "Woof! I can't handle too much information, try again by reducing your request.")


# ----- creative button is clicked ------#
if btn == "Creative".lower():
    news_chat_msgs.clear()
    financial_chat_msgs.clear()
    weather_chat_msgs.clear()

    # Set up LLM for creative mode
    llm_creative = HuggingFaceEndpoint(
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
        llm=llm_creative,
        prompt=chatPrompt,
        verbose=True,
        memory=creative_conversational_memory,
    )
    with st.sidebar:
        st.session_state.question_button = st.selectbox(label="",
                                                        options=creative_options,
                                                        placeholder="Try a creative question...",
                                                        key="selection",
                                                        index=None,
                                                        )
    # wait for selectbox option
    if st.session_state.question_button:
        prompt = st.session_state.question_button
        # enable st.chat_input() when selectbox option are selected
        st.session_state.question_button = st.chat_input(
            f"Ask a question in {btn} mode", key='creative_prompt', on_submit=reset_selectbox)
    # else wait for chat_input
    else:
        prompt = st.chat_input(
            f"Ask a question in {btn} mode", key='creative_prompt')

    with st.container(border=True, height=200):

        if prompt:
            st.markdown(f":red[{prompt.upper()}]")

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
                        time.sleep(0.02)

                if human is not None:
                    # exclude "Human:" located at end of string
                    response = response[:human.start()]
                    st.write_stream(stream_data)
                    st_copy_to_clipboard(response)

                else:
                    st.write_stream(stream_data)
                    st_copy_to_clipboard(response)

                st.session_state.question_button = None

hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.sidebar.write(footer_html, unsafe_allow_html=True)

