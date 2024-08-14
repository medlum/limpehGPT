import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from utils import *
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub.utils._errors import HfHubHTTPError
from huggingface_hub.errors import OverloadedError
from langchain_huggingface import HuggingFaceEndpoint
import requests
from streamlit_lottie import st_lottie
from st_copy_to_clipboard import st_copy_to_clipboard
from duckduckgo_search.exceptions import RatelimitException
from huggingface_hub.inference._common import ValidationError
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Cosmo the ChatDog",
                   layout="wide", page_icon="🐶")

# ---- set up history for chat and document messages ----#
chat_msgs = StreamlitChatMessageHistory(key="special_app_key")
chat_history_size = 3
doc_msgs = StreamlitChatMessageHistory()

if len(doc_msgs.messages) == 0:
    doc_msgs.clear()
    doc_msgs.add_ai_message(
        "You have uploaded a PDF document, how can I help?")

if len(chat_msgs.messages) == 0:
    chat_msgs.clear()


# --- callback function for clear history button ----#

def clear_history():
    chat_msgs.clear()
    doc_msgs.clear()
    st.session_state.selection = None

# --- callback function to clear selectbox for sample questions ----#


def clear_selectbox():
    st.session_state.selection = None


# ---- lottie files ---- #
url = "https://lottie.host/4ef3b238-96dd-4078-992a-50f5a41d255c/mTUUT5AegN.json"
url = "https://lottie.host/ec0907dc-d6ac-4ecf-b267-e98d2b1c558d/eGhP7jwBj3.json"
url = requests.get(url)

url_json = dict()
if url.status_code == 200:
    url_json = url.json()
else:
    print("Error in the URL")

st_lottie(url_json,
          # change the direction of our animation
          reverse=True,
          # height and width of animation
          height=250,
          width=250,
          # speed of animation
          speed=1,
          # means the animation will run forever like a gif, and not as a still image
          loop=True,
          # quality of elements used in the animation, other values are "low" and "medium"
          quality='high',
          # This is just to uniquely identify the animation
          key='bot'
          )


uploaded_files = False

# Create widgets for sidebar
with st.sidebar:
    # create sample questions
    prompt = ""

    def on_change(key):
        selection = st.session_state[key]
        # st.write(f"Selection changed to {selection}")
    # https://icons.getbootstrap.com/
    selected = option_menu("Cosmo", ["Questions", "Clear Chat", 'Upload File', 'About Cosmo'],
                           icons=["list-task", 'bi-archive',
                                  "bi-cloud-upload", 'gear'],
                           menu_icon="bi-robot",
                           default_index=1,
                           key='menu_5',
                           on_change=on_change,
                           styles={
        # "container": {"padding": "0!important"},
        "menu-title": {"font-size": "20px"},
        "icon": {"color": "orange", "font-size": "15px"},
        "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        # "nav-link-selected": {"background-color": "grey"},
    })

    if selected == "About Cosmo":
        st.write(text)

    elif selected == "Questions":
        prompt = st.selectbox(label="",
                              options=options,
                              placeholder="Select a sample question",
                              key="selection",
                              index=None,
                              )
    elif selected == "Upload File":
        uploaded_files = st.file_uploader(
            label='Upload PDF file', type=["pdf"],
            accept_multiple_files=True,
            on_change=doc_msgs.clear)

    elif selected == "Clear Chat":
        clear_history()


model_mistral8B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llama3_70B = "meta-llama/Meta-Llama-3-70B-Instruct"
llama3p1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
llm = HuggingFaceEndpoint(
    repo_id=llama3p1_70B,
    max_new_tokens=700,
    do_sample=False,
    temperature=0.01,
    repetition_penalty=1.1,
    return_full_text=False,
    top_p=0.2,
    top_k=40,
    huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
)

# If no files are uploaded, llm will be used for agent tools.
if not uploaded_files:

    conversational_memory = ConversationBufferMemory(
        memory_key='chat_history',
        chat_memory=chat_msgs,
        k=chat_history_size,
        return_messages=True)

    react_agent = create_react_agent(llm, tools, PROMPT)

    executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        memory=conversational_memory,
        max_iterations=10,
        # return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=True,
        agent_kwargs=agent_kwargs,
    )

    for msg in chat_msgs.messages:
        st.chat_message(msg.type).write(
            msg.content.replace('<|eot_id|>', ''))

    if prompt := st.chat_input("Woof! Ask me a question or choose one from the side bar!", on_submit=clear_selectbox) or prompt:
        st.chat_message("human").write(prompt)

        try:
            with st.spinner("Grrrr..."):
                response = executor.invoke({'input': prompt})
                response = str(response['output'].replace('<|eot_id|>', ''))

            def stream_data():
                for word in response.split(" "):
                    yield word + " "
                    time.sleep(0.04)

            st.chat_message("ai").write_stream(stream_data)
            st_copy_to_clipboard(response)
            # st.chat_message("ai").write(response)
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


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


if uploaded_files:
    chat_msgs.clear()
    retriever = configure_retriever(uploaded_files)

    # Setup memory for contextual conversation

    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=doc_msgs, return_messages=True)

    # Setup LLM and QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    avatars = {"human": "user", "ai": "assistant"}
    for msg in doc_msgs.messages:
        st.chat_message(avatars[msg.type]).write(
            msg.content.replace('<|eot_id|>', ''))

    if user_query := st.chat_input(placeholder="Ask me about the document..."):
        st.chat_message("user").write(user_query)

        try:
            with st.spinner("Grrrr..."):
                response = qa_chain.run(user_query)

            def stream_data():
                for word in response.split(" "):
                    yield word + " "
                    time.sleep(0.04)
            st.chat_message("ai").write_stream(stream_data)
            st_copy_to_clipboard(response)
            # st.chat_message("ai").write(response)

        except HfHubHTTPError as error:
            st.write(endpoint_error_message)

        except OverloadedError as error:
            st.write(model_error_message)

        except ValidationError as error:
            st.write("Max out tokens as there are too much data to process ")


st.sidebar.write(footer_html, unsafe_allow_html=True)
