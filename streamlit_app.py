import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from utils import *
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub.utils._errors import HfHubHTTPError
from huggingface_hub.errors import OverloadedError

st.set_page_config(page_title="LimPehGPT: Chat with search", page_icon="🧔")
container = st.container(border=True)
container.title("🧔 LimPehGPT  ")

# ---- set up history for chat and document messages ----#
chat_msgs = StreamlitChatMessageHistory(key="special_app_key")
chat_history_size = 3
doc_msgs = StreamlitChatMessageHistory()

if len(doc_msgs.messages) == 0:
    doc_msgs.clear()
    doc_msgs.add_ai_message(
        "How can I help you?")

if len(chat_msgs.messages) == 0:
    chat_msgs.clear()
    chat_msgs.add_ai_message(
        "How can I help you?")

# callback function for clear history button


def clear_history():
    chat_msgs.clear()
    doc_msgs.clear()


# Enable chat agent and conversational memory with upload_files = False
uploaded_files = False

# Create widgets for sidebar
with st.sidebar:
    huggingfacehub_api_token = st.text_input(
        "Hugging Face Access Token", type="password")

    # if not huggingfacehub_api_token:
    #    st.info("Please add your HuggingFace access token to continue.")
    #    st.stop()

    st.button("Clear message history", on_click=clear_history)

    if st.toggle("Activate file uploader"):
        uploaded_files = st.file_uploader(
            label='Upload PDF file', type=["pdf"], accept_multiple_files=True, on_change=doc_msgs.clear)

    with st.expander('About the chatbot'):
        st.write(text % url)

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
            msg.content.replace('</s>', ''))

    if prompt := st.chat_input("What are the headlines in Singapore?"):
        st.chat_message("human").write(prompt)

        try:
            with st.spinner("Thinking..."):
                response = executor.invoke({'input': prompt})
                response = response['output'].replace('</s>', '')

            def stream_data():
                for word in response.split(" "):
                    yield word + " "
                    time.sleep(0.1)

            st.chat_message("ai").write_stream(stream_data)
            # st.chat_message("ai").write(response)
        except HfHubHTTPError as error:
            st.write(endpoint_error_message)

        except OverloadedError as error:
            st.write(model_error_message)


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
            msg.content.replace('</s>', ''))

    if user_query := st.chat_input(placeholder="Ask me anything about the document!"):
        st.chat_message("user").write(user_query)

        try:
            with st.spinner("Thinking..."):
                response = qa_chain.run(user_query)

            def stream_data():
                for word in response.split(" "):
                    yield word + " "
                    time.sleep(0.1)
            st.chat_message("ai").write_stream(stream_data)
            # st.chat_message("ai").write(response)

        except HfHubHTTPError as error:
            st.write(endpoint_error_message)

        except OverloadedError as error:
            st.write(model_error_message)

st.sidebar.write(footer_html, unsafe_allow_html=True)
