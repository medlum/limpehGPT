from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEndpoint
import requests
from bs4 import BeautifulSoup
from langchain.tools import StructuredTool
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(page_title="LimPehGPT: Chat with search", page_icon="🦜")
st.title("🧔 LimPehGPT")

url = "https://huggingface.co/docs/hub/en/security-tokens"
text = """Hello there!\n
LimPeh is an all-round AI chatbot built with LangChain and Streamlit.\n
It is powered by :blue[Mixtral 8x7B language model] and :blue[HuggingFace's inference endpoint].\n
Equipped with agentic tools, it can also provide the latest information and news.\n
To get started, head to HuggingFace to obtain an access token.\n
For more information: [HuggingFace Token Documentation](%s).
"""
with st.expander(':blue[Introduction]'):
    st.write(text % url)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token.replace('</s>', '')
        self.container.markdown(self.text)


def CNAheadlines(genre: str):
    """
    genere: 'business, world, singapore'
    """
    url = "https://www.channelnewsasia.com"
    response = requests.get(url)
    if response.status_code == 200:
        news = []
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find('body').find_all('h6')  # headlines at h6
        for x in headlines:
            if x.select(f"a[href*={genre}]"):
                news.append(x.text.strip())
                # yield x.text.strip()
                # time.sleep(0.1)

        return '.'.join(news)
    else:
        # yield "No response from news provider."
        return "No response from news provider."


news = StructuredTool.from_function(
    func=CNAheadlines,
    name="CNA_headlines",
    description="use this function to provide headlines on the world, business and singapore."
)

huggingfacehub_api_token = st.sidebar.text_input(
    "Hugging Face Access Token", type="password")

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")


for msg in msgs.messages:
    st.chat_message(msg.type).write(
        msg.content.replace('</s>', ''))

if prompt := st.chat_input(placeholder="What is the latest news on Ukraine?"):
    st.chat_message("user").write(prompt)

    if not huggingfacehub_api_token:
        st.info("Please add your HuggingFace access token to continue.")
        st.stop()

    model_mistral8B = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    llm = HuggingFaceEndpoint(
        repo_id=model_mistral8B,
        max_new_tokens=500,
        do_sample=False,
        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=False,
        top_p=0.2,
        top_k=40,
        streaming=True,
        huggingfacehub_api_token=huggingfacehub_api_token
    )
    tools = [DuckDuckGoSearchRun(
        name="duckduckgo", description="use this function to find the latest information."), news]

    system_message = "Your name is LimPeh, a helpful and friendly chatbot."
    human_message = "Number each of the news headlines and put them in bullet points."

    chat_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm, tools=tools, system_message=system_message)

    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        # return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=True,

    )

    with st.spinner("Thinking..."):
        with st.chat_message("assistant"):
            # response = executor.run(
            #    prompt, callbacks=[StreamHandler(st.empty())])
            # st.write(response)

            response = executor.invoke(prompt)
            st.write(response["output"])


footer_html = """<div style='text-align: center;'>
<p style="font-size:70%;">Developed with ❤️ by Andy Oh</p>
</div>"""
st.markdown(footer_html, unsafe_allow_html=True)
