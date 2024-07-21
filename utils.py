from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
import requests
from bs4 import BeautifulSoup
from langchain.tools import StructuredTool
from langchain_core.prompts import (PromptTemplate, MessagesPlaceholder)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


url = "https://huggingface.co/docs/hub/en/security-tokens"
text = """WoofWoofGPT is an all-round AI chatdog built with LangChain and Streamlit. Just to name a few of its capabilities:\n
- Document Question Answering with PDF files.\n
- Latest information using agentic tools.\n
- Summarise and generate new text.\n
- Assist in coding.\n
It is powered by :blue[Mixtral 8x7B language model] and :blue[HuggingFace's inference endpoint].\n
"""


template = """You are SearchGPT, a professional search engine who provides 
informative answers to users. Answer the following questions as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to give detailed, informative answers
Previous conversation history:
{chat_history}

New question: {input}
{agent_scratchpad}"""


def CNAheadlines(genre='singapore'):

    url = "https://www.channelnewsasia.com"
    response = requests.get(url)
    if response.status_code == 200:
        news = []
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find('body').find_all('h6')  # headlines at h6
        for x in headlines:
            news.append(x.text.strip())
        return '.'.join(news)
    else:
        return "No response from news provider."


news = StructuredTool.from_function(
    func=CNAheadlines,
    name="CNA_headlines",
    description="use this function to provide news headlines on the world, business and singapore."
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="DuckDuckGo",
    func=search.run,
    description="useful for when you need to answer questions about most current events including places, news and person",
)

tools = [search_tool, news]

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}

PROMPT = PromptTemplate(input_variables=[
                        "chat_history", "input", "agent_scratchpad"], template=template)

endpoint_error_message = "I'm sorry, HuggingFace endpoint has too many requests now. Please try again later."
model_error_message = "I'm sorry, the AI model is overloaded at the endpoint. Please try again later."


footer_html = """<div style='text-align: center;'>
<p style="font-size:70%;">Developed with ❤️ by Andy Oh</p>
<p style="font-size:70%;">Ngee Ann Polytechnic</p>
</div>"""
