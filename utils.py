from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
import requests
from bs4 import BeautifulSoup
from langchain.tools import StructuredTool
from langchain_core.prompts import (PromptTemplate, MessagesPlaceholder)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit
from langchain_community.utilities.polygon import PolygonAPIWrapper
import yfinance as yf
import re
import altair as alt
import streamlit as st
import json

url = "https://huggingface.co/docs/hub/en/security-tokens"
text = """🐶 is an all-round AI chatdog built with LangChain and Streamlit. Some of its woofwoof capabilities:\n
- Document-Question-Answering\n
- Sinagpore News Headlines\n
- US Stock Price Quotes\n
- Singapore Weather Forecast\n
- Summarize|Generate Texts\n
- Coding Assistance\n
🐶 is powered by Mixtral 8x7B language model and HuggingFace🤗 inference endpoint.\n
"""


template = """You are Cosmo the chatdog, a professional search engine who provides 
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


options = ("Latest headlines in Singapore",
           "Closing price of Nvidia's in the past 5 days",
           "Trendline of Nvidia's stock price",
           "Top 5 financial metrics of Nvidia",
           "How is the weather today?",
           "Weather forecast in the next few days",
           "Will it rain in the west of Singapore tomorrow?",
           "Who is the Prime Minister of United Kingdom?"
           )


def weather4days(url):
    url = "https://api-open.data.gov.sg/v2/real-time/api/four-day-outlook"
    res = requests.get(url)
    data = json.dumps(res.json(), indent=4)
    return data


weather4days_tool = StructuredTool.from_function(
    func=weather4days,
    name='nea_api_4days',
    description="useful for when you need to find out weather outlook in the next 4 days in singapore"
)


def weather24hr(url):
    url = "https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast"
    res = requests.get(url)
    data = json.dumps(res.json(), indent=4)
    return data


weather24hr_tool = StructuredTool.from_function(
    func=weather24hr,
    name='nea_api_24hr',
    description="useful for when you need to find out the next 24 hour weather in singapore"
)


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


news_tool = StructuredTool.from_function(
    func=CNAheadlines,
    name="CNA_headlines",
    description="use this function to provide breaking news, headlines on the world, business and singapore."
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="DuckDuckGo",
    func=search.run,
    description="use this function to answer questions about most current events including places, news and person",
)

pattern = r'[A-Z]+\d+[A-Z]*\.SI|[A-Z]+\b'


def stockPrice(ticker: str) -> str:
    """
    Download stock prices
    """
    matches = re.findall(pattern, ticker)
    ticker = ''.join(matches)
    tick = yf.Ticker(ticker)
    price = tick.history()
    return price


stockPrice_tool = StructuredTool.from_function(
    func=stockPrice,
    name='yfinance',
    description="use this function strictly for finding stock or share prices of companies."
)


def stockLineChart(ticker: str):
    """
    Download stock price to draw line chart.
    """
    matches = re.findall(pattern, ticker)
    ticker = ''.join(matches)
    tick = yf.Ticker(ticker)
    price = tick.history(period='2y')
    # data = yf.download(ticker, period='5y')
    data_closePrice = price['Close']
    line_chart = alt.Chart(
        data_closePrice.reset_index()).mark_line(strokeWidth=0.8).encode(
            alt.X('Date:T'),
            alt.Y('Close:Q',
                  title='Closing Price').scale(zero=False)).interactive()
    return st.altair_chart(line_chart)


stockLineChart_tool = StructuredTool.from_function(
    func=stockLineChart,
    name='linechart',
    description="use this function to find stock price and draw line chart or trendline"
)


def financialIndicators(ticker: str):
    """
    Download company's financial information like EPS, EBITA, Book Value.
    """
    matches = re.findall(pattern, ticker)
    ticker = ''.join(matches)
    symbol = yf.Ticker(ticker)
    data = {}
    for key, value in symbol.info.items():
        data[key] = value
    return data


financialIndicator_tool = StructuredTool.from_function(
    func=financialIndicators,
    name='financial_metrics',
    description="use this function to download company's financial indicators like price earnings, earnings per share etc"
)

# remove linechart_tool
tools = [search_tool, news_tool,
         stockPrice_tool,
         financialIndicator_tool,
         stockLineChart_tool,
         weather24hr_tool,
         weather4days_tool]

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}

PROMPT = PromptTemplate(input_variables=[
                        "chat_history", "input", "agent_scratchpad"], template=template)

endpoint_error_message = "I'm sorry, HuggingFace endpoint has too many requests now. Please try again later."
model_error_message = "I'm sorry, the AI model is overloaded at the endpoint. Please try again later."


footer_html = """<div style='text-align: center;'>
<p style="font-size:70%;">Developed with 💗 by Andy Oh</p>
<p style="font-size:70%;">Ngee Ann Polytechnic</p>
</div>"""

# chat_msgs.add_user_message(
#    """
#    :blue[Woof woof! I can answer general questions like these:]
#    - Latest headlines in Singapore\n
#    - Closing price of Nvidia's for the past 5 days in a table\n
#    - Trendline of Nvidia's stock price\n
#    - Earnings per share of Microsoft\n
#    - Weather forecast today\n
#    - Will it rain in the west of Singapore tomorrow?\n
#    - Prime Minister of United Kingdom
#    """)
