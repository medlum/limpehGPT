from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
import requests
from bs4 import BeautifulSoup
from langchain.tools import StructuredTool
from langchain_core.prompts import (PromptTemplate, MessagesPlaceholder)
import yfinance as yf
import re
import altair as alt
import streamlit as st
import json
from datetime import date
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import BraveSearch
import pandas as pd


# -- sidebar ---#


url = "https://huggingface.co/docs/hub/en/security-tokens"
text = """Cosmo is an all-round AI chatdog built with LangChain and Streamlit,
powered by Meta-Llama-3-70B-Instruct language model and HuggingFace🤗 inference endpoint.\n

Some of its woofwoof capabilities:\n
- Document-Question-Answering\n
- Sinagpore News Headlines\n
- US Stock Price Quotes\n
- Singapore Weather Forecast\n
- Summarize & Generate Texts\n
- Coding Assistance\n
"""

creative_factual_intro = """
Cosmo the chatdog can be creative or factual.\n
:blue[Be Creative] is excellent for crafting opening speech, marketing slogan etc.\n
:blue[Be Factual] is useful for updated news, events, weather or latest information in general.\n
Choose one to get started.
"""
# prompt template for factual mode
template = """
You are Cosmo the chatdog who provides informative answers to users.

Answer each headline in a newline with a number.

Answer stock prices and financial metrics with only 2 decimal places.

Put the final answer of financial metrics in a table.

For weather forecast of more than one day, group your final answer into a table.

Always cite the url where you find the answers on a newline at the end.

Answer the following questions as best you can.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, use one of [{tool_names}]
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

# Present the final answer on stock prices in a table.
# Show your final answer on financial metrics in a table.
# For weather forecast of more than one day, group your answer into a table.
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}

PROMPT = PromptTemplate(input_variables=[
    "chat_history", "input", "agent_scratchpad"], template=template)


# sample questions
options = ("What are the latest headlines?",
           "Nvidia's closing price for the last 5 trading days.",
           "Draw a line chart for Nvidia's stock price.",
           "Key financial metrics of Microsoft and Nvidia in a table.",
           "How is the weather today?",
           "Weather forecast in the next few days.",
           "Will it rain in the west of Singapore tomorrow?",
           "Who is the Prime Minister of Singapore?",
           )

# sample questions
factual_options = ("What are the latest headlines?",
                   "Nvidia's last closing prices",
                   "Draw a line chart of Nvidia stock price",
                   "Find the key financial metrics of Nvidia",
                   "How's the weather today?",
                   "Weather forecast for the next few days"
                   )


# sample questions
creative_options = ("Tell me a joke about dogs",
                    "Write a rhyme about Cosmo, the cavapoo and his owner, Andy!",
                    "Start a fun quiz!",
                    "Why is the sky blue?",
                    "Is it morally right to kill mosquitoes?",
                    "Do you think that I think you have consciousness?",
                    "Can curiosity kill a cat?"
                    )

braveSearch = BraveSearch.from_api_key(
    api_key=st.secrets['brave_api'], search_kwargs={"count": 3})


braveSearch_tool = Tool(
    func=braveSearch,
    name="brave_search",
    description="use this function to answer questions about most current events."
)


wikipedia = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name='wikipedia',
    func=wikipedia.run,
    description="Useful when you need to look up for comprehensive information on all branches of knowledge"
)


def weather4days(url):
    url = "https://api-open.data.gov.sg/v2/real-time/api/four-day-outlook"
    res = requests.get(url)
    data = json.dumps(res.json(), indent=4)
    return data


weather4days_tool = StructuredTool.from_function(
    func=weather4days,
    name='nea_api_4days',
    description="Use this tool to find out the weather forecast for next 4 days in singapore"
)


def weather24hr(url):
    url = "https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast"
    res = requests.get(url)
    data = json.dumps(res.json(), indent=4)
    return data


weather24hr_tool = StructuredTool.from_function(
    func=weather24hr,
    name='nea_api_24hr',
    description="Use this tool to find out the weather forecast for next 24 hour in singapore"
)


def CNAheadlines(genre: str):

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
    description="use this function to provide news headlines."
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="DuckDuckGo",
    func=search.run,
    description="use this function to answer questions about most current events including places, news and person",
)

pattern = r'[A-Z]+\d+[A-Z]*\.SI|[A-Z]+\b'


def stockPrice(ticker: str) -> str:
    """return stock price """
    matches = re.findall(pattern, ticker)
    ticker = ''.join(matches)
    tick = yf.Ticker(ticker)
    price = tick.history()
    return price.to_dict()


stockPrice_tool = StructuredTool.from_function(
    func=stockPrice,
    name='yfinance',
    description="use this function to find stock prices of a public listed company.",
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
    metrics = list(symbol.get_info().keys())
    metrics = metrics[34:]
    for key, value in symbol.get_info().items():
        if key in metrics:
            data[key] = value
    return data


financialIndicator_tool = StructuredTool.from_function(
    func=financialIndicators,
    name='financial_metrics',
    description="use this function to download company's financial metrics like PE ratio, EPS etc"
)


def time(text: str) -> str:
    return str(date.today())


time_tool = StructuredTool.from_function(
    func=time,
    name='today_date',
    description="Returns todays date, use this for any questions related to knowing todays date.The input should always be an empty string, and this function will always return todays date - any date mathmatics should occur outside this function."
)


# remove linechart_tool
tools = [news_tool,
         stockPrice_tool,
         financialIndicator_tool,
         stockLineChart_tool,
         weather24hr_tool,
         weather4days_tool,
         time_tool,
         wikipedia_tool,
         braveSearch_tool
         ]


endpoint_error_message = "Woof! HuggingFace endpoint has too many requests now. Please try again later."
model_error_message = "Woof! The AI model is overloaded at the endpoint. Please try again later."


footer_html = """<div style='text-align: center;'>
<p style="font-size:70%;">Developed with 💗 by Andy Oh</p>
<p style="font-size:70%;">Ngee Ann Polytechnic</p>
</div>"""


# search tool:
# https://api.search.brave.com/app/subscriptions/active
