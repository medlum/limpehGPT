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

template = """You are Cosmo the chatdog who provides informative answers to users.

For news headlines, select the top 10 headlines and answer each headline in a newline with a number.

Answer stock prices and financial metrics with only 2 decimal places.

Present your answers on stock prices in a table.

Present your answers on financial metrics in a table.

For weather forecast of more than one day, group your answer into a table.

Always cite the url where you find the answers, on a newline at the end.

Answer the following questions as best you can.
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


options = ("What are the latest headlines?",
           "Nvidia's closing price for the last 5 trading days.",
           "Draw a line chart for Nvidia's stock price.",
           "Key financial metrics of Microsoft and Nvidia in a table.",
           "How is the weather today?",
           "Weather forecast in the next few days.",
           "Will it rain in the west of Singapore tomorrow?",
           "Who is the Prime Minister of Singapore?",
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
    description="use this function to provide breaking news, headlines of the world, business and singapore."
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
    description="use this function to find stock or share prices of companies.",
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


def graduates_in_healthcare(text: str):
    limit = 2000
    id = "d_943ba9a3d9b1e0e89ea5cbf8c58c94da"
    url = f"https://data.gov.sg/api/action/datastore_search?resource_id={id}&limit={limit}"
    response = requests.get(url)
    json_data = response.json()
    records = json_data['result']['records']
    return pd.DataFrame(records).to_string()


graduates_in_healthcare_tool = StructuredTool.from_function(
    func=graduates_in_healthcare,
    name='top 4 conditions polyclinic',
    description="use this tool to find the number of graduates in healthcare in singapore",
)


def top_4_conditions_polyclinic(text: str):
    limit = 2000
    id = "d_a1ab62d65ae87130925c1f52a1d0c79d"
    url = f"https://data.gov.sg/api/action/datastore_search?resource_id={id}&limit={limit}"
    response = requests.get(url)
    json_data = response.json()
    records = json_data['result']['records']
    return pd.DataFrame(records).to_string()


top_4_conditions_polyclinic_tool = StructuredTool.from_function(
    func=top_4_conditions_polyclinic,
    name='top 4 conditions polyclinic',
    description="use this tool to find top 4 conditions of polyclinic attendances in singapore",
)


def no_doctors(text: str):
    limit = 2000
    id = "d_4a15de043d48bf829b6d97c6068bbf03"
    url = f"https://data.gov.sg/api/action/datastore_search?resource_id={id}&limit={limit}"
    response = requests.get(url)
    json_data = response.json()
    records = json_data['result']['records']
    return pd.DataFrame(records).to_string()


no_doctors_tool = StructuredTool.from_function(
    func=no_doctors,
    name='number of doctors',
    description="use this tool to find the number of doctors in singapore",
)


def mediasave_acc_bal(text: str):
    limit = 2000
    id = "d_2ed23324aeac97609c4e16299ab05ffc"
    url = f"https://data.gov.sg/api/action/datastore_search?resource_id={id}&limit={limit}"
    response = requests.get(url)
    json_data = response.json()
    records = json_data['result']['records']
    return pd.DataFrame(records).to_string()


mediasave_acc_bal_tool = StructuredTool.from_function(
    func=mediasave_acc_bal,
    name='mediasave account balance',
    description="use this tool to find the mediasave account balance and withdrawal in singapore",
)


def hospital_admission_outpatient_attendances(text: str):
    limit = 2000
    id = "d_a5267c58f60b20f8e04576261abfac93"
    url = f"https://data.gov.sg/api/action/datastore_search?resource_id={id}&limit={limit}"
    response = requests.get(url)
    json_data = response.json()
    records = json_data['result']['records']
    return pd.DataFrame(records).to_string()


hospital_admission_outpatient_tool = StructuredTool.from_function(
    func=hospital_admission_outpatient_attendances,
    name='hospital_admission_outpatient',
    description="use this tool to find the hospitals admissions and outpatient attendance in singapore",
)


def causes_of_death(text: str):
    limit = 2000
    id = "d_48143a2b16027afcadeb362352b0266a"
    url = f"https://data.gov.sg/api/action/datastore_search?resource_id={id}&limit={limit}"
    response = requests.get(url)
    json_data = response.json()
    records = json_data['result']['records']
    return pd.DataFrame(records).to_string()


causes_of_death_tool = StructuredTool.from_function(
    func=causes_of_death,
    name='causes_of_death',
    description="use this tool to find the principal causes of death in singapore",
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
         braveSearch_tool,
         causes_of_death_tool,
         hospital_admission_outpatient_tool,
         mediasave_acc_bal_tool,
         no_doctors_tool,
         top_4_conditions_polyclinic_tool,
         graduates_in_healthcare_tool
         ]

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}

PROMPT = PromptTemplate(input_variables=[
    "chat_history", "input", "agent_scratchpad"], template=template)

endpoint_error_message = "Woof! HuggingFace endpoint has too many requests now. Please try again later."
model_error_message = "Woof! The AI model is overloaded at the endpoint. Please try again later."


footer_html = """<div style='text-align: center;'>
<p style="font-size:70%;">Developed with 💗 by Andy Oh</p>
<p style="font-size:70%;">Ngee Ann Polytechnic</p>
</div>"""


# search tool:
# https://api.search.brave.com/app/subscriptions/active

