import pandas as pd
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
import datetime
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import BraveSearch
from io import BytesIO
from github import Github
import requests
import pandas as pd
from io import StringIO


# ---- online search with brave search ---#
braveSearch = BraveSearch.from_api_key(
    api_key=st.secrets['brave_api'], search_kwargs={"count": 3})

braveSearch_tool = Tool(
    func=braveSearch,
    name="brave_search",
    description="use this function to answer questions about most current events."
)

# ---- wikipedia search ---#

wikipedia = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name='wikipedia',
    func=wikipedia.run,
    description="Useful when you need to look up for comprehensive information on all branches of knowledge"
)

# ---- weather forecast ---#

# nea api - 4 days forecast


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

# nea api - 24 hours forecast


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


# ---- news headlines ---#

# webscrape on CNA headlines
def CNAheadlines(genre: str):

    url = "https://www.channelnewsasia.com"
    response = requests.get(url)
    if response.status_code == 200:
        news = []
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find('body').find_all('h6')  # headlines at h6
        for x in headlines:
            news.append(x.text.strip())
        return '. '.join(news)
    else:
        return "No response from news provider."

# webscrape on CNA headlines


news_tool = StructuredTool.from_function(
    func=CNAheadlines,
    name="CNA_headlines",
    description="use this function to provide news headlines."
)


def businessnews(story: str):

    url_placeholder = "https://www.channelnewsasia.com"
    url = "https://www.channelnewsasia.com/business"
    news = []

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # h3 tag in CNA --> single feature news
    h3_result = soup.find('h3', class_="h3 list-object__heading")
    h3_text = h3_result.find('a').get_text().strip()
    h3_href = h3_result.find('a').get('href')
    h3_url = f"{url_placeholder}{h3_href}"
    # h5 tags in CNA --> multiple news
    h5_result = soup.find_all('h5', class_="h5 list-object__heading")
    for element in h5_result:
        news.append((element.find('a').get_text().strip(),
                    f"{url_placeholder}{element.find('a').get('href')}"))

    news.insert(0, (h3_text, h3_url))

    return news


businessnews_tool = StructuredTool.from_function(
    func=businessnews,
    name="business_headlines",
    description="use this function to provide business news headlines."
)


def trending_today(story: str):

    url = "https://www.today.com/trending"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        container = soup.find('div', class_="styles_itemsContainer__saJYW")

        # search for images with img tag and loading:'lazy'
        # this is a 200px x 200px image
        img = container.find_all("img", loading="lazy")
        img_urls = []
        for index, element in enumerate(img):
            if index % 2 == 0:
                continue
            # url in 'src' tag
            img_urls.append(element['src'])

        for i in container:
            headlines = soup.find_all('h2', class_='wide-tease-item__headline')
            descriptions = soup.find_all(
                'div', class_='wide-tease-item__description')
            links = soup.find_all(
                'div', class_="wide-tease-item__image-wrapper flex-none relative dt dn-m")

        trend_headlines = [headline.text.strip() for headline in headlines]
        trend_descriptions = [description.text.strip()
                              for description in descriptions]
        trend_urls = [link.find('a').get('href') for link in links]

        trending_story = {}

        for i in range(len(trend_headlines)):
            trending_story[f"headline of story {i+1}"] = f'{trend_headlines[i]}.'
            trending_story[f"description of story {i+1}"] = f'{trend_descriptions[i]}.'
            trending_story[f"url of story {i+1}"] = f'{trend_urls[i]}'
            trending_story[f"image of story {i+1}"] = f'<img src={img_urls[i]} width="100" height="100">\n'

            # if trend_headlines[i] not in trending_story:
            #    trending_story[trend_headlines[i]
            #                   ] = f"{trend_descriptions[i]} {trend_urls[i]}"

        return trending_story


trending_stories_tool = StructuredTool.from_function(
    func=trending_today,
    name="trending_today_usa",
    description="use this function to provide trending stories."
)


def mustsharenews(story: str):
    news = []
    url = "https://mustsharenews.com/"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # single feature news
        h2_feature = soup.find('h2', class_="title-cat-main")
        # latest news
        h2_result = soup.find_all('h2', class_="title-cat-white")
        # extract text and href from feature news
        feature_headline = h2_feature.get_text().strip()
        feature_href = h2_feature.find('a').get('href')
        # extract text and href from remaining news
        for h2 in h2_result:
            headlines = h2.get_text().strip()
            href = h2.find('a').get('href')
            news.append((headlines, href))

        news.insert(0, (feature_headline, feature_href))

    return news


mustsharenews_tool = StructuredTool.from_function(
    func=mustsharenews,
    name="MustShareNews",
    description="use this function to provide local news in singapore from mustsharenews.com."
)

# --------- image tools ------------#

# search image with bravesearch


def query_bravesearch_image(query: str):
    brave_api_key = "BSANRhMz7xnB_dIA1nzDwO2uaw3cpVA"
    url = "https://api.search.brave.com/res/v1/images/search"
    headers = {
        "X-Subscription-Token": brave_api_key
    }
    params = {
        "q": query,
        "count": 1,
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    return f'<img src={data["results"][0]["properties"]["url"]}>'


# def query_bravesearch_image(query: str):
#    brave_api_key = "BSANRhMz7xnB_dIA1nzDwO2uaw3cpVA"
#    url = "https://api.search.brave.com/res/v1/images/search"
#    headers = {
#        "X-Subscription-Token": brave_api_key
#    }
#    params = {
#        "q": query,
#        "count": 1,
#    }
#
#    response = requests.get(url, headers=headers, params=params)
#
#    result = response.json()
#
#    url_html = {}
#
#    for img in result['results']:
#        if img['title'] not in url_html:
#            url_html[img['title']] = f"<img src={img['thumbnail']['src']}'>"
#        # url_html.append(f"<img src={img['thumbnail']['src']} alt=f'{img['title']}'>")
#
#    return (url_html)


img_search_tool = StructuredTool.from_function(
    func=query_bravesearch_image,
    name="image_bravesearch",
    description="use this function to search for image url and put the url into html."
)

# view image with a url


def get_img(url: str):
    r = requests.get(url)
    return st.image(BytesIO(r.content), width=100)


img_view_tool = StructuredTool.from_function(
    func=get_img,
    name="image_tool",
    description="use this function to display an image from a url."
)

# ---- stock and financial data ---#


# yahoo finance api for single stock


def stockPrice(ticker: str) -> str:
    pattern = r'[A-Z]+\d+[A-Z]*\.SI|[A-Z]+\b'
    """return stock price """
    matches = re.findall(pattern, ticker)
    ticker = ''.join(matches)
    tick = yf.Ticker(ticker)
    price = tick.history()
    return price.to_string()


stockPrice_tool = StructuredTool.from_function(
    func=stockPrice,
    name='yfinance',
    description="use this function to find stock prices of a public listed company.",
)

# draw line chart of single stock price


def stockLineChart(ticker: str):
    """
    Download stock price to draw line chart.
    """
    pattern = r'[A-Z]+\d+[A-Z]*\.SI|[A-Z]+\b'
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

# yahoo finance - financial metrics


def financialIndicators(ticker: str):
    """
    Download company's financial information like EPS, EBITA, Book Value.
    """
    pattern = r'[A-Z]+\d+[A-Z]*\.SI|[A-Z]+\b'
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

# -------- today's date --------#


def time(text: str) -> str:
    weekdays_map = {i: ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday'][i] for i in range(7)}

    today = datetime.datetime.today()
    day = weekdays_map[today.weekday()]

    return today, day


time_tool = StructuredTool.from_function(
    func=time,
    name='today_date',
    description="Return today's date, use this for any questions related to knowing today's date.The input should always be an empty string, and this function will always return todays date "
)


# ------appointment -------#

def github_schedule_check(schedule: str):
    repo_owner = 'medlum'
    repo_name = 'limpehGPT'
    github_file_path = 'data/data.json'
    github_url = f'https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{github_file_path}'
    response = requests.get(github_url)
    data = json.dumps(response.text)
    # df = pd.read_csv(StringIO(response.text)).to_csv()
    return data


github_schedulecheck_tool = StructuredTool.from_function(
    func=github_schedule_check,
    name='github_schedule_check',
    description="Use this function to check on user's schedule."
)

# ------health -------#


def health_check(name: str):
    repo_owner = 'medlum'
    repo_name = 'limpehGPT'
    github_file_path = 'data/health_data.json'
    github_url = f'https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{github_file_path}'
    response = requests.get(github_url)
    return str(response.text)


health_check_tool = StructuredTool.from_function(
    func=health_check,
    name='health_check',
    description="Use this function to check health data of patients."
)

tools_for_schedule = [github_schedulecheck_tool,
                      time_tool, health_check_tool, img_search_tool]

tools_for_weather = [weather24hr_tool,
                     weather4days_tool,
                     braveSearch_tool]

tools_for_stock = [stockPrice_tool,
                   financialIndicator_tool,
                   stockLineChart_tool,
                   time_tool,
                   # braveSearch_tool,
                   ]

tools_for_news = [
    # news_tool,
    trending_stories_tool,
    # img_search_tool,
    mustsharenews_tool,
    braveSearch_tool,
    businessnews_tool,
]

endpoint_error_message = "Woof! HuggingFace endpoint has too many requests now. Please try again later."
model_error_message = "Woof! The AI model is overloaded at the endpoint. Please try again later."


footer_html = """<div style='text-align: center;'>
<p style="font-size:70%;">Developed with 💗 by Andy Oh</p>
<p style="font-size:70%;">Ngee Ann Polytechnic</p>
</div>"""
