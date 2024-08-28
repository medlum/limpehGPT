import streamlit as st
import datetime
import streamlit as st
from pathlib import Path
import csv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit_antd_components as sac


def breakingnews(news):
    """
    creates news ticker with CNAheadlines tools from utils
    """

    sac.alert(label='Breaking news...',
              description=news,
              size='xs',
              radius='0px',
              # icon=True,
              variant='filled',
              closable=True,
              banner=[False, True])


def mode_button():
    """
    creates mode button 
    """
    return sac.segmented(
        items=[
            sac.SegmentedItem(label='creative'),
            sac.SegmentedItem(label='news'),
            sac.SegmentedItem(label='weather'),
            sac.SegmentedItem(label='finance'),
            sac.SegmentedItem(label='schedule'),

        ], align='center', size="xs",
    )


# set up options with tryout questions for selectbox at sidebar

schedule_options = ("Do I have appointments tomorrow?",
                    "Show me all my schedule",
                    "Birthday reminders"
                    )

news_options = ("Local news from mustsharenews.com",
                "Trending stories USA",
                "Latest news headlines from CNA"
                "")

financial_options = ("Nvidia's last closing price",
                     "Draw a line chart of Nvidia stock price",
                     "Find the key financial metrics of Nvidia",
                     "Latest business news",)

weather_options = ("How's the weather today?",
                   "Weather forecast for the next few days"
                   )


creative_options = ("Tell me a joke about dogs",
                    "Write a rhyme about Cosmo, the cavapoo and his owner, Andy!",
                    "Start a fun quiz!",
                    "Why is the sky blue?",
                    "Is it morally right to kill mosquitoes?",
                    "Do you think that I think you have consciousness?",
                    "Can curiosity kill a cat?"
                    )
