import streamlit as st
import streamlit_antd_components as sac
from session_chat_utils import *


def breakingnews(news):
    """
    creates news ticker with CNAheadlines tools from utils
    """

    sac.alert(label='Breaking news...',
              description=news,
              size='md',
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

        ], index=None, align='center', size='xs', on_change=chat_msg_change, key='btn', divider=False
    )


def schedule_buttons():
    return sac.buttons([
        sac.ButtonsItem(icon=sac.BsIcon(name='pen', size=20),
                        label="Pen schedule"),
        sac.ButtonsItem(icon=sac.BsIcon(
            name='table', size=20), label="View schedule", ),

    ], align='right', size="sm", index=2, gap='md', radius='md')


# set up options with tryout questions for selectbox at sidebar

schedule_options = ("Work reminder",
                    "Friends reminder",
                    "Birthday reminder",
                    "Holiday reminder",
                    "Medical reminder",
                    "Event reminder"
                    )

news_options = ("Local news from mustsharenews.com",
                "Trending stories USA",
                "Business news"
                )

financial_options = ("Nvidia's last closing price",
                     "Draw a line chart of Nvidia stock price",
                     "Find the key financial metrics of Nvidia")

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
