import datetime
import streamlit as st
from github import Github
from io import StringIO
import requests
import pandas as pd
from news_btn_options_utils import *
import json

repo_owner = 'medlum'
repo_name = 'limpehGPT'
github_file_path = 'data/data.json'
token = st.secrets["github_personal_token"]
commit_message = 'Update json file'
github = Github(token)


def reset_selectbox():
    """
    callback function to clear dropdown question
    when the form submit button is clicked
    """
    st.session_state.selectbox = None

# ------setup schedule widgets -------#


def schedule_widgets():
    # to map day from datetime
    weekdays_map = {i: ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday'][i] for i in range(7)}

    # --- setup date time ---#

    # system date
    today = datetime.datetime.today()
    # create a range of days for date_input
    one_day = datetime.timedelta(days=1)
    next = today + one_day
    # one_day = datetime.timedelta(days=1)

    st.subheader(":blue[Schedule]")

    # --- setup schedule form ---#
    with st.form("schedule_form", clear_on_submit=True, border=True):

        type_of_schedule = st.selectbox(":blue[Type]",
                                        options=["Please select",
                                                 "Work",
                                                 "Friends",
                                                 "Family",
                                                 "Personal Errand",
                                                 "Medical",
                                                 "Birthday Reminder",
                                                 "Anniversary",
                                                 "Special Event",
                                                 "Holiday"])

        details = st.text_input(":blue[Title]")

        location = st.text_input(":blue[Location]")

        # create 3 columns for date and time entry
        date_col, start_time_col, end_time_col = st.columns([2, 1, 1])

        # col 1
        select_date = date_col.date_input(":blue[Date]",
                                          (today.date(),
                                           next.date()))

        # col 2 start time
        start_time = start_time_col.time_input(":blue[Start time]",
                                               datetime.time(00, 00),
                                               key='start')

        # col 3 end time
        end_time = end_time_col.time_input(":blue[End time]",
                                           datetime.time(00, 00),
                                           key='end')

        # extract start and end date
        start_date = select_date[0]

        if len(select_date) == 2:
            end_date = select_date[1]
        else:
            end_date = select_date[0]

        # map weekdays
        start_day = weekdays_map[start_date.weekday()]
        end_day = weekdays_map[end_date.weekday()]

        # remove YYYY for birthday reminder since birthday is recurring
        if type_of_schedule.lower() == "birthday reminder":
            start_date = str(start_date)[5:]
            end_date = str(end_date)[5:]

        new_data = {
            "date": {
                "start": str(start_date),
                "end": str(end_date)
            },
            "day": {
                "start": start_day,
                "end": end_day
            },
            "time": {
                "start": str(start_time),
                "end": str(end_time)
            },
            "type_of_schedule": type_of_schedule,
            "details": details,
            "location": location

        }

        # callback function to reset dropdown questions
        submitted = st.form_submit_button("Submit", on_click=reset_selectbox)

        if submitted:
            # call on github url with request
            repo = github.get_user(repo_owner).get_repo(repo_name)
            github_url = f'https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{github_file_path}'
            response = requests.get(github_url)
            # load reponse as json
            data = json.loads(response.text)
            # append new data to response data
            # json is stored in the format of {"records": [] }
            data["records"].append(new_data)
            # store as json
            # dumps method for double quotation as json format required ""
            data_json = json.dumps(data, indent=4)
            # first get contents from github
            content = repo.get_contents(github_file_path)
            # then update github with new data
            repo.update_file(github_file_path,
                             commit_message,
                             str(data_json),  # convert to str
                             content.sha)

            st.toast('Pinned to calendar.', icon="✅")

    # button to view data
    if sac.buttons([sac.ButtonsItem(icon=sac.BsIcon(name='table', size=15),
                                    label="View schedule", )],
                   align='left',
                   size="sm",
                   index=2,
                   gap='md',
                   radius='md'):

        repo = github.get_user(repo_owner).get_repo(repo_name)
        github_url = f'https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{github_file_path}'
        response = requests.get(github_url)
        st.write(json.dumps(response.text, indent=4))


# https://stackoverflow.com/questions/76238677/how-to-programmatically-read-and-update-a-csv-file-stored-in-a-github-repo
