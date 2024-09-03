
import streamlit as st
import datetime
import streamlit as st
from github import Github
from io import StringIO
import requests
import pandas as pd
from news_btn_options_utils import *

repo_owner = 'medlum'
repo_name = 'limpehGPT'
github_file_path = 'data/calendar.csv'
token = st.secrets["github_personal_token"]
commit_message = 'Update CSV file'
github = Github(token)

# ------setup schedule widgets -------#


def schedule_widgets():

    st.subheader(":blue[Appointment Pad]")

    with st.container(height=360):

        appt_type = st.selectbox(":blue[Type]", options=["Work",
                                                         "Friends",
                                                         "Family",
                                                         "Errand",
                                                         "Medical",
                                                         "Birthday",
                                                         "Anniversary",
                                                         "Special Event",
                                                         "Holiday"])

        appt_title = st.text_input(":blue[Title]")

        location = st.text_input(":blue[Location]")

        #  Columns for date, and time entry
        date_col, start_time_col, end_time_col = st.columns([1, 1, 1])
        # col 1
        with date_col:
            today = datetime.datetime.today()
            one_day = datetime.timedelta(days=2)
            next = today + one_day
            one_day = datetime.timedelta(days=1)

            select_date = st.date_input(
                ":blue[Date]",  (today.date(), next.date()))

            # extract start and end date
            start_date = select_date[0]
            if len(select_date) == 2:
                end_date = str(select_date[1])
            else:
                end_date = str(select_date[0])

        # col 2 start time
        with start_time_col:
            start_time = st.time_input(
                ":blue[Start time]", datetime.time(00, 00), key='start')

        # col 3 end time
        with end_time_col:
            end_time = st.time_input(
                ":blue[End time]", datetime.time(00, 00), key='end')

    # confirm_button = st.button(":blue[Pen it]")
    btn_schedule = schedule_buttons()

    # if confirm_button:
    if btn_schedule == "Pen schedule":
        # df.to_csv('temp_data.csv', index=False)

        repo = github.get_user(repo_owner).get_repo(repo_name)
        github_url = f'https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{github_file_path}'
        response = requests.get(github_url)
        df = pd.read_csv(StringIO(response.text))

        # insert to last row of df from github
        df.loc[len(df.index)] = [start_date,
                                 end_date,
                                 start_time,
                                 end_time,
                                 appt_type,
                                 appt_title,
                                 location]
        # convert to csv
        contents = df.to_csv(index=False)
        # fetch calendar.csv  from github
        content = repo.get_contents(github_file_path)

        # with open('temp_data.csv', 'rb') as f:
        #    contents = f.read()

        # update github calendar.csv
        repo.update_file(github_file_path,
                         commit_message,
                         contents,
                         content.sha)

        st.toast('Pinned in caldendar.', icon="✅")

    # view_button = st.button(":blue[View it]")
    # if view_button:
    if btn_schedule == "View schedule":

        repo = github.get_user(repo_owner).get_repo(repo_name)
        github_url = f'https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{github_file_path}'
        response = requests.get(github_url)
        st.write(pd.read_csv(StringIO(response.text)))

        # with st.container(height=230):
        #    st.write("Date : ", select_date)
        #    st.write("From : ", start_time, "To : ", end_time)
        #    st.write("Type : ",  appt_type)
        #    st.write("Title : ", appt_type)
        #    st.write("Location : ", location)

        # https://stackoverflow.com/questions/76238677/how-to-programmatically-read-and-update-a-csv-file-stored-in-a-github-repo
        # write data to calendar.csv

        #        data_path = Path.cwd()/"data/calendar.csv"
        #        if data_path.exists():
        #            with data_path.open(mode='a', encoding="UTF-8", newline="") as write_file:
        #                writer = csv.writer(write_file)
        #                # writer.writerow(["date", "start_time",
        #                #                "end_time", "select_type", "title", "location"])
        #                writer.writerow(
        #                    [select_date, start_time, end_time, select_type, title, location]
