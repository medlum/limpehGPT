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

    # setup date time
    today = datetime.datetime.today()
    one_day = datetime.timedelta(days=2)
    next = today + one_day
    one_day = datetime.timedelta(days=1)

    st.subheader(":blue[Schedule]")

    with st.form("schedule_form", clear_on_submit=True, border=True):

        appt_type = st.selectbox(":blue[Type]",
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

        appt_title = st.text_input(":blue[Title]")

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
            end_date = str(select_date[1])
        else:
            end_date = str(select_date[0])

        submitted = st.form_submit_button("Submit")

        if submitted:
            # remove YYYY from date entry as annual reminder
            if appt_type.lower() == "birthday reminder":
                start_date = str(start_date)[5:]
                end_date = str(end_date)[5:]

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

            # update github calendar.csv
            repo.update_file(github_file_path,
                             commit_message,
                             contents,
                             content.sha)

            st.toast('Pinned in caldendar.', icon="✅")

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
        st.write(pd.read_csv(StringIO(response.text)))


# https://stackoverflow.com/questions/76238677/how-to-programmatically-read-and-update-a-csv-file-stored-in-a-github-repo
