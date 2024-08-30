# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
# https://stackoverflow.com/questions/76238677/how-to-programmatically-read-and-update-a-csv-file-stored-in-a-github-repo


from github import Github
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
import streamlit as st
import datetime

repo_owner = 'medlum'
repo_name = 'limpehGPT'
file_path = 'data/calendar.csv'
token = st.secrets["github_personal_token"]
commit_message = 'Update CSV file'

github = Github(token)
repo = github.get_user(repo_owner).get_repo(repo_name)

url = f'https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{file_path}'
response = requests.get(url)
data = StringIO(response.text)
df = pd.read_csv(StringIO(response.text))

df.loc[len(df.index)] = [datetime.datetime.today(), '00:00:00',
                         '00:00:00',  'Birthday Reminder', 'Wife', 'NaN']

print(df)
content = repo.get_contents(file_path)
df.to_csv('test.csv', index=False)

with open('test.csv', 'rb') as f:
    contents = f.read()

repo.update_file(file_path, commit_message, contents, content.sha)
