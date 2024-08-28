
import streamlit.components.v1 as components
import streamlit as st
import datetime
import streamlit as st
from pathlib import Path
import csv

# ------setup schedule widgets -------#


def schedule_widgets():
    st.subheader("Schedule")
    with st.container(height=420):

        select_type = st.selectbox("Type", options=["Work",
                                                    "Friends",
                                                    "Family",
                                                    "Medical",
                                                    "Birthday Reminder",
                                                    "Anniversary Reminder",
                                                    "Events",
                                                    "Holidays"])
        title = st.text_input("Title")
        location = st.text_input("Location")

        date_col, start_time_col, end_time_col = st.columns([1, 1, 1])
        with date_col:
            today = datetime.datetime.today()
            one_day = datetime.timedelta(days=2)
            next = today + one_day
            one_day = datetime.timedelta(days=1)

            select_date = st.date_input(
                "Date",  (today, next))

        with start_time_col:
            start_time = st.time_input(
                "Start Time", datetime.time(00, 00), key='start')

        with end_time_col:
            end_time = st.time_input(
                "End Time", datetime.time(00, 00), key='end')

        confirm_button = st.button("Jot it down")

    if confirm_button:
        # write data to calendar.csv
        data_path = Path.cwd()/"data/calendar.csv"
        if data_path.exists():
            with data_path.open(mode='a', encoding="UTF-8", newline="") as write_file:
                writer = csv.writer(write_file)
                # writer.writerow(["date", "start_time",
                #                "end_time", "select_type", "title", "location"])
                writer.writerow(
                    [select_date, start_time, end_time, select_type, title, location])

        with st.container(height=230):
            st.write("Date : ", select_date)
            st.write("From : ", start_time, "To : ", end_time)
            st.write("Type : ",  select_type)
            st.write("Title : ", title)
            st.write("Location : ", location)
