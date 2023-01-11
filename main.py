import streamlit as st
import nltk
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
import os


nltk.downloader.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

dir_path = "dairy"
dir_cont = os.listdir(dir_path)
y_neg = []
y_pos = []
x = []
for entry in dir_cont:
    with open(f"dairy/{entry}", "r") as file:
        content = file.readline()
    scores = analyzer.polarity_scores(content)  
    date = entry.split(".")
    x.append(date[0]) 
    y_neg.append(scores["neg"])
    y_pos.append(scores["pos"])
print(dir_cont)


st.title("Dairy tone")

st.subheader("Positivity")
positive_fig = px.line(x=x, y=y_pos, labels={"x": "Date", "y": "Positivity"})
st.plotly_chart(positive_fig)


st.subheader("Negativity")
negative_fig = px.line(x=x, y=y_neg, labels={"x": "Date", "y": "Negativity"})
st.plotly_chart(negative_fig)