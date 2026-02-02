import pandas as pd
import streamlit as st
import plotly.express as px

def plot_pie_chart(confidence: float):
    df = pd.DataFrame({
    "Class": ["NORMAL", "PNEUMONIA"],
    "Probability": [1-confidence, confidence]
    })
    fig = px.pie(df, names='Class', values='Probability', 
                color='Class', color_discrete_map={'NORMAL':'#00E676','PNEUMONIA':'#FF1744'})
    st.plotly_chart(fig)
