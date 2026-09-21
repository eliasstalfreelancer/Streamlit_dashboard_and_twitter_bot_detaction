import streamlit as st
import pandas as pd
import numpy as np

def show_starting_screen():

    st.title("Twitter Bot Detection")
    st.write("Choose what you want to explore")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.subheader("📊 Explore Data")
            st.write("Explore the Twitter dataset.")

            if st.button(
                "Explore",
                key="explore_data_button",
                width= "stretch"
            ):
                st.session_state.page = "eda"

    with col2:
        with st.container(border=True):
            st.subheader("🤖 Test an account")
            st.write("Analyze an account.")

            if st.button(
                "Analyze",
                key="bot_detector_button",
                width= "stretch"
            ):
                st.session_state.page = "model"
                st.rerun()

    with col3:
        with st.container(border=True):
            st.subheader("📈 Model Results")
            st.write("Explore model performance.")

            if st.button(
                "Results",
                key="model_results_button",
                width= "stretch"
            ):
                st.session_state.page = "prediction"
                st.rerun()