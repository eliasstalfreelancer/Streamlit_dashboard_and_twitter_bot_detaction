import streamlit as st



from screens.starting_screen import show_starting_screen
from screens.eda_screen import show_eda_screen
from screens.model_screen import show_model_screen
from screens.model_results_screen import show_model_result_screen
#from screens.model_screen import show_model_screen


st.set_page_config(
    page_title="Twitter Bot Detection",
    page_icon="🤖",
    layout="wide"
)


if "page" not in st.session_state:

    st.session_state.page = "start"


if st.session_state.page == "start":
    
    show_starting_screen()

elif st.session_state.page == "eda":
    
    show_eda_screen()

elif st.session_state.page == "prediction":
   show_model_result_screen()

elif st.session_state.page == "model":
    show_model_screen()