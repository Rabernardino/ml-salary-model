

import streamlit as st
import pandas as pd

st.markdown('Data Salary Model')

raw_data = pd.read_excel('../../data/raw_data.xlsx')

select_box_pos = st.selectbox('Education', options=raw_data['EDUCATION'].unique())

