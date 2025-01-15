# Libraries
import streamlit as st

import eda
import classifier

# navigation
navigation = st.sidebar.selectbox('Navigation :',('Classifier','EDA'))

# pilih page
if navigation == 'Classifier':
    classifier.run()
else :
    eda.run()