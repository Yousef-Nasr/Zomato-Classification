import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import sklearn
print(sklearn.__version__)
# Load the model
model = joblib.load('final_model.pkl')

df = pd.read_csv('zomato_cleaned.csv')
cuisines_df = pd.read_csv('cuisines.csv')

# function to convert multiselect to multilabel
def multilabel_binarizer(inputs, labeles):
    bi_list = []
    for i in labeles:
        if i in inputs:
            bi_list.append(1)
        else:
            bi_list.append(0)
    return pd.DataFrame([bi_list], columns=labeles)

# prediction function
def predict(online_order, book_table, votes, rest_type, approx_cost, listed_in_type, listed_in_city, cuisines):
    # convert multiselect to multilabel
    cuisines_multilabel = multilabel_binarizer(cuisines, cuisines_df.columns.tolist())

    # convert radio button to binary
    online_order = 1 if online_order == 'yes' else 0
    book_table = 1 if book_table == 'yes' else 0

    # create a dataframe
    df = pd.DataFrame({'online_order': online_order,
                       'book_table': book_table,
                       'votes': votes,
                       'rest_type': rest_type,
                       'approx_cost(for two people)': approx_cost,
                       'listed_in(type)': listed_in_type,
                       'listed_in(city)': listed_in_city})
    # concat the dataframes
    df = pd.concat([df, cuisines_multilabel], axis=1)
    # predict the result
    result = model.predict(df)
    return result


# streamlit app body
def main():
    html_temp="""
                <div style="background: linear-gradient(to right, #ef3b36, #ffffff); margin-bottom:20px;">
                <h2 style="color:black;text-align:center; font-family:unset;">Zomato Prediction </h2>
                </div>
              """
    html_temp1="""
                <div style="background: linear-gradient(to left, #ef3b36, #ffffff); margin-top:100%;" >
                <h4 style="color:white;text-align:center; font-family:unset;">
                <a style="color:black; text-decoration:none; font-size:30px" href="https://github.com/Yousef-Nasr/Zomato-Classification">🚀 Git repository</a>
                </h4>
                </div>
              """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.sidebar.markdown(html_temp1,unsafe_allow_html=True)
    online_order = st.radio('online order',['yes', 'no'])
    book_table = st.radio('book table',['yes', 'no'])
    votes = st.number_input('votes', min_value=0, max_value=5000, value=0, step=1)
    rest_type = st.multiselect('rest type', df['rest_type'].unique().tolist(), max_selections = 1)
    approx_cost = st.number_input('approx cost(for two people)',min_value=0, max_value=10000, value=0, step=1)
    listed_in_type =  st.multiselect('listed in(type)', df['listed_in(type)'].unique().tolist(), max_selections= 1)
    listed_in_city = st.multiselect('listed_in(city)', df['listed_in(city)'].unique().tolist(), max_selections= 1)
    cuisines = st.multiselect('pick The cuisines', cuisines_df.columns.tolist())

    if st.button('predict'):
        result = predict(online_order, book_table, votes, rest_type, approx_cost, listed_in_type, listed_in_city, cuisines)
        result = 'Success' if result[0] == 1 else 'Disappointment'
        st.success(result)
 
if __name__ =='__main__':
    main()