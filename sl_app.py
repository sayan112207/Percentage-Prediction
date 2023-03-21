#Importing Libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd 

model = pickle.load(open('model.pkl','rb'))

def predict(val):
    # Use the model to make predictions on new data
    b = np.array(val, dtype=float) 
    new_data = np.array([[b]])
    new_pred = model.predict(new_data)
    pred='{0:.{1}f}'.format(new_pred[0][0], 2)
    return float(pred)

def main():
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Study-Hours Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    val = st.text_input("Study Hours","Type Here")
    
    if st.button("Predict"):
        output=predict(val)
        st.success('Probable Percentage of Student is {}'.format(output))
    
    
            
if __name__=='__main__':
    main()