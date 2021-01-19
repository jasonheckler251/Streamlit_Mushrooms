import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

@st.cache(persist=True)
def load_mushroom_data():
    mushroom_data = pd.read_csv('data/mushrooms.csv')
    label = LabelEncoder()
    for col in mushroom_data.columns:
        mushroom_data[col] = label.fit_transform(mushroom_data[col])
    return mushroom_data
    

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous?")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?")

    mushroom_df = load_mushroom_data()

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(mushroom_df)

if __name__ == '__main__':
    main()