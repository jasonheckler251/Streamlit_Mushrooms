import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

@st.cache(persist=True)
def load_mushroom_data():
    mushroom_data = pd.read_csv('data/mushrooms.csv')
    label = LabelEncoder()
    for col in mushroom_data.columns:
        mushroom_data[col] = label.fit_transform(mushroom_data[col])
    return mushroom_data

@st.cache(persist=True)
def split_df(df):
    y = df.type
    x = df.drop(columns=['type'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state=0)
    return x_train, x_test, y_train, y_test

def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()
        
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot.plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous?")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?")

    mushroom_df = load_mushroom_data()
    x_train, x_test, y_train, y_test = split_df(mushroom_df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", \
                                                     "Logistic Regression", \
                                                     "Random Forest"))


    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("Radial Basis Function", "Linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma Kernel Coefficient", ('scale', 'auto'), key='gamma')


    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(mushroom_df)

if __name__ == '__main__':
    main()