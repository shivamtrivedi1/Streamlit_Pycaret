import pandas as pd
import time
import streamlit as st
import plotly.express as px


from pycaret.classification import *
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment_28042021')
"""
def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions
"""

def run():

    from PIL import Image
    
    image_hospital = Image.open('titanic.png')
"""
    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    #st.sidebar.info('This app is created to predict patient hospital charges')
    #st.sidebar.success('https://www.pycaret.org')
    
    st.sidebar.image(image_hospital)

    st.title("Insurance Charges Prediction App")

    if add_selectbox == 'Online':

        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        pclass= st.number_input('P Class', 1,3)
        sib_sp=  st.multiselect('Number of Siblings And Spouse',[0,1,2,3,4,5,8])
        parch= st.multiselect('Parch',[0,1,2,3,4,5,6])
        fare=  st.slider('Fare', 0,600)
        embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])
"""
"""     
        output=""

        input_dict = {'age' : age, 'sex' : sex, 'pclass':pclass,'sib_sp':sib_sp,'parch':parch,'fare':fare,'embarked':embarked}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
"""

if __name__ == '__main__':
    run()

"""def load_dataset(data_link):
    dataset = pd.read_csv(data_link)
    return dataset
titanic_link = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
titanic_data = load_dataset(titanic_link)
st.title("Titanic Survived Prediction")

sidebar = st.sidebar
sidebar.title("This is the sidebar.")
sidebar.write("You can add any elements to the sidebar.")


st.header("Dataset Overview")
st.dataframe(titanic_data)

st.header("Data Description")

selected_class = st.radio("Select Class", titanic_data['class'].unique())
st.write("Selected Class:", selected_class)

st.markdown("___")

selected_sex = st.selectbox("Select Sex", titanic_data['sex'].unique())
st.write(f"Selected Option: {selected_sex!r}")

st.markdown("___")

selected_decks = st.multiselect("Select Decks", titanic_data['deck'].unique())
st.write("Selected Decks:", selected_decks)

st.markdown("___")

age_columns = st.beta_columns(2)
age_min = age_columns[0].number_input("Minimum Age", value=titanic_data['age'].min())
age_max = age_columns[1].number_input("Maximum Age", value=titanic_data['age'].max())
if age_max < age_min:
    st.error("The maximum age can't be smaller than the minimum age!")
else:
    st.success("Congratulations! Correct Parameters!")
    subset_age = titanic_data[(titanic_data['age'] <= age_max) & (age_min <= titanic_data['age'])]
    st.write(f"Number of Records With Age Between {age_min} and {age_max}: {subset_age.shape[0]}")

optionals = st.beta_expander("Optional Configurations", True)
fare_min = optionals.slider(
    "Minimum Fare",
    min_value=float(titanic_data['fare'].min()),
    max_value=float(titanic_data['fare'].max())
)
fare_max = optionals.slider(
    "Maximum Fare",
    min_value=float(titanic_data['fare'].min()),
    max_value=float(titanic_data['fare'].max())
)
subset_fare = titanic_data[(titanic_data['fare'] <= fare_max) & (fare_min <= titanic_data['fare'])]
st.write(f"Number of Records With Fare Between {fare_min} and {fare_max}: {subset_fare.shape[0]}")
"""

