import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

st.title('Diabetes Prediction')

pregnancies = st.sidebar.number_input("Number of Pregnancies", 0, 20, 0, 1)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 0, 1)
skinthickness = st.sidebar.slider("Skin Thickness", 0, 99, 0, 1)
bloodpressure = st.sidebar.slider('Blood Pressure', 0, 122, 0, 1)
insulin = st.sidebar.slider("Insulin", 0, 846, 0, 1)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 0.0, 0.1)
dpf = st.sidebar.slider("Diabetics Pedigree Function", 0.000, 2.420, 0.000, 0.001)
age = st.sidebar.number_input("Age", 1, 150, 1, 1)
outcome = st.sidebar.number_input("Outcome", 0, 1, 0, 1)

row = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age, outcome]


if (st.button('Find Health Status')):
    df = pd.read_csv("C:\\Users\\ARDA\\PycharmProjects\\pythonProjectLastDance\\main.py")
    df.loc[768] = row #yeni girilen değerleri dataframeye ekliyoruz

    #datayı okuyoruz ve analiz ediyoruz
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    fill_values = SimpleImputer(missing_values=0, strategy='mean')
    X_train = fill_values.fit_transform(X_train)
    X_test = fill_values.transform(X_test)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    #yeni girdiğimiz değerleri analiz ediyoruz ve diyabet olma olasılığını hesaplıyoruz
    personData = df[768:]
    personDataFeatures = np.asarray(personData.drop('Outcome',1))
    predictionProbability = rfc.predict_proba(personDataFeatures)
    prediction = rfc.predict(personDataFeatures)

    diabetes = predictionProbability[0][1]
    notDiabetes = predictionProbability[0][0]
    st.subheader('Your chance of having diabetes is %{} Go to medicine'.format(diabetes * 100))
    st.subheader('Your chance of not having diabetes is %{} You are healthy'.format(notDiabetes * 100))




