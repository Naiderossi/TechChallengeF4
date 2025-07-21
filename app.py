import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

# Define novamente a classe IMCCalculator usada no pipeline
class IMCCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['imc'] = X_copy['weight'] / (X_copy['height']**2)
        return X_copy.drop(columns=['weight', 'height'])

try:
    full_pipeline = joblib.load('full_pipeline.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model or encoder file not found. Make sure 'full_pipeline.pkl' and 'label_encoder.pkl' are in the same directory.")
    st.stop() # Stop the app if files are not found

st.title("Aplicativo para predição de nível de obesidade")
st.write("Responda as questões abaixo para prever o nível de obesidade do paciente.")

filepath = "/Users/Work/TechChallengeF4/Obesity.csv"
df = pd.read_csv(filepath)


input_dict = {'Feminino' : 'Female', 
              'Masculino' : 'Male',
              'Sim': 'yes', 
              'Não': 'no',
              'As vezes' : 'Sometimes', 
              'Sempre': 'Always', 
              'Frequentemente': 'Frequently',
              'Transporte publico': 'Public_Transportation', 
              'Automóvel' : 'Automobile', 
              'Motocicleta' : 'Motorbike', 
              'Bicicleta' : 'Bike', 
              'Caminhando' : 'Walking'
              }

st.header('Informações pessoais')
input_gender = st.selectbox("Informe seu sexo", ["", "Feminino", "Masculino"])
input_gender = input_dict.get(input_gender)


input_age = st.number_input("Informe sua idade", min_value=1, max_value=120, step=1, format="%d")


input_hight = st.number_input("Informe sua altura (em metros)", min_value=1.0, max_value=2.5, step=0.01, format="%.2f")


input_weight = st.number_input("Informe seu peso (em kg)", min_value=1.0, max_value=400.0, step=0.1, format="%.1f")

input_family_history = st.radio("Algum membro da família sofreu ou sofre de excesso de peso?", ["Sim", "Não"])
input_family_history = input_dict.get(input_family_history)

st.header('Alimentação')

input_FAVC = st.radio("Você come alimentos altamente calóricos com frequência?", ["Sim", "Não"])
input_FAVC = input_dict.get(input_FAVC)

##st.write('### Você costuma comer vegetais nas suas refeições?')
##input_FCVC =*/

input_NCP =  st.number_input("Quantas refeições principais você faz diariamente?'", min_value=1, max_value= 30, step=1, format="%d")

input_CAEC = st.radio("Você come alguma coisa entre as refeições?", ["Não", "As vezes", "Sempre", "Frequetemente"])
input_CAEC =  input_dict.get(input_CAEC)

input_SMOKE =st.radio("Você fuma?", ["Sim", "Não"])
input_SMOKE = input_dict.get(input_SMOKE)

input_CH2O = st.number_input("Quanta água você bebe diariamente?'", min_value=0.0, max_value= 10.0, step=0.1, format="%.1f")

input_SCC = st.radio("Você monitora as calorias que ingere diariamente?", ["Sim", "Não"])
input_SCC = input_dict.get(input_SCC)

st.header('Atividade física')

input_FAF = int(st.slider('Com que frequência você pratica atividade física', 0, 7))

input_TUE =st.number_input("Quanto tempo você usa dispositivos tecnológicos como celular,videogame, televisão, computador e outros?'", min_value=0.0, max_value= 20.0, step=0.5, format="%.1f")

input_CALC = st.radio("Com que frequência você bebe álcool?", ["Não", "As vezes", "Sempre", "Frequetemente"])
input_CALC = input_dict.get(input_CALC)

input_MTRANS = st.selectbox("Qual meio de transporte você costuma usar??", ["", "Transporte publico", "Automóvel", "Motocicleta", "Bicicleta", "Caminhando"])
input_MTRANS = input_dict.get(input_MTRANS)


if st.button("Predict Obesity Level"):
    novo_paciente = pd.DataFrame(
                 {
                  'gender' : [input_gender],
                  'age': [input_age],
                  'hight': [input_hight],
                  'weight': [input_weight],
                  'family_history':[input_family_history],
                  'favc':[input_FAVC],
                  'ncp': [input_NCP],
                  'caec': [input_CAEC],
                  'smoke' : [input_SMOKE],
                  'ch2o':[input_CH2O],
                  'scc': [input_SCC],
                  'faf':[input_FAF],
                  'tue': [input_TUE],
                  'calc': [input_CALC],
                  'mtrans': [input_MTRANS]
                 }  
)
pred_encoded = full_pipeline.predict(novo_paciente)
pred_label = label_encoder.inverse_transform(pred_encoded)[0]
    
st.success(f"Categoria de obesidade prevista: **{pred_label}**")
