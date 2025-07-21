
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Carregando os dados
df = pd.read_csv("/Users/Work/TechChallenge4/Obesity_tratado-2.csv")

# Padronizando colunas
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^\w\s]", "", regex=True)
)
df.drop_duplicates(inplace=True)

# Definindo features
cat_features = ['gender', 'family_history', 'favc', 'caec', 'smoke', 'scc', 'calc', 'mtrans']
num_features = ['age', 'height', 'weight', 'fcvc', 'ncp', 'ch2o', 'faf', 'tue']
target = 'obesity'

# Criando transformador customizado de IMC
class IMCCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['imc'] = X_copy['weight'] / (X_copy['height'] ** 2)
        return X_copy.drop(columns=['height', 'weight'])

# Pipelines de transformação
num_pipeline = Pipeline(steps=[
    ("imc_calc", IMCCalculator()),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Modelo
model = XGBClassifier(
    colsample_bytree=0.8,
    learning_rate=0.2,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

# Pipeline final
full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Preparando dados
X = df.drop(columns=[target])
y = df[target]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Treinando o modelo
full_pipeline.fit(X_train, y_train)

# Salvando para deploy
joblib.dump(full_pipeline, "full_pipeline.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Modelo e pipeline salvos com sucesso.")
