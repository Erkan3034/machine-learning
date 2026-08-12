"""
AMAC:
    - Eksik veri tespiti, cıkartilması ve uygun degerler ile doldurma
    - IQR yontemiyle sayısal sutunlardaki saykırı degerleri tespit etmek
    - Kategorik verileir label encoding ve one hot encoding ile donustur
    - veriyi train, validasyon ve test kumelerine ayir
    - sayısal ozelliklere standardization ve normalization uygula
"""


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split # egitim ve test veri seti olsuturur
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler

pd.set_option("display.max_columns", None)
df = pd.read_csv("musteri_verisi_ml_pratik.csv")

print(df.head())
print(df.info())

# eksik veri analizis
print(df.isnull().sum())

df_dropna = df.dropna() # eksik veriyi cıkart

print(f"Eksik veri cıktıktan sonra : \n{df_dropna}")


df_filled = df.copy()





