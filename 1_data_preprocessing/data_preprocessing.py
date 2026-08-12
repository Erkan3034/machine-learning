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

sayisaL_sutunlar = df.select_dtypes(include = "float64")

#sayisal sutunları median ile doldur
for sutun in sayisaL_sutunlar:
    median_degeri = df_filled[sutun].median()
    df_filled[sutun] = df_filled[sutun].fillna(median_degeri)


#kategorik sutunları en sık tekrar edenle doldur

df_filled["egitim"] = df_filled["egitim"].fillna(df_filled["egitim"].mode()[0])
print(f"Eksik veriler dolduruldutkan sonra: \n{df_filled}")

#IQR yontemiyle aykırı degerleri tespit etme

aykiri_deger_maskesi = pd.Series(False, index = df_filled.index)

for sutun in sayisaL_sutunlar:

    q1 = df_filled[sutun].quantile(0.25)
    q3 = df_filled[sutun].quantile(0.75)

    iqr = q3 - q1

    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr

    sutun_maskesi = (
        (df_filled[sutun] < alt_sinir) | (df_filled[sutun] > ust_sinir)
    )

    aykiri_deger_maskesi = aykiri_deger_maskesi | sutun_maskesi


    print(f"Aykırı değer sayısı: {sutun_maskesi.sum()}")

    if sutun_maskesi.any():
        print(f"Aykırı değerler: \n{df_filled.loc[sutun_maskesi, sutun]}")

print(f"En az bir aykırı değer içeren satırlar \n{df_filled.loc[aykiri_deger_maskesi]}")

# aykırı değer içeren satırları veri setinden çıkartalım
df_clean = df_filled.loc[~aykiri_deger_maskesi].copy()
df_clean.reset_index(drop=True, inplace=True)

print(f"Aykırı değerler çıktıktan sonra \n{df_clean}")

#  label encoding ve one-hot encoding 

label_encoder = LabelEncoder()

# hedef değişkeni sayısal hale getir
y = label_encoder.fit_transform(df_clean["satin_aldi"])

print(f"Hedef değişken sınıfları: \n {label_encoder.classes_}")
print(y)