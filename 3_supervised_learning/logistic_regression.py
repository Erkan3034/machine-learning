"""
Amaç:
    - UCI heart disease veri setini kullanarak logistic regression modeli ile ikili sınıflandırma problemi çözme
    - model, bir bireyin kallp hastalıgına sahip olup olmadıgını tahmit etmeyi amaclar ve accuracy metrigi ile değerlendirir

veri seti:
    - veri seti bireylere ait demografik ve klinik olcumlerini iceriyor
    - features: yas,cinsiyet,agri tipi, kolestrol, kan basıncı vb.
    - hedef degisken: 
        - 0 : hastalık yok
        - 1 : hastalık var
plan:
    - veri setini yükle ve temel analizleri yap
    - veri seti icerisinde eksik deger kotnrolü yağ gerekirse temizle
    - ozniteik ve hedef degiskenlerin ayrılması
    - egitim ve test veri setlerinin olusturulması
    - logistic regression modelinin tanımlanması ve egitilmesi
    - modelin test veri seti ile degerlendirilmesi

kurulumlar:
pip install pandas matplotlib scikit-learn ucimlrepo
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
from scipy.stats import pearson3
from linear_polynomial_lasso_ridge import X_train
from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.set_option("display.max_columns",None)
# veri setini yukle 
heart_disease = fetch_ucirepo(id = 45)
df = pd.DataFrame(data = heart_disease.data.features)
df["target"] = heart_disease.data.targets
df["target"] = df["target"].apply(lambda x : 0 if x >= 0.5 else 1)
print(df.head())

if df.isna().any().any():
    print(f"Eksik deger sayısı (temizlenmeden önce):\n{df.isna().sum()}")
    df = df.dropna()
    print("Eksik degerler temizlendi.")
else:
    print("Eksik deger bulunamadı") 

#  Öznitelik ve hedef değişkenlerin ayrılması
X = df.drop(["target"], axis = 1).values # features
y = df.target.values

#  Eğitim ve test veri setlerinin oluşturulması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#  Logistic regression modelinin tanımlanması ve eğitilmesi
log_reg = LogisticRegression(penalty="l2", C = 1, max_iter = 100)
log_reg.fit(X_train, y_train)

# Modelin test veri seti ile değerlendirilmesi
acc = log_reg.score(X_test, y_test)
print(f"Accuracy: {acc}")