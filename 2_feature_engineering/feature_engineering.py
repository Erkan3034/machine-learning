"""
öznitelik mühendisliği

Amaç:
    - Mevcut sütunlardan yeni öznitelik üretme mantığını basit bir ornek ile uygulama
    - Korelasyon üzerinden model daha faydalı olabilecek öznitelikleri seçme mantığını öğrenme

Adımlar:
    - gerekli kkutuphanelerin yuklenmesei
    - veri setini yukleme
    - mevcut sutunlardan yeni oznitelikler uretme
    - hedef değişken ile oznitekşikler arasındaki korelasyonları inceleme
    - Mutlak korleasyon değerine göre yüksek lan özniteliklerin seçilmesi(feature selection)


kurulumlar:
pip install pandas


"""

from pandas._libs.tslibs import period
from pandas._libs.tslibs import period
import pandas as pd

df = pd.read_csv("oznitelik_muhendisligi_pratik.csv")
print(df.head())
pd.set_option("display.max_columns", None)

print(df.shape)

# mevcut sütunlardan yeni öznitelikler üretmek(feature extraction)
df["deneyim_orani"] = df["deneyim_yili"] / df["yas"]

df["yillik_harcama_tahmini"] = df["aylik_harcama"] * 12

print(df.head())

# hedef degisken ile oznitelikler arasındaki korelasyonları inceleme
sayisal_df = df.drop("sehir", axis=1)
korelasyonlar = sayisal_df.corr(numeric_only=True)["performans_puani"].sort_values(ascending=False)
print(f"korelasyonlar: \n{korelasyonlar}")

"""
performans_puani          1.000000
deneyim_orani             0.821244(yüksek pozitif korelasyon)
deneyim_yili              0.597232(orta yüksek pozitif korelasyon)
yillik_harcama_tahmini    0.317301(orta pozitif korelasyon)
aylik_harcama             0.317301(orta pozitif korelasyon)
yas                      -0.224902
uyelik_suresi_ay         -0.238212

"""

#mutlak korelasyon degerine gore yüksek olan iznitelikleirn secilmesi(feature selection)
secilen_ozellikler= korelasyonlar[abs(korelasyonlar) > 0.75].index.tolist()
secilen_ozellikler.remove("performans_puani")
print(secilen_ozellikler)