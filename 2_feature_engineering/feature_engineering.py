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

import pandas as pandas

df = pandas.read_csv("oznitelik_muhendisligi_pratik.csv")
print(df.head())
