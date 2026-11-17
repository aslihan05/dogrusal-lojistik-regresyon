# Doğrusal ve Lojistik Regresyon ile İkili Sınıflandırma

Makine Öğrenimine Giriş dersi proje ödevi.

Bu projede, iki sınıflı ve doğrusal olarak ayrılamayan **Pima Indians Diabetes** veri kümesinde
doğrusal regresyon ve lojistik regresyon algoritmaları **sıfırdan (yalnızca NumPy ile)** kodlanmış,
sınıflandırma performansları aynı test verisi üzerinde karşılaştırılmıştır.
Hiçbir hazır makine öğrenimi kütüphanesi (ör. scikit-learn) kullanılmamıştır.

## Veri Kümesi

- **Kaynak:** [Pima Indians Diabetes Database (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Örnek sayısı:** 768 (500 diyabet yok / 268 diyabet var)
- **Özellikler (8):** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Hedef:** Outcome (0 / 1)

## Dosyalar

| Dosya | İçerik |
|---|---|
| `data/diabetes.csv` | Veri kümesi |
| `veri_analizi.py` | Keşifsel veri analizi ve grafikler |
| `veri_hazirlama.py` | Eksik değer işleme, tabakalı eğitim/test ayrımı, standardizasyon |
| `modeller.py` | `DogrusalRegresyon` ve `LojistikRegresyon` sınıfları |
| `metrikler.py` | Karmaşıklık matrisi, accuracy, precision, recall, F1, ROC, AUC |
| `main.py` | Modellerin eğitimi, değerlendirilmesi ve sonuç grafikleri |
| `tekrarli_deney.py` | 10 farklı eğitim/test ayrımında tekrarlanan ek deney |
| `sonuclar/` | Üretilen grafikler ve CSV tabloları |

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python veri_analizi.py
python main.py
python tekrarli_deney.py
```

## Sonuçlar (test kümesi, eşik = 0.5)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Doğrusal Regresyon | 0.773 | 0.757 | 0.519 | 0.615 | 0.833 |
| Lojistik Regresyon | 0.773 | 0.744 | 0.537 | 0.624 | 0.836 |

Ayrıntılı analiz ve yorumlar proje raporunda yer almaktadır.
