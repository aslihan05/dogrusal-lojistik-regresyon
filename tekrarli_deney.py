"""
Ek Deney: Farklı Eğitim/Test Ayrımlarında Tekrar
------------------------------------------------
Tek bir eğitim/test ayrımında görülen küçük farkların rastlantısal olup
olmadığını görmek için deney, 10 farklı rastgele tohumla oluşturulan
tabakalı ayrımlar üzerinde tekrarlanır.

Çalıştırma:  py tekrarli_deney.py
Çıktılar: sonuclar/tekrarli_deney.csv, sonuclar/11_tekrarli_deney_f1.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from veri_hazirlama import veriyi_hazirla
from modeller import DogrusalRegresyon, LojistikRegresyon
from metrikler import tum_metrikler, roc_egrisi, auc

VERI_YOLU = os.path.join("data", "diabetes.csv")
CIKTI = "sonuclar"
os.makedirs(CIKTI, exist_ok=True)

TOHUMLAR = range(1, 11)
METRIKLER = ["Accuracy", "Recall", "F1", "AUC"]

# ------------------------------------------------------------------ Deney
kayitlar = []
for tohum in TOHUMLAR:
    X_egitim, X_test, y_egitim, y_test, _ = veriyi_hazirla(VERI_YOLU, test_orani=0.2, tohum=tohum)
    dogrusal = DogrusalRegresyon(ogrenme_orani=0.01, iterasyon=5000).fit(X_egitim, y_egitim)
    lojistik = LojistikRegresyon(ogrenme_orani=0.1, iterasyon=5000).fit(X_egitim, y_egitim)

    for ad, model, skor in [("Doğrusal Regresyon", dogrusal, dogrusal.ham_cikti(X_test)),
                            ("Lojistik Regresyon", lojistik, lojistik.predict_proba(X_test))]:
        m = tum_metrikler(y_test, model.predict(X_test))
        fpr, tpr = roc_egrisi(y_test, skor)
        kayitlar.append({"Tohum": tohum, "Model": ad, "Accuracy": m["Accuracy"],
                         "Recall": m["Recall"], "F1": m["F1"], "AUC": auc(fpr, tpr)})

df = pd.DataFrame(kayitlar)
df.round(4).to_csv(os.path.join(CIKTI, "tekrarli_deney.csv"), index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ Özet
print("10 farklı ayrım üzerinde test performansı (ortalama ± standart sapma)")
print("=" * 70)
ozet = df.groupby("Model")[METRIKLER].agg(["mean", "std"])
for ad in ozet.index:
    satir = ", ".join(f"{k}={ozet.loc[ad, (k, 'mean')]:.3f} ± {ozet.loc[ad, (k, 'std')]:.3f}"
                      for k in METRIKLER)
    print(f"{ad}: {satir}")

# Her ayrımda hangi model daha iyi?
d = df[df["Model"] == "Doğrusal Regresyon"].set_index("Tohum")[METRIKLER]
l = df[df["Model"] == "Lojistik Regresyon"].set_index("Tohum")[METRIKLER]
print("\nHer metrik için 10 ayrımdaki karşılaştırma (lojistik - doğrusal):")
for k in METRIKLER:
    esit = np.isclose(l[k], d[k])
    lojistik_iyi = (l[k] > d[k]) & ~esit
    dogrusal_iyi = (l[k] < d[k]) & ~esit
    print(f"  {k:9s}: lojistik daha iyi {lojistik_iyi.sum()}, eşit {esit.sum()}, "
          f"doğrusal daha iyi {dogrusal_iyi.sum()}")

# ------------------------------------------------------------------ Grafik
fig, ax = plt.subplots(figsize=(10, 4.2))
x = list(TOHUMLAR)
ax.plot(x, d["F1"].values, "o-", color="#f59e0b", label="Doğrusal Regresyon")
ax.plot(x, l["F1"].values, "s-", color="#8b5cf6", label="Lojistik Regresyon")
ax.set_xticks(x)
ax.set_xlabel("Rastgele ayrım (tohum) no")
ax.set_ylabel("Test F1-skoru")
ax.set_title("10 Farklı Eğitim/Test Ayrımında F1-skoru")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(CIKTI, "11_tekrarli_deney_f1.png"), dpi=150)
plt.close(fig)

print(f"\nSonuçlar '{CIKTI}/' klasörüne kaydedildi.")
