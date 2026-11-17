"""
Veri Analizi (Keşifsel Veri Analizi - EDA)
------------------------------------------
Pima Indians Diabetes veri kümesini tanımak için kullanılır:
  - örnek / özellik sayısı, sınıf dağılımı
  - özelliklerin istatistikleri
  - eksik (0 olarak kodlanmış) değerler
  - verinin doğrusal olarak ayrılamadığını gösteren grafikler

Çalıştırma:  py veri_analizi.py
Grafikler 'sonuclar/' klasörüne kaydedilir (rapor için).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # grafikleri ekrana değil dosyaya çiz
import matplotlib.pyplot as plt

VERI_YOLU = os.path.join("data", "diabetes.csv")
CIKTI_KLASORU = "sonuclar"
os.makedirs(CIKTI_KLASORU, exist_ok=True)

df = pd.read_csv(VERI_YOLU)
ozellikler = [s for s in df.columns if s != "Outcome"]

print("=" * 60)
print("VERİ KÜMESİ GENEL BİLGİ")
print("=" * 60)
print(f"Örnek sayısı   : {df.shape[0]}")
print(f"Özellik sayısı : {len(ozellikler)}")
print(f"Özellikler     : {', '.join(ozellikler)}")

print("\nSınıf dağılımı (Outcome):")
sayim = df["Outcome"].value_counts().sort_index()
for sinif, adet in sayim.items():
    ad = "Diyabet yok (0)" if sinif == 0 else "Diyabet var (1)"
    print(f"  {ad:18s}: {adet:4d}  (%{100 * adet / len(df):.1f})")

print("\nTemel istatistikler:")
print(df.describe().T.round(2).to_string())

# Biyolojik olarak 0 olamayacak sütunlarda 0 değeri = ölçülmemiş (eksik) veri
sifir_olamaz = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\n0 olarak kodlanmış eksik değerler:")
for s in sifir_olamaz:
    n = (df[s] == 0).sum()
    print(f"  {s:15s}: {n:4d}  (%{100 * n / len(df):.1f})")

# ---------------------------------------------------------------- Grafikler
renkler = {0: "#3b82f6", 1: "#ef4444"}
etiket = {0: "Diyabet yok (0)", 1: "Diyabet var (1)"}

# 1) Sınıf dağılımı
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar([etiket[0], etiket[1]], sayim.values, color=[renkler[0], renkler[1]])
for i, v in enumerate(sayim.values):
    ax.text(i, v + 5, str(v), ha="center")
ax.set_ylabel("Örnek sayısı")
ax.set_title("Sınıf Dağılımı")
fig.tight_layout()
fig.savefig(os.path.join(CIKTI_KLASORU, "01_sinif_dagilimi.png"), dpi=150)
plt.close(fig)

# 2) Her özelliğin sınıflara göre histogramı
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, s in zip(axes.ravel(), ozellikler):
    for c in (0, 1):
        ax.hist(df.loc[df["Outcome"] == c, s], bins=25, alpha=0.55,
                color=renkler[c], label=etiket[c])
    ax.set_title(s)
axes[0, 0].legend()
fig.suptitle("Özelliklerin Sınıflara Göre Dağılımı (iç içe geçmiş dağılımlar)")
fig.tight_layout()
fig.savefig(os.path.join(CIKTI_KLASORU, "02_ozellik_histogramlari.png"), dpi=150)
plt.close(fig)

# 3) Korelasyon matrisi
kor = df.corr().values
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(kor, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(df.columns)))
ax.set_yticks(range(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=45, ha="right")
ax.set_yticklabels(df.columns)
for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        ax.text(j, i, f"{kor[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("Korelasyon Matrisi")
fig.tight_layout()
fig.savefig(os.path.join(CIKTI_KLASORU, "03_korelasyon.png"), dpi=150)
plt.close(fig)

# 4) En ayırt edici iki özellik: Glucose - BMI saçılım grafiği
temiz = df[(df["Glucose"] > 0) & (df["BMI"] > 0)]
fig, ax = plt.subplots(figsize=(7, 5.5))
for c in (0, 1):
    d = temiz[temiz["Outcome"] == c]
    ax.scatter(d["Glucose"], d["BMI"], s=14, alpha=0.6, color=renkler[c], label=etiket[c])
ax.set_xlabel("Glucose")
ax.set_ylabel("BMI")
ax.set_title("Glucose - BMI: sınıflar tek bir doğruyla ayrılamıyor")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(CIKTI_KLASORU, "04_glucose_bmi_sacilim.png"), dpi=150)
plt.close(fig)

# 5) PCA ile tüm özelliklerin 2 boyuta indirgenmesi (numpy SVD ile)
X = df[ozellikler].values.astype(float)
X = (X - X.mean(axis=0)) / X.std(axis=0)
_, S, Vt = np.linalg.svd(X, full_matrices=False)
Z = X @ Vt[:2].T
aciklanan = (S ** 2) / (S ** 2).sum()
fig, ax = plt.subplots(figsize=(7, 5.5))
for c in (0, 1):
    m = df["Outcome"].values == c
    ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.6, color=renkler[c], label=etiket[c])
ax.set_xlabel(f"1. Temel Bileşen (%{100 * aciklanan[0]:.1f})")
ax.set_ylabel(f"2. Temel Bileşen (%{100 * aciklanan[1]:.1f})")
ax.set_title("PCA ile 2B izdüşüm: sınıflar iç içe")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(CIKTI_KLASORU, "05_pca_2b.png"), dpi=150)
plt.close(fig)

print(f"\nGrafikler '{CIKTI_KLASORU}/' klasörüne kaydedildi.")
