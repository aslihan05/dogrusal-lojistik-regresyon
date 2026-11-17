"""
ANA PROGRAM
-----------
Doğrusal ve lojistik regresyonu aynı eğitim/test verisi üzerinde çalıştırır,
metrikleri hesaplar, karşılaştırır ve grafikleri 'sonuclar/' klasörüne kaydeder.

Çalıştırma:  py main.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from veri_hazirlama import veriyi_hazirla
from modeller import DogrusalRegresyon, LojistikRegresyon
from metrikler import tum_metrikler, karmasiklik_matrisi, roc_egrisi, auc

VERI_YOLU = os.path.join("data", "diabetes.csv")
CIKTI = "sonuclar"
os.makedirs(CIKTI, exist_ok=True)

# ------------------------------------------------------------------ 1) Veri
X_egitim, X_test, y_egitim, y_test, ozellikler = veriyi_hazirla(VERI_YOLU, test_orani=0.2, tohum=42)
print(f"Eğitim kümesi: {X_egitim.shape[0]} örnek  (sınıf 1 oranı: %{100 * y_egitim.mean():.1f})")
print(f"Test kümesi  : {X_test.shape[0]} örnek  (sınıf 1 oranı: %{100 * y_test.mean():.1f})")

# ------------------------------------------------------------------ 2) Eğitim
dogrusal = DogrusalRegresyon(ogrenme_orani=0.01, iterasyon=5000, yontem="gd").fit(X_egitim, y_egitim)
lojistik = LojistikRegresyon(ogrenme_orani=0.1, iterasyon=5000).fit(X_egitim, y_egitim)

# Doğrulama: gradyan inişi, normal denklemle aynı çözüme ulaşmalı
dogrusal_normal = DogrusalRegresyon(yontem="normal").fit(X_egitim, y_egitim)
fark = np.max(np.abs(dogrusal.w - dogrusal_normal.w))
print(f"\n[Kontrol] Doğrusal regresyon: GD ile normal denklem ağırlıkları arasındaki en büyük fark = {fark:.2e}")

# ------------------------------------------------------------------ 3) Değerlendirme
modeller = {"Doğrusal Regresyon": dogrusal, "Lojistik Regresyon": lojistik}
skorlar = {
    "Doğrusal Regresyon": dogrusal.ham_cikti(X_test),
    "Lojistik Regresyon": lojistik.predict_proba(X_test),
}

satirlar = []
for ad, model in modeller.items():
    m_egitim = tum_metrikler(y_egitim, model.predict(X_egitim))
    m_test = tum_metrikler(y_test, model.predict(X_test))
    fpr, tpr = roc_egrisi(y_test, skorlar[ad])
    satir = {"Model": ad, "Eğitim Accuracy": m_egitim["Accuracy"]}
    satir.update({k: v for k, v in m_test.items()})
    satir["AUC"] = auc(fpr, tpr)
    satirlar.append(satir)

tablo = pd.DataFrame(satirlar).set_index("Model")
print("\n" + "=" * 90)
print("TEST KÜMESİ SONUÇLARI (eşik = 0.5)")
print("=" * 90)
print(tablo.round(4).to_string())
tablo.round(4).to_csv(os.path.join(CIKTI, "metrikler.csv"), encoding="utf-8-sig")

for ad, model in modeller.items():
    cm = karmasiklik_matrisi(y_test, model.predict(X_test))
    print(f"\n{ad} - Karmaşıklık matrisi [[TN FP] [FN TP]]:\n{cm}")

ham = skorlar["Doğrusal Regresyon"]
print(f"\nDoğrusal regresyon ham çıktı aralığı (test): [{ham.min():.3f}, {ham.max():.3f}]")
print(f"  0'ın altında: {np.sum(ham < 0)} örnek,  1'in üstünde: {np.sum(ham > 1)} örnek")

agirliklar = pd.DataFrame({
    "Doğrusal (w)": dogrusal.w,
    "Lojistik (w)": lojistik.w,
}, index=ozellikler)
print("\nÖğrenilen ağırlıklar (standartlaştırılmış özellikler üzerinde):")
print(agirliklar.round(4).to_string())
agirliklar.round(4).to_csv(os.path.join(CIKTI, "agirliklar.csv"), encoding="utf-8-sig")

# ------------------------------------------------------------------ 4) Grafikler
RENK = {"Doğrusal Regresyon": "#f59e0b", "Lojistik Regresyon": "#8b5cf6"}

# 6) Kayıp eğrileri
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(dogrusal.kayip_gecmisi, color=RENK["Doğrusal Regresyon"])
axes[0].set_title("Doğrusal Regresyon - MSE")
axes[1].plot(lojistik.kayip_gecmisi, color=RENK["Lojistik Regresyon"])
axes[1].set_title("Lojistik Regresyon - Log-loss")
for ax in axes:
    ax.set_xlabel("İterasyon")
    ax.set_ylabel("Kayıp")
    ax.grid(alpha=0.3)
fig.suptitle("Gradyan İnişi Sırasında Eğitim Kaybı")
fig.tight_layout()
fig.savefig(os.path.join(CIKTI, "06_kayip_egrileri.png"), dpi=150)
plt.close(fig)

# 7) Karmaşıklık matrisleri
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, (ad, model) in zip(axes, modeller.items()):
    cm = karmasiklik_matrisi(y_test, model.predict(X_test))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Tahmin 0", "Tahmin 1"])
    ax.set_yticklabels(["Gerçek 0", "Gerçek 1"])
    ax.set_title(ad)
fig.suptitle("Test Kümesi Karmaşıklık Matrisleri")
fig.tight_layout()
fig.savefig(os.path.join(CIKTI, "07_karmasiklik_matrisleri.png"), dpi=150)
plt.close(fig)

# 8) ROC eğrileri
fig, ax = plt.subplots(figsize=(6, 5.5))
for ad in modeller:
    fpr, tpr = roc_egrisi(y_test, skorlar[ad])
    ax.plot(fpr, tpr, color=RENK[ad], lw=2, label=f"{ad} (AUC = {auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Rastgele tahmin")
ax.set_xlabel("Yanlış Pozitif Oranı (FPR)")
ax.set_ylabel("Doğru Pozitif Oranı (TPR)")
ax.set_title("ROC Eğrileri (Test Kümesi)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(CIKTI, "08_roc_egrileri.png"), dpi=150)
plt.close(fig)

# 9) Model çıktılarının dağılımı: doğrusal çıktı [0,1] dışına taşar, lojistik taşmaz
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for ax, ad in zip(axes, modeller):
    s = skorlar[ad]
    bins = np.linspace(min(s.min(), 0) - 0.05, max(s.max(), 1) + 0.05, 30)
    ax.hist(s[y_test == 0], bins=bins, alpha=0.6, color="#3b82f6", label="Gerçek sınıf 0")
    ax.hist(s[y_test == 1], bins=bins, alpha=0.6, color="#ef4444", label="Gerçek sınıf 1")
    ax.axvline(0.5, color="k", ls="--", label="Eşik = 0.5")
    ax.axvspan(bins[0], 0, color="gray", alpha=0.15)
    ax.axvspan(1, bins[-1], color="gray", alpha=0.15)
    ax.set_title(f"{ad} - çıktı dağılımı")
    ax.set_xlabel("Model çıktısı")
    ax.legend()
fig.suptitle("Gri alanlar: [0, 1] aralığı dışı (olasılık olarak yorumlanamaz)")
fig.tight_layout()
fig.savefig(os.path.join(CIKTI, "09_cikti_dagilimlari.png"), dpi=150)
plt.close(fig)

# 10) Metrik karşılaştırma çubuk grafiği
gosterilecek = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC"]
x = np.arange(len(gosterilecek))
fig, ax = plt.subplots(figsize=(10, 4.5))
for i, ad in enumerate(modeller):
    degerler = tablo.loc[ad, gosterilecek].values
    cubuk = ax.bar(x + (i - 0.5) * 0.38, degerler, 0.38, color=RENK[ad], label=ad)
    ax.bar_label(cubuk, fmt="%.3f", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(gosterilecek)
ax.set_ylim(0, 1.05)
ax.set_title("Test Kümesi Performans Karşılaştırması")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(CIKTI, "10_metrik_karsilastirma.png"), dpi=150)
plt.close(fig)

print(f"\nTüm sonuçlar '{CIKTI}/' klasörüne kaydedildi.")
