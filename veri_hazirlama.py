"""
Veri Ön İşleme
--------------
1) Veriyi okuma
2) 0 olarak kodlanmış eksik değerleri NaN yapma
3) Eğitim / test ayrımı (tabakalı - sınıf oranları korunur)
4) Eksik değerleri EĞİTİM kümesinin medyanıyla doldurma
5) Standardizasyon (EĞİTİM kümesinin ortalama ve std'si ile)

Not: 4. ve 5. adımlarda istatistikler yalnızca eğitim verisinden hesaplanır.
Böylece test verisi hakkında modele bilgi sızdırılmaz (data leakage önlenir).
Tüm adımlar numpy/pandas ile elle yazılmıştır.
"""
import numpy as np
import pandas as pd

HEDEF = "Outcome"
SIFIR_OLAMAZ = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def veriyi_oku(yol):
    df = pd.read_csv(yol)
    # Biyolojik olarak 0 olamayacak değerler aslında ölçülmemiş (eksik) veridir
    df[SIFIR_OLAMAZ] = df[SIFIR_OLAMAZ].replace(0, np.nan)
    return df


def egitim_test_ayir(X, y, test_orani=0.2, tohum=42):
    """Tabakalı (stratified) ayrım: her sınıftan aynı oranda test örneği alınır."""
    rng = np.random.default_rng(tohum)
    test_idx = []
    for sinif in np.unique(y):
        idx = np.where(y == sinif)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_orani))
        test_idx.extend(idx[:n_test])
    test_idx = np.array(sorted(test_idx))
    egitim_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    rng.shuffle(egitim_idx)
    return X[egitim_idx], X[test_idx], y[egitim_idx], y[test_idx]


def eksikleri_doldur(X_egitim, X_test):
    medyan = np.nanmedian(X_egitim, axis=0)
    X_egitim = np.where(np.isnan(X_egitim), medyan, X_egitim)
    X_test = np.where(np.isnan(X_test), medyan, X_test)
    return X_egitim, X_test, medyan


def standartlastir(X_egitim, X_test):
    """z = (x - ortalama) / std  ->  her özellik ortalama 0, std 1 olur."""
    ort = X_egitim.mean(axis=0)
    std = X_egitim.std(axis=0)
    std[std == 0] = 1.0
    return (X_egitim - ort) / std, (X_test - ort) / std, ort, std


def veriyi_hazirla(yol, test_orani=0.2, tohum=42):
    df = veriyi_oku(yol)
    ozellikler = [s for s in df.columns if s != HEDEF]
    X = df[ozellikler].values.astype(float)
    y = df[HEDEF].values.astype(int)

    X_egitim, X_test, y_egitim, y_test = egitim_test_ayir(X, y, test_orani, tohum)
    X_egitim, X_test, _ = eksikleri_doldur(X_egitim, X_test)
    X_egitim, X_test, _, _ = standartlastir(X_egitim, X_test)
    return X_egitim, X_test, y_egitim, y_test, ozellikler
