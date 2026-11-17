"""
Performans Değerlendirme Metrikleri (numpy ile elle yazılmıştır)
"""
import numpy as np


def karmasiklik_matrisi(y_gercek, y_tahmin):
    """
                    Tahmin 0   Tahmin 1
        Gerçek 0  [   TN         FP   ]
        Gerçek 1  [   FN         TP   ]
    """
    tn = np.sum((y_gercek == 0) & (y_tahmin == 0))
    fp = np.sum((y_gercek == 0) & (y_tahmin == 1))
    fn = np.sum((y_gercek == 1) & (y_tahmin == 0))
    tp = np.sum((y_gercek == 1) & (y_tahmin == 1))
    return np.array([[tn, fp], [fn, tp]])


def _bol(a, b):
    return a / b if b > 0 else 0.0


def tum_metrikler(y_gercek, y_tahmin):
    (tn, fp), (fn, tp) = karmasiklik_matrisi(y_gercek, y_tahmin)
    dogruluk = _bol(tp + tn, tp + tn + fp + fn)       # Accuracy
    kesinlik = _bol(tp, tp + fp)                      # Precision
    duyarlilik = _bol(tp, tp + fn)                    # Recall / Sensitivity
    ozgulluk = _bol(tn, tn + fp)                      # Specificity
    f1 = _bol(2 * kesinlik * duyarlilik, kesinlik + duyarlilik)
    return {
        "Accuracy": dogruluk,
        "Precision": kesinlik,
        "Recall": duyarlilik,
        "Specificity": ozgulluk,
        "F1": f1,
    }


def roc_egrisi(y_gercek, skor):
    """Her olası eşik için (FPR, TPR) çiftlerini hesaplar."""
    esikler = np.r_[np.inf, np.sort(np.unique(skor))[::-1], -np.inf]
    P = np.sum(y_gercek == 1)
    N = np.sum(y_gercek == 0)
    fpr, tpr = [], []
    for t in esikler:
        tahmin = (skor >= t).astype(int)
        tpr.append(np.sum((tahmin == 1) & (y_gercek == 1)) / P)
        fpr.append(np.sum((tahmin == 1) & (y_gercek == 0)) / N)
    return np.array(fpr), np.array(tpr)


def auc(fpr, tpr):
    """ROC eğrisi altındaki alan (yamuk kuralı)."""
    sira = np.argsort(fpr)
    fpr, tpr = fpr[sira], tpr[sira]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
