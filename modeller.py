"""
Doğrusal ve Lojistik Regresyon - SIFIRDAN (yalnızca numpy ile) uygulama
=======================================================================
Hiçbir hazır makine öğrenimi kütüphanesi (scikit-learn vb.) kullanılmamıştır.

Gösterim:
    X : (m, n) özellik matrisi   (m = örnek sayısı, n = özellik sayısı)
    y : (m,)   etiketler, 0 veya 1
    w : (n,)   ağırlıklar
    b : skaler sapma (bias)
"""
import numpy as np


# ---------------------------------------------------------------------------
# 1) DOĞRUSAL REGRESYON (sınıflandırıcı olarak)
# ---------------------------------------------------------------------------
class DogrusalRegresyon:
    """
    Model      : y_hat = X·w + b              (çıktı sınırsız bir reel sayı)
    Kayıp      : MSE = (1/m) * Σ (y_hat - y)²
    Gradyanlar : dJ/dw = (2/m) * Xᵀ (y_hat - y)
                 dJ/db = (2/m) * Σ (y_hat - y)
    Sınıflandırma: y_hat >= esik (0.5) ise sınıf 1, değilse sınıf 0

    yontem="gd"     -> gradyan inişi ile iteratif çözüm
    yontem="normal" -> normal denklem ile kapalı-form çözüm: θ = (XᵀX)⁻¹ Xᵀy
    """

    def __init__(self, ogrenme_orani=0.01, iterasyon=5000, yontem="gd", esik=0.5):
        self.ogrenme_orani = ogrenme_orani
        self.iterasyon = iterasyon
        self.yontem = yontem
        self.esik = esik
        self.w = None
        self.b = None
        self.kayip_gecmisi = []

    def fit(self, X, y):
        m, n = X.shape
        y = y.astype(float)

        if self.yontem == "normal":
            # Bias'ı ağırlık vektörüne katmak için X'in başına 1'lerden oluşan sütun eklenir
            Xb = np.hstack([np.ones((m, 1)), X])
            # θ = (XᵀX)⁻¹ Xᵀ y   (ters alma yerine doğrusal sistem çözülür: sayısal olarak daha kararlı)
            theta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
            self.b, self.w = theta[0], theta[1:]
            self.kayip_gecmisi = [self._mse(X, y)]
            return self

        # Gradyan inişi
        self.w = np.zeros(n)
        self.b = 0.0
        self.kayip_gecmisi = []
        for _ in range(self.iterasyon):
            y_hat = X @ self.w + self.b
            hata = y_hat - y
            dw = (2 / m) * (X.T @ hata)
            db = (2 / m) * np.sum(hata)
            self.w -= self.ogrenme_orani * dw
            self.b -= self.ogrenme_orani * db
            self.kayip_gecmisi.append(np.mean(hata ** 2))
        return self

    def _mse(self, X, y):
        return np.mean((X @ self.w + self.b - y) ** 2)

    def ham_cikti(self, X):
        """Eşik uygulanmadan önceki sürekli çıktı (0-1 aralığında OLMAK ZORUNDA DEĞİL)."""
        return X @ self.w + self.b

    def predict(self, X):
        return (self.ham_cikti(X) >= self.esik).astype(int)


# ---------------------------------------------------------------------------
# 2) LOJİSTİK REGRESYON
# ---------------------------------------------------------------------------
def sigmoid(z):
    # Taşmayı (overflow) önlemek için z kırpılır
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LojistikRegresyon:
    """
    Model      : p = σ(X·w + b),   σ(z) = 1 / (1 + e^(-z))     (çıktı 0-1 arası olasılık)
    Kayıp      : Log-loss (ikili çapraz entropi)
                 J = -(1/m) * Σ [ y·log(p) + (1-y)·log(1-p) ]  (+ isteğe bağlı L2 cezası)
    Gradyanlar : dJ/dw = (1/m) * Xᵀ (p - y)  (+ (λ/m)·w)
                 dJ/db = (1/m) * Σ (p - y)
    Sınıflandırma: p >= esik (0.5) ise sınıf 1, değilse sınıf 0
    """

    def __init__(self, ogrenme_orani=0.1, iterasyon=5000, esik=0.5, l2=0.0):
        self.ogrenme_orani = ogrenme_orani
        self.iterasyon = iterasyon
        self.esik = esik
        self.l2 = l2
        self.w = None
        self.b = None
        self.kayip_gecmisi = []

    def _kayip(self, p, y, m):
        eps = 1e-12   # log(0) hatasını önlemek için
        p = np.clip(p, eps, 1 - eps)
        kayip = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        kayip += (self.l2 / (2 * m)) * np.sum(self.w ** 2)
        return kayip

    def fit(self, X, y):
        m, n = X.shape
        y = y.astype(float)
        self.w = np.zeros(n)
        self.b = 0.0
        self.kayip_gecmisi = []
        for _ in range(self.iterasyon):
            p = sigmoid(X @ self.w + self.b)
            hata = p - y
            dw = (1 / m) * (X.T @ hata) + (self.l2 / m) * self.w
            db = (1 / m) * np.sum(hata)
            self.w -= self.ogrenme_orani * dw
            self.b -= self.ogrenme_orani * db
            self.kayip_gecmisi.append(self._kayip(p, y, m))
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= self.esik).astype(int)
