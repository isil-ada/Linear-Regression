# Linear-Regression
Aşağıda, iki farklı linear regresyon modelini karşılaştıran ve README dosyası olarak GitHub deposuna eklenebilecek örnek bir metin yer almaktadır. Her iki modelin mantığı, tahmin sonuçları ve görselleştirme çıktıları maliyet (cost) açısından yorumlanmıştır.

---

# 📊 Linear Regression - Least Squares Yöntemi Karşılaştırması

Bu çalışmada, aynı veri seti (`advertising.csv`) üzerinde iki farklı doğrusal regresyon modeli uygulanmıştır:

1. **Model 1 - NumPy Tabanlı Yöntem** (`LeastSquareMethod.py`)
2. **Model 2 - Manuel (Kütüphanesiz) Matris Hesaplamalı Yöntem** (`LeastSquareMethod2.py`)

Her iki modelde de doğrusal regresyonun temel formülü olan:

$$
\theta = (X^TX)^{-1}X^Ty
$$

kullanılmıştır. Fakat ikinci modelde bu işlem tamamen manuel olarak, matris çarpımı ve ters alma işlemleri Python listeleri ile gerçekleştirilmiştir.

---

## 🔧 Kullanılan Veri Seti

* Veri seti: `advertising.csv`
* Hedef değişken: `Sales` (bin adet)
* Özellikler (features): `TV`, `Radio`, `Newspaper` reklam harcamaları (bin \$)

---

## 🔍 Model Karşılaştırması

| Özellik           | Model 1 (NumPy)                 | Model 2 (Manuel)                  |
| ----------------- | ------------------------------- | --------------------------------- |
| Hesaplama yöntemi | NumPy `@` operatörü ve `.inv()` | Manuel transpose, matmul, inverse |
| Kodlama zorluğu   | Düşük                           | Orta - Yüksek                     |
| Hız / Performans  | Yüksek                          | Düşük                             |
| Esneklik          | Düşük (kütüphane bağımlı)       | Yüksek (bağımsız çalışabilir)     |
| Tahmin Sonuçları  | %100 aynı                       | %100 aynı                         |

Her iki modelde elde edilen `theta` (ağırlık) değerleri ve tahmin edilen `y_pred` değerleri birebir aynıdır. Bu da manuel yöntemin doğru çalıştığını göstermektedir.

---

## 📉 Maliyet (Cost) Fonksiyonu Karşılaştırması

Her iki modelde de Mean Squared Error (MSE) değeri aşağıdaki şekilde hesaplanabilir:

$$
\text{Cost} = \frac{1}{n} \sum (y_{\text{gerçek}} - y_{\text{tahmin}})^2
$$

Her iki modelde de MSE değeri birebir aynı çıkmaktadır. Bu, performans açısından modellerin eşit olduğunu; farkın yalnızca uygulama yöntemi ve esneklikte olduğunu göstermektedir.

---

## 📷 Görselleştirmeler

Aşağıda her iki modelin çıktılarından elde edilen tahmin vs. gerçek değer karşılaştırmalarına ilişkin grafikler bulunmaktadır.

### Model 1 (NumPy):
#### TV Reklam Harcaması vs. Satış
![Figure_1](https://github.com/user-attachments/assets/00bdbea5-d85f-4698-b33d-f1b925d647bd) 

#### Radio Reklam Harcaması vs. Satış
![Figure_2](https://github.com/user-attachments/assets/f4f2e95d-6f26-4227-8fc3-5e228bbfcd83)

#### Newspaper Reklam Harcaması vs. Satış
![Figure_3](https://github.com/user-attachments/assets/187be424-4852-413c-a1b6-84d158ff8c41)


### Model 2 (Manuel):

#### TV Reklam Harcaması vs. Satış
![Figure_4](https://github.com/user-attachments/assets/ff86bd89-e27c-4029-8e6f-7de6fc51c658)

#### Radio Reklam Harcaması vs. Satış
![Figure_5](https://github.com/user-attachments/assets/bfbb7412-f653-4c11-abde-f498b3175229)

#### Newspaper Reklam Harcaması vs. Satış
![Figure_6](https://github.com/user-attachments/assets/03be4412-6459-416b-9c7c-6a991fda1b0c)

*Not: Görseller yukarıdaki grafiklerle birebir aynı çıktılar vermektedir.*

---

## 📝 Sonuç

Bu çalışma, doğrusal regresyonun hem kütüphane tabanlı hem de manuel yöntemle nasıl gerçekleştirilebileceğini göstermektedir. Her iki yaklaşım da benzer doğruluk ve maliyet değerleri verirken, uygulama yöntemleri açısından farklılık göstermektedir:

* **Model 1**, hızlı geliştirme ve yüksek performans açısından avantajlıdır.
* **Model 2**, matematiksel temelin pekiştirilmesi ve bağımsız çalışabilme açısından değerlidir.


