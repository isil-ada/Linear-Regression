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

## 💰 Maliyet (Cost) Karşılaştırması

Modellerin başarımı, doğrusal regresyonda sıkça kullanılan **Ortalama Kare Hatası (Mean Squared Error - MSE)** metriği ile değerlendirilmiştir:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n}(y_{\text{gerçek}}^{(i)} - y_{\text{tahmin}}^{(i)})^2
$$

* **Model 1** (NumPy ile hesaplanan):
  NumPy’nin optimize edilmiş lineer cebir fonksiyonları (`np.linalg.inv`, `@`) kullanılarak `θ` parametreleri hızlı ve doğru biçimde hesaplanmıştır. Elde edilen `y_pred` (tahmin edilen satışlar), gerçek `y` değerleriyle karşılaştırılmış ve düşük bir MSE değeri elde edilmiştir.

* **Model 2** (manuel matris işlemleriyle):
  Transpoz alma, matris çarpımı ve ters alma gibi işlemler hiçbir hazır matematiksel kütüphane kullanılmadan elle kodlanmıştır. Bu yaklaşımda `y_pred` hesaplanırken yalnızca Python listeleri kullanılmıştır. Sonuç olarak aynı tahmin değerleri ve **birebir aynı** MSE değeri elde edilmiştir.

> ✅ Her iki modelde de `y_pred` vektörü aynı olduğu için, **maliyet fonksiyonu değeri eşittir**.
> Bu durum, manuel implementasyonun matematiksel olarak doğru çalıştığını ve teorik formülü doğru şekilde takip ettiğini göstermektedir.

---

### 🧠 Modeller Arası Yorum ve Farklılıkların Değerlendirilmesi

Maliyet açısından aynı sonucun elde edilmesine rağmen, iki model arasında önemli farklar bulunmaktadır:

| Kriter                  | Model 1 (NumPy)                        | Model 2 (Manuel Python)                             |
| ----------------------- | -------------------------------------- | --------------------------------------------------- |
| **Kütüphane Kullanımı** | NumPy, Pandas, Matplotlib              | Sadece Python yerleşik veri yapıları                |
| **Performans**          | Optimize edilmiştir, çok hızlı çalışır | Büyük veri için verimsiz, sadece eğitim amaçlı      |
| **Okunabilirlik**       | Kod kısa, net ve anlaşılır             | Kod daha uzun ve karmaşık                           |
| **Eğitimsel Katkı**     | Temel kullanım sağlar                  | Matris hesaplarını daha iyi kavramaya yardımcı olur |
| **Esneklik**            | NumPy gerektirir                       | Hiçbir dış kütüphaneye ihtiyaç duymaz               |

**Yorum:**
Her iki model, aynı doğrusal regresyon denklemine dayansa da, uygulama yaklaşımları farklıdır. NumPy tabanlı yöntem, endüstride yaygın olarak kullanılan, yüksek performanslı ve güvenilir bir yaklaşımdır. Öte yandan, manuel olarak yazılmış matris işlemleri modeli, algoritmanın mantığını derinlemesine anlamak isteyenler için büyük bir eğitim değerine sahiptir. Ancak büyük veri setlerinde kullanılabilirliği sınırlıdır ve hata yapma riski daha yüksektir.

Bu nedenle:

* **Gerçek dünya uygulamaları** için NumPy tabanlı model tercih edilmelidir.
* **Eğitsel veya akademik amaçlar** için manuel model faydalı bir alternatiftir.
  
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

---

## 📝 Sonuç

Bu çalışma, doğrusal regresyon modelinin hem hazır bilimsel kütüphanelerle (NumPy) hem de manuel matris işlemleriyle nasıl uygulanabileceğini göstermektedir. Her iki model de aynı tahmin sonuçlarını ve maliyet (cost) değerlerini üretmiş, böylece matematiksel yöntemin doğruluğu ve tutarlılığı doğrulanmıştır.

* NumPy kullanan Model 1, yüksek performans ve sade kod yapısıyla özellikle pratik uygulamalarda avantajlıdır.
* Manuel hesaplama yapan Model 2, algoritmanın temelini öğrenmek ve dış kütüphane olmadan işlem yapmak isteyenler için idealdir.

Sonuç olarak, her iki model de doğrusal regresyon problemini başarıyla çözmektedir. Tercih, kullanım amacına ve ortam koşullarına göre belirlenmelidir.




