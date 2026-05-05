# Hotel Cancellation Risk Prediction
****
### Latar Belakang
****
Sebuah hotel di Portugal mengalami penurunan pendapatan tahunan sebesar ~37% akibat pembatalan reservasi. Setiap kamar yang tidak terisi adalah perishable inventory berpotensi pendapatan hilang selamanya. Manajemen tidak memiliki sistem peringatan dini untuk mengidentifikasi pemesanan berisiko tinggi. 

Akibatnya:

- Tidak bisa melakukan overbooking yang terukur.

- Tidak bisa menawarkan intervensi preventif (misal: upsell, reminder, deposit non‑refundable).

- Efisiensi operasional terganggu (staffing, persediaan, revenue management).

[*Industri perhotelan*](https://link.springer.com/article/10.1007/s40558-025-00349-9) merupakan sektor jasa yang sangat bergantung pada kemampuan dalam mengelola permintaan pelanggan secara efektif.Permintaan terhadap kamar hotel tidak bersifat konstan, melainkan berfluktuasi seiring waktu akibat berbagai faktor seperti musim, perilaku wisatawan, serta dinamika ekonomi. Oleh karena itu, pemahaman terhadap pola permintaan menjadi aspek penting dalam menjaga kinerja operasional dan finansial hotel.

Salah satu mekanisme utama dalam pengelolaan permintaan adalah sistem reservasi, di mana pelanggan melakukan pemesanan sebelum tanggal kedatangan. Meskipun sistem ini membantu perencanaan, fleksibilitas yang diberikan kepada pelanggan juga memunculkan potensi pembatalan reservasi. [**Tingginya tingkat pembatalan dapat menyebabkan ketidaksesuaian antara permintaan yang diperkirakan dengan realisasi hunian**](https://www.hftp.org/news/4128776/free-hotel-cancellations-smart-strategy-or-revenue-risk)

---
### Problem Statement
****
## 2.1 Permasalahan Utama (Primary Problem) Hypothesis
Tingginya tingkat pembatalan reservasi **(cancellation rate)** menyebabkan ketidakpastian dalam perencanaan okupansi, yang berdampak langsung terhadap potensi kehilangan pendapatan **(revenue loss)** dan inefisiensi operasional hotel.

Untuk mengatasi permasalahan tersebut, diperlukan pendekatan berbasis data yang mampu mengidentifikasi reservasi dengan risiko pembatalan sejak awal proses pemesanan.

Oleh karena itu **Permasalahan utama dalam project ini adalah:**
<blockquote style="background: #CDDDF6; padding: 10px; border-left: 4px solid #A6C5F2; margin: 10px 0;">
<ul>
    <li>Bagaimana memanfaatkan data historis pemesanan untuk memprediksi probabilitas pembatalan reservasi secara akurat, .</li>
    <li>sehingga hotel dapat melakukan intervensi proaktif (seperti reminder, penyesuaian kebijakan deposit, atau strategi overbooking) guna meminimalkan revenue loss.</li>
</ul>
</blockquote>

Model yang dikembangkan bertujuan sebagai early warning system yang dapat membantu tim revenue management dan operasional dalam mengambil keputusan berbasis risiko.

Sebagai target performa, model diharapkan:

- Mampu mencapai recall ≥ 0,75 pada kelas pembatalan, untuk meminimalkan risiko false negative (booking berisiko tinggi yang tidak terdeteksi)

- Dengan tetap mempertimbangkan trade-off terhadap precision agar intervensi tetap efisien secara operasional

---

## 2.2 Pertanyaan Analitis Spesifik (Specific Analytical Questions)

Untuk menjawab permasalahan utama, project ini dirumuskan ke dalam beberapa pertanyaan analitis berikut:

| No | Pertanyaan Analitis                                                             | Tujuan                                               |
| -- | ------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 1  | Berapa tingkat pembatalan dan apakah terjadi class imbalance?                   | Menentukan baseline risiko dan kebutuhan strategi modeling |
| 2  | Faktor apa yang paling mempengaruhi pembatalan?                                 | Mengidentifikasi driver utama untuk intervensi             |
| 3  | Bagaimana perilaku customer berisiko tinggi? (lead_time, deposit, segment, dll) | Membentuk segmentasi risiko                                |
| 4  | Model mana yang paling optimal dalam mendeteksi pembatalan?                     | Memilih model terbaik untuk implementasi                   |
| 5  | Berapa potensi revenue yang dapat diselamatkan jika model digunakan?            | Justifikasi bisnis implementasi model                      |


---

### Analytical Approach
****
Analisis difokuskan untuk menjawab pertanyaan kunci:

**“Booking seperti apa yang berisiko tinggi dibatalkan, dan bagaimana hotel dapat bertindak sebelum pembatalan terjadi?”**

Dengan pendekatan ini, model yang dikembangkan tidak hanya berfungsi sebagai alat prediksi, tetapi diharapkan juga sebagai:

- Risk scoring system untuk mengklasifikasikan tingkat risiko pembatalan pada setiap reservasi

- Early warning system untuk mendeteksi potensi pembatalan sejak awal proses booking

- Decision support tool untuk membantu tim operasional dan revenue management dalam menentukan strategi intervensi yang lebih tepat sasaran

| Tahapan                                 | Fokus Utama                                                                                                               | Output yang Diharapkan                |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| **Data Understanding**                  | Mengidentifikasi struktur data, kualitas data, serta potensi risiko seperti missing value, duplicate, dan class imbalance | Pemahaman awal + potensi issue        |
| **EDA (Hypothesis-driven)**             | Menguji hipotesis terkait faktor pembatalan (lead time, deposit, segment, dll)                                            | Insight tentang driver cancellation   |
| **Data Cleaning & Feature Engineering** | Memastikan data siap modeling dan mencerminkan perilaku customer                                                          | Dataset yang lebih representatif      |
| **Modeling**                            | Membangun model untuk mengidentifikasi booking berisiko tinggi                                                            | Model prediksi risiko                 |
| **Evaluation (Business-oriented)**      | Evaluasi model dengan fokus pada recall dan trade-off bisnis                                                              | Model yang relevan secara operasional |
| **Business Impact Analysis**            | Mengukur potensi pengurangan cancellation dan peningkatan revenue                                                         | Justifikasi implementasi model        |
| **Deployment Scenario**                 | Simulasi penggunaan model dalam sistem operasional                                                                        | Gambaran implementasi nyata           |

### Analytical Approach
****
1. Data Collection & Preprocessing
- Handling missing values, duplicate removal
- Feature engineering dan selection
2. Exploratory Data Analysis (EDA)
Fokus Demografi:
Fokus Geografi:
Fokus Contract and Services:
Fokus Monetary:

3. Machine Learning Modeling
Binary classification problem (canceled vs non-canceled)
Algoritma: Logistic Regression, Decision Tree, Random Forest, XGBoost
Handling imbalanced data: SMOTE, RandomOverSampler, RandomUnderSampler, NearMiss, class_weight='balanced'
Hyperparameter tuning menggunakan RandomizedSearchCV
4. Model Interpretation
SHAP values untuk explainability

### Metrics Evaluation
****

- **Data Tidak Seimbang**: Dengan hanya 28% kelas positif, akurasi akan menyesatkan karena akan selalu memprediksi "tidak ada pembatalan"
- **Pertimbangan Biaya**: Rasio biaya FP/FN yang dramatis (3:1) membuat false positive jauh lebih mahal dibandingkan false negative
- **Metrik yang Dipilih**: **F-beta** adalah metrik optimal untuk skenario ini karena kita mementingkan kedua metrik antara recall dan precision
  
$$F_\beta = \frac{1 + \beta^2}{\frac{\beta^2}{\text{Recall}} + \frac{1}{\text{Precision}}}$$

Keterangan:
- untuk **F-0.5 Score, β = 0.5**
- **Presisi** = TP / (TP + FP) - Proporsi pengunjung yang diprediksi akan membatalkan pemesanan dan benar-benar membatalkan pemesanan

- **Recall** = TP / (TP + FN) - Proporsi pelanggan yang benar-benar membatalkan pemesanan dan teridentifikasi dengan tepat

- **β = 0.5** berarti precision diberi bobot **4x** lebih penting daripada recall

### Business Insight & Findings
****
**Key Findings**

<li><b>Cancellation adalah masalah multi-faktor, bukan single driver</b><br>
Tidak ada variabel tunggal dengan korelasi kuat terhadap pembatalan. 
Risk terbentuk dari kombinasi <b>timing (lead time), value (ADR), dan behavior (customer intent)</b>.</li>

<li><b>Lead time adalah driver paling konsisten</b><br>
Semakin jauh waktu booking, semakin tinggi ketidakpastian -> cancel rate meningkat signifikan hingga >50% pada very early booking.</li>

<li><b>Harga (ADR) memperbesar risiko pada kondisi tertentu</b><br>
ADR bukan faktor utama secara linear, tetapi menjadi <b>risk amplifier</b> ketika dikombinasikan dengan lead time tinggi.</li>

<li><b>Customer behavior lebih penting daripada pricing</b><br>
- Special requests -> menurunkan cancel (indikasi komitmen)<br>
- Previous cancellations -> meningkatkan cancel (repeat behavior)</li>

<li><b>Channel & segment menentukan tingkat fleksibilitas</b><br>
- OTA / Online TA -> cancel rate tinggi (fleksibel)<br>
- Direct / Corporate -> lebih stabil (komitmen tinggi)</li>

<li><b>Deposit policy adalah kontrol paling efektif terhadap revenue risk</b><br>
- No deposit -> risiko kehilangan revenue tertinggi<br>
- Non refundable -> cancel tinggi, tapi revenue relatif aman</li>

**7.1 Faktor Utama Penyebab Cancellation**
</div>

<ul>
<li><b>Lead Time (Driver Utama)</b><br>
Semakin jauh waktu booking, semakin tinggi ketidakpastian maka risiko cancel meningkat signifikan</li>

<li><b>Customer Intent (Penentu Komitmen)</b><br>
- Special requests : menurunkan cancel (high intent)<br>
- Previous cancellations : meningkatkan cancel (repeat behavior)</li>

<li><b>Pricing sebagai Risk Amplifier</b><br>
Harga tidak berdampak secara langsung, tetapi memperbesar risiko pada booking dengan lead time tinggi</li>
</ul>

---
 **7.2 Segmentasi High-Risk Customer**
</div>

<ul>
<li><b>High Risk Segment</b><br>
Booking dengan <b>lead time panjang + harga tinggi</b> -> probabilitas cancel tertinggi</li>

<li><b>Low Risk Segment</b><br>
Booking last minute : menunjukkan komitmen tinggi dan cancel rate rendah</li>

<li><b>Behavioral Risk</b><br>
Customer dengan riwayat cancel : akan cenderung mengulang perilaku yang sama</li>

</ul>

---
  **7.3 Channel dengan Risiko Tertinggi**
</div>

<ul>
<li><b>Online TA / OTA</b><br>
Memiliki cancel rate tertinggi karena fleksibilitas tinggi dalam perubahan booking</li>

<li><b>Direct & Corporate</b><br>
Lebih stabil karena komitmen lebih tinggi dan proses booking lebih terkontrol</li>

<li><b>Channel = Level of Control</b><br>
Semakin fleksibel channel, semakin tinggi risiko pembatalan</li>
</ul>

---

 **7.4 Dampak terhadap Revenue & Occupancy**
</div>

<ul>
<li><b>Cancellation tidak selalu berarti revenue loss</b><br>
Pada booking non refundable, hotel tetap menerima pembayaran meskipun terjadi pembatalan</li>

<li><b>Revenue Risk berasal dari No Deposit</b><br>
Booking tanpa deposit memiliki risiko kehilangan revenue tertinggi</li>

<li><b>Peluang Double Revenue</b><br>
Jika kamar dari no show dapat dijual kembali, hotel berpotensi mendapatkan revenue tambahan</li>

<li><b>Fokus utama: Revenue Risk, bukan Cancellation Rate</b><br>
Strategi bisnis harus berfokus pada dampak finansial, bukan hanya jumlah pembatalan</li>
</ul>


**Link Tableau** : https://public.tableau.com/views/PortugalHotelBookingDemandDashboard/Dashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

**Link Streamlit**: https://finpro-ds-it39wabbxb2kztzj9axplu.streamlit.app/

**Link PPT**: 
