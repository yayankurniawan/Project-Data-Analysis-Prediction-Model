# Project Data Analysis Prediction Model

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/0ac077cf-efe3-4624-9bf2-26f11115c9f7" />

---

## Business Understanding
FinanKu adalah sebuah startup fintech imajiner yang memberikan fasilitas simpanan dan pinjaman kepada nasabahnya. Jasa yang mereka tawarkan di antaranya tabungan, deposito, pinjaman tanpa agunan, kartu kredit, dan pembiayaan kendaraan mobil dan motor.

Saat ini, FinanKu memiliki pelanggan sebanyak ~20.000 yang tersebar di 3 kota besar di Indonesia; Jakarta, Bandung, dan Surabaya. Angka ini cukup besar mengingat FinanKu baru berjalan selama 1,5 tahun, di mana diekspektasikan dalam 3 tahun ke depan pelanggan mereka akan berjumlah 300.000+.

Perkembangan yang cepat ini membuat para stakeholders di divisi kredit FinanKu semakin berhati-hati dalam menyalurkan kredit yang dimiliki agar tidak mengalami gagal bayar, khususnya dari lini Kartu Kredit yang memiliki fitur instant approval dalam 1 menit.

### Problem Statement
Kekhawatiran adanya keterlambatan pembayaran kartu kredit pada FinanKu yang akan merugikan bisnis sehingga orang-orang yang memiliki potensi untuk mengalami keterlambatan bayar dapat diprediksi lebih cepat dengan cara menentukan strategi yang sesuai dalam menghadapi kondisi di masa yang akan datang.

### Objective
Membuat sebuah model yang dapat memprediksi setidaknya 60% dari pelanggan yang akan mengalami telat bayar kartu kredit (Accuracy dan Recall > 60%).

### Experiment
Periode tinjauan:

- Nasabah di-review selama satu tahun terakhir.
- Nasabah di-review selama 6 bulan terakhir.
  
Penyesuaian variabel:

- Balance dilihat dari rata-rata selama horizon waktu dan dilihat perubahan pada akhir tinjauan dan awal tinjauan.
- Melihat kepemilikan jumlah produk dari rata-rata, maksimum, dan minimum pada periode tinjauan.
Status keaktifan nasabah dilihat dalam bentuk bulan.

---

## Data Understanding
Sebelum membuat model, kita perlu memahami variabel apa saja yang tersedia dan dapat digunakan.

- Customer ID: unique ID customer.
- Branch: lokasi cabang nasabah terdaftar.
- City: lokasi kota nasabah terdaftar.
- Age: umur nasabah pada periode observasi.
- Avg. Annual Income: rata-rata penghasilan nasabah dalam satu tahun.
- Balance (Q1-Q4): saldo mengendap yang dimiliki nasabah di akhir kuartal.
- Num of Product (Q1-Q4): jumlah kepemilikan produk nasabah yang dimiliki di akhir kuartal.
- HasCrCard (Q1-Q4): status kepemilikan produk kartu kredit nasabah di akhir kuartal.
- Active Member (Q1-Q4): status keaktifan nasabah.
- Unpaid tagging: status nasabah gagal bayar.

---

## Data Preparation
Persiapan data yang dilakukan berupa pengubahan bentuk, pemisahan jumlah sample, maupun penambahan variabel tertentu.

Berikut adalah hal yang perlu dilakukan.

1. Pengecekan data terduplikasi dan data yang hilang.
2. Menambah variabel atau fitur baru (rata-rata, min, max).
3. Transformasi data dengan melakukan encoding data kategorikal maupun standardisasi untuk data numerik.
4. Mengulang tahapan yang sama untuk dataset validasi.
5. Pengecekan korelasi.
6. Pemisahan data train atau test.

---

## Modeling
Proses yang dilakukan pada tahap ini adalah:

1. pemilihan algoritma, misalnya logistic regression, random forest, dan gradient boosting.
2. pencarian hyperparameter terbaik. Kita lakukan tuning model dengan menggunakan GridSearchCV.
3. pembangunan model. Kita buat model machine learning meggunakan parameter.

---

## Evaluation

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/9dc5720f-951e-418b-b196-cb579dd6b7c5" />


1. Recall adalah parameter yang mengukur proporsi positif (gagal bayar) yang berhasil diidentifikasi.
  
  <img width="451" height="97" alt="image" src="https://github.com/user-attachments/assets/fad15143-1fa5-494b-952e-00c87cecf6b5" />

2. Precision adalah parameter yang mengukur seberapa banyak prediksi positif (gagal bayar) yang memang benar.

<img width="525" height="141" alt="image" src="https://github.com/user-attachments/assets/929a77ac-f7c5-4400-8771-fe9bb771a6b2" />

3.  Accuracy adalah parameter yang mengukur keberhasilan model.

<img width="677" height="121" alt="image" src="https://github.com/user-attachments/assets/2c3fa490-37e9-49b5-bcc9-04064eb864a3" />

---

## Deployment
Pada tahap ini, deployment yang dilakukan berupa coding dalam bentuk jupyter notebook.

---

## Hands-On Coding

### Import Package
Import package yang dibutuhkan untuk melakukan prediction.

```
!pip install jcopml

# Library untuk manipulasi data tabular
import pandas as pd

# Library untuk operasi numerik dan array multidimensi
import numpy as np

# Library untuk visualisasi data dasar
import matplotlib.pyplot as plt

# Library untuk visualisasi statistik yang lebih menarik
import seaborn as sns

# Library untuk pengukuran waktu (misalnya untuk profiling runtime)
import time

# Import model klasifikasi linear dan versi dengan cross-validation
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# Import metode pencarian parameter terbaik untuk tuning model
from sklearn.model_selection import GridSearchCV

# Import model klasifikasi berbasis gradient boosting yang powerful
from xgboost import XGBClassifier

# Import berbagai metrik evaluasi performa model klasifikasi
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Import model ensemble berbasis decision tree
from sklearn.ensemble import RandomForestClassifier

# Import modul metrik tambahan dari scikit-learn
from sklearn import metrics

# Import fungsi untuk analisis feature importance dari JCOPML
from jcopml.feature_importance import mean_score_decrease
```
```
# Menampilkan semua kolom dataframe saat dipanggil atau dicetak
pd.set_option('display.max_columns', None)
```

### Data For Prediction

```
# Path dataset utama untuk training dan eksplorasi data
path_1 = "https://raw.githubusercontent.com/yayankurniawan/Project-Data-Analysis-Prediction-Model/main/FinanKu%20Data%20All.csv"

# Path dataset untuk validasi model (data unseen)
path_2 = "https://raw.githubusercontent.com/yayankurniawan/Project-Data-Analysis-Prediction-Model/main/FinanKu%20Data%20Validasi.csv"

# Membaca dataset utama ke dalam dataframe df_all
df_all = pd.read_csv(path_1)

# Membaca dataset validasi ke dalam dataframe df_val
df_val = pd.read_csv(path_2)
```
