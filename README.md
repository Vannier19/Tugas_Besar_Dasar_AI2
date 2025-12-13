# Deteksi Fraud Transaksi dengan Machine Learning

## Kelompok 33

Repositori ini berisi implementasi model machine learning untuk mendeteksi transaksi fraud dari dataset sintetis yang mirip kondisi real-world. Proyek ini merupakan bagian dari Tugas Besar 2 mata kuliah Dasar AI.

---

## Deskripsi Singkat

Kasus fraud dalam transaksi keuangan itu challenging banget karena:
- Data yang sangat tidak seimbang (transaksi normal jauh lebih banyak dari fraud)
- Banyak missing values dan outliers
- Interaksi antar fitur yang kompleks
- Harus akurat tapi juga cepat dalam deteksi

Di proyek ini, kami mencoba tiga algoritma klasik: **Decision Tree**, **Logistic Regression**, dan **KNN**, untuk melihat mana yang paling efektif dalam menangani masalah fraud detection ini.

---

## Dataset

Dataset yang digunakan adalah data sintetis dari kompetisi Kaggle dengan karakteristik:
- **Ukuran**: ~100,000 baris transaksi
- **Fitur**: Informasi transaksi (amount, merchant, location, device, user behavior, dll.)
- **Target**: `is_fraud` (binary: 0 = normal, 1 = fraud)
- **Karakteristik**: Imbalanced, ada missing values, outliers, dan perilaku mencurigakan yang tersembunyi

Dataset terdiri dari:
- `train.csv` - Data untuk training dan validasi
- `test.csv` - Data untuk prediksi submission
- `submission.csv` - Format output prediksi untuk Kaggle

---

## Struktur Proyek

```
├── src/
│   ├── Kelompok_33_Notebook.ipynb    # Notebook utama (EDA, preprocessing, modeling)
│   ├── train.csv                     # Dataset training
│   ├── test.csv                      # Dataset test
│   └── submission.csv                # File submission untuk Kaggle
├── doc/                              # Folder untuk laporan
├── .gitignore                        # File gitignore
└── README.md                         # File ini
```

---

## Tahapan Pengerjaan

### 1. **Exploratory Data Analysis (EDA)**
Mulai dengan memahami data: distribusi fitur, korelasi, pola missing values, dan identifikasi outliers. Di sini kami nemuin beberapa insight penting tentang karakteristik transaksi fraud vs normal.

### 2. **Data Cleaning**
Handle missing values dengan strategi yang berbeda untuk numerik (median) dan kategorikal (modus atau "Unknown"). Juga menghapus kolom yang tidak relevan seperti ID yang cuma identifier.

### 3. **Preprocessing Pipeline**
Tahap ini cukup kompleks karena harus handle berbagai jenis transformasi:

- **Feature Encoding**: 
  - One-hot encoding untuk fitur kategorikal dengan sedikit kategori (gender, device_type, dll.)
  - Target encoding untuk fitur dengan banyak kategori (merchant_category, country, dll.) supaya tidak dimensi explosion
  
- **Feature Scaling**: 
  - Log transformation untuk fitur yang sangat skewed (transaction_amount, distance_from_home)
  - RobustScaler untuk scaling numerik karena tahan terhadap outliers

- **Handling Imbalanced Data**: 
  - Pakai SMOTE untuk meng-generate sampel sintetis kelas fraud supaya seimbang
  - **Penting**: SMOTE cuma diterapkan di training set, biar validasi tetap objektif

- **Dimensionality Reduction**: 
  - PCA dengan threshold 95% variance untuk mengurangi dimensi setelah encoding
  - Membantu model lebih cepat training dan mengurangi overfitting

### 4. **Feature Selection**
Identifikasi fitur yang paling berpengaruh terhadap prediksi fraud. Beberapa fitur ternyata lebih informatif dibanding yang lain.

### 5. **Modeling & Validation**
Implementasi dan evaluasi tiga algoritma:

- **Decision Tree Learning (DTL)**
  - Mudah diinterpretasi, bisa visualisasi decision path
  - Rentan overfit kalau tidak di-pruning dengan baik
  
- **Logistic Regression**
  - Simple tapi powerful untuk binary classification
  - Cepat training dan prediksi
  
- **K-Nearest Neighbors (KNN)**
  - Non-parametric, cocok untuk pattern yang kompleks
  - Komputasi agak berat karena harus hitung distance ke semua tetangga

Validasi dilakukan dengan train-test split dan evaluasi pakai metrics yang sesuai untuk imbalanced data (precision, recall, F1-score, ROC-AUC).

---

## Cara Menjalankan

1. **Clone repository ini**
   ```bash
   git clone <repository-url>
   cd Tugas_Besar_2
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
   ```

3. **Download dataset dari Kaggle**
   - Letakkan `train.csv` dan `test.csv` di folder `src/`

4. **Run notebook**
   ```bash
   cd src
   jupyter notebook Kelompok_33_Notebook.ipynb
   ```
   Atau buka langsung di VS Code dengan Jupyter extension.

5. **Generate submission**
   Jalankan semua cell hingga section "Submission" untuk generate file `submission.csv` di folder `src/` yang bisa di-upload ke Kaggle.

---

## Hasil dan Insight

Kami membandingkan performa ketiga algoritma berdasarkan beberapa metrics:
- **Accuracy**: Bisa menyesatkan karena data imbalanced
- **Precision & Recall**: Lebih penting untuk kasus fraud
- **F1-Score**: Harmonic mean dari precision & recall
- **ROC-AUC**: Mengukur kemampuan model membedakan kelas

*(Detail hasil ada di notebook dan laporan)*

Insight menarik yang kami dapet:
- Algoritma dengan implementasi dari scratch vs library scikit-learn punya hasil yang cukup mirip, validasi bahwa implementasi kami benar
- Trade-off antara precision dan recall jadi pertimbangan penting dalam fraud detection
- Feature engineering dan preprocessing ternyata lebih berpengaruh daripada pemilihan algoritma

---

## Referensi

1. Scikit-learn Documentation: https://scikit-learn.org/
2. Imbalanced-learn (SMOTE): https://imbalanced-learn.org/
3. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"

---

## Catatan

- Pipeline preprocessing harus konsisten antara training dan inference untuk menghindari data leakage
- Jangan lupa set random_state yang sama di semua tahap supaya reproducible
- Kalau mau eksperimen dengan hyperparameter tuning, bisa pakai GridSearchCV atau RandomizedSearchCV

---

## Lisensi

Proyek ini dibuat untuk keperluan akademis. Dilarang menjiplak untuk tugas yang sama di semester/kampus lain.

---

**Happy Coding! 🚀**

*Last updated: Desember 2025*
