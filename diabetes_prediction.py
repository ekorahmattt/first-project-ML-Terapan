# %% [markdown]
# # Proyek Pertama Machine Learning Terapan - Dicoding
# 
# - Nama          : Eko Rahmat Darmawan
# - Email         : erdarmawan7@gmail.com
# - ID Dicoding   : echo_ramled
# 
# ## Prediksi Diabetes (Diabetes Prediction)
# 
# Prediksi diabetes merupakan proyek untuk memprediksi data medis seseorang terindikasi terkena diabetes atau tidak. Dataset yang digunakan menggunakan data public dari situs kaggle yang bisa diunduh <a href='https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset'>disini</a>.
# 
# Dataset berupa data medis tabular pasien dengan status diabetes (positif dan negatif). Data berisi dengan 9 kolom sebagai fitur mulai dari:
# - Usia (age)
# - Jenis kelamin (gender) 
# - Riwayat hipertensi (hypertension)
# - Riwayat penyakit jantung (heart_disease)
# - Riwayat Merokok (smoking_history)
# - Indeks Berat Badan / Body Mass Index (BMI)
# - HbA1c Level
# - Level Gula Darah (blood_glucose_level)
# - Diabetes

# %% [markdown]
# ## Load Dataset

# %%
# Import Library yang dibutuhkan

# Library untuk load dataset
import pandas as pd

# Library untuk data visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# Library untuk data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Library untuk model machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Library untuk evaluasi model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %%
# Load dataset kedalam dataframe
df = pd.read_csv('diabetes_prediction_dataset.csv')
df.head()

# %%
# Menampilkan informasi dataset
df.info()

# %%
# Menampilkan dimensi baris dan kolom dataframe
df.shape

# %% [markdown]
# ## Data Cleaning
# 
# Merupakan proses untuk membersihkan dataset dari nilai kosong, data duplikat, dan lain-lain yang dapat mempengaruhi machine learning dalam mempelajari data

# %%
# Menampilkan jumlah data yang bernilai kosong setiap kolomnya
df.isnull().sum()

# %%
# Menampilkan jumlah data yang duplikat
df.duplicated().sum()

# %%
# Menghapus data duplikat
df = df.drop_duplicates()

# %% [markdown]
# ## Exploratory Data Analyst (EDA)
# 
# Proses untuk mempelajari karakteristik data lebih dalam. Setiap fitur atau kolom memiliki karakter yang berbeda dengan tipe data yang berbeda juga, proses ini penting untuk menentukan langkah preprocessing atau langkah berikutnya agar data dapat dipelajari algoritma machine learning dengan baik.

# %%
# Menampilkan nama kolom dataframe
df.columns

# %% [markdown]
# ### Jenis Kelamin (Gender)

# %%
# Menghitung kategorikal value
df['gender'].value_counts()

# %%
# Menghapus kategori Other
df.drop(df.index[df['gender'] == 'Other'], inplace=True)

print("Data gender setelah menghapus kategori Other")
df['gender'].value_counts()

# %%
# Visualisasi Data Gender
fig, ax = plt.subplots(1, 2)

# Diagaram Batang merepresentasikan jumlah
sns.countplot(x='gender', data=df, ax=ax[0])
ax[0].set_title("Jumlah")

# Diagram lingkaran merepresentasikan persentase pada dataset
ax[1].pie(df['gender'].value_counts(), labels=df['gender'].value_counts().index, autopct='%1.1f%%')
ax[1].set_title("Persentase")

plt.suptitle("Distribusi Jenis Kelamin")
plt.show()

# %%
#Visualisasi Diabetes pada Gender

sns.countplot(x='gender', hue='diabetes', data=df)
plt.title("Distribusi Diabetes Berdasarkan Jenis Kelamin")
plt.show()

# %% [markdown]
# ### Usia (Age)

# %%
# Mengetahui rata-rata usia pada data
dia_age = df[df['diabetes'] == 1]

print("Rata-rata usia pasien diabetes : ", round(dia_age['age'].mean(), 0), "tahun")
print("Rata-rata usia pasien pada dataset : ",round(df['age'].mean(), 0), "tahun")

# %%
# Visualisasi Distribusi Data

sns.boxplot(x='diabetes', y='age', data=df)
plt.title("Distribusi Diabetes Berdasarkan Usia")
plt.show()

# %% [markdown]
# ### Hipertensi (Hypertension)

# %%
# Menghitung riwayat hipertensi
df['hypertension'].value_counts()

# %%
# Visualisasi Data Distribusi Riwayat Hipertensi
fig, ax = plt.subplots(1, 2)

# Diagram Batang merepresentasikan jumlah
sns.countplot(x='hypertension', data=df, ax=ax[0])
ax[0].set_title("Jumlah")

# Diagram Lingkaran merepresentasikan persentase
ax[1].pie(df['hypertension'].value_counts(), labels=df['hypertension'].value_counts().index, autopct='%1.1f%%')
ax[1].set_title("Persentase")

plt.suptitle("Distribusi Hipertensi")
plt.legend(['No-Hipertensi','Hipertensi'], loc='upper right')
plt.show()

# %% [markdown]
# ### Riwayat Sakit Jantung (Heart_Disease)

# %%
# Menghitung distribusi kolom riwayat sakit jantung
df['heart_disease'].value_counts()

# %%
# Visualisasi Distribusi Riwayat Sakit Jantung
fig, ax = plt.subplots(1, 2)

# Diagram Batang merepresentasikan jumlah
sns.countplot(x='heart_disease', data=df, ax=ax[0])
ax[0].set_title("Jumlah")

# Diagram lingkaran merepresentasikan persentase
ax[1].pie(df['heart_disease'].value_counts(), labels=df['heart_disease'].value_counts().index, autopct='%1.1f%%')
ax[1].set_title("Persentase")

plt.suptitle("Distribusi Riwayat Jantung")
plt.legend(['Jantung Sehat','Sakit Jantung'], loc='upper right')
plt.show()

# %% [markdown]
# ### Riwayat Merokok (Smoking_History)

# %%
# Menghitung distribusi riwayat merokok
df['smoking_history'].value_counts()

# %% [markdown]
# ### Riwayat Merokok

# %%
# Visualisasi distribusi riwayat merokok
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Diagram Batang merepresentasikan jumlah
sns.countplot(x='smoking_history', data=df, ax=ax[0])
ax[0].set_title("Jumlah")

# Diagram lingkaran merpresentasikan persentase
ax[1].pie(df['smoking_history'].value_counts(), labels=df['smoking_history'].value_counts().index, autopct='%1.1f%%')
ax[1].set_title("Persentase")

plt.suptitle("Distribusi Riwayat Merokok")
plt.show()

# %%
# Visualisasi Distribusi Diabetes Berdasarkan Riwayat Merokok
sns.countplot(x='smoking_history', hue='diabetes', data=df)
plt.title("Distribusi Diabetes Berdasarkan Riwayat Merokok")
plt.show()

# %% [markdown]
# ### Indeks Berat Badan / Body Mass Index (BMI)

# %%
# Visualiasi distribusi berat badan
sns.displot(df['bmi'], bins=30)
plt.title("Distribusi Indeks Berat Badan")
plt.show()

# %%
# Visualisasi Distribusi diabetes berdasarkan BMI
sns.boxplot(x='diabetes', y='bmi', data=df)
plt.title("Distribusi Diabetes Berdasarkan Indeks Berat Badan")
plt.show()

# %% [markdown]
# ## Data Preprocessing
# 
# Proses menyiapkan data agar dapat digunakan pada proses training.

# %%
# Data Encoding untuk mengubah data kategorikal menjadi numerik
encoder = LabelEncoder()

# Mentransform nilai pada kolom gender dan smoking history menjadi numerikal
df['gender'] = encoder.fit_transform(df['gender'])
df['smoking_history'] = encoder.fit_transform(df['smoking_history'])
df.head()

# %%
# Memisahkan Fitur dan Label

# Fitur sebagai x semua kolom kecuali kolom diabetes
x = df.drop(columns=['diabetes'], axis=1)

# Label sebagai y
y = df['diabetes']

# %%
# Membagi data menjadi data training untuk melatih machine learning dan data testing untuk evaluasi model
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
x_train.shape

# %% [markdown]
# ## Training
# 
# Proses training untuk melatih machine learning dalam mempelajari dataset yang di inputkan. Dalam proyek ini, machine learning menggunakan 2 algoritma sebagai perbandingan performa, yaitu:
# - Logistic Regression
# - Random Forest Classifier

# %% [markdown]
# ### Logistic Regression

# %%
# Inisiasi model Logistic Regression
lr = LogisticRegression(max_iter=3000)

# Proses Training
lr.fit(x_train, y_train)

# %%
# Uji prediksi menggunakan data testing
y_pred = lr.predict(x_test)

# Menghitung tingkat akurasi model
lr_acc = accuracy_score(y_test, y_pred)
print("Akurasi Model : ", round(lr_acc, 2) * 100, "%")

# %%
# Menampilkan detail performa model Logistic Regression
lr_class = classification_report(y_test, y_pred)
print(lr_class)

# %%
# Menampilkan hasil rediksi dalam bentuk confusion matrix
lr_conf = confusion_matrix(y_test, y_pred)

sns.heatmap(lr_conf, annot=True, cmap='Blues', fmt='g')
plt.title("Logistic Regression")
plt.show()

# %% [markdown]
# ### Random Forest Classifier

# %%
# Inisiasi model Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Proses Training
rf.fit(x_train, y_train)

# %%
# Uji prediksi menggunakan data testing
y_pred = rf.predict(x_test)

# Menghitung tingkat akurasi model
rf_acc = accuracy_score(y_test, y_pred)
print("Akurasi Model : ", round(rf_acc, 2) * 100, "%")

# %%
# Menampilkan detail performa model Logistic Regression
rf_class = classification_report(y_test, y_pred)
print(rf_class)

# %%
# Menampilkan hasil rediksi dalam bentuk confusion matrix
rf_conf = confusion_matrix(y_test, y_pred)

sns.heatmap(rf_conf, annot=True, cmap='Blues', fmt='g')
plt.title("Random Forest Classifier")
plt.show()

# %% [markdown]
# ## Hasil

# %% [markdown]
# Berdasarkan hasil analisis dan evaluasi model, dari kedua algoritma machine learning yang digunakan pada proyek ini. Algoritma Random Forest Classifier memiliki tingkat akurasi lebih tinggi dibanding algoritma Logistic Regression yaitu sekitar 97%.


