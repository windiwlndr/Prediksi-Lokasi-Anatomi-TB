import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import LocalOutlierFactor

# Judul aplikasi
st.title("Prediksi Lokasi Anatomi TB Menggunakan Naive Bayes")

# Sidebar untuk navigasi
st.sidebar.title("Kelompok 6")
options = st.sidebar.radio("Pilih Halaman:", ["Tampilan Data", "Analisis Data", "Modeling", "Prediksi Data Baru"])

# Memuat dataset
df = pd.read_excel('TB.xlsx')

# Mendefinisikan kolom kategori dan kolom target
kolom_kategori = ['JENIS KELAMIN', 'KECAMATAN', 'FOTO TORAKS', 'STATUS HIV', 'RIWAYAT DIABETES', 'HASIL TCM']
kolom_target = 'LOKASI ANATOMI (target/output)'

# Inisialisasi SimpleImputer dan LabelEncoder
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
label_encoders = {}

for col in kolom_kategori:
    le = LabelEncoder()
    df_imputed[col] = le.fit_transform(df_imputed[col])
    label_encoders[col] = le

le_target = LabelEncoder()
df_imputed[kolom_target] = le_target.fit_transform(df_imputed[kolom_target])
label_encoders[kolom_target] = le_target

# Menyimpan encoders dalam state session
if 'label_encoders' not in st.session_state:
    st.session_state['label_encoders'] = label_encoders
if 'le_target' not in st.session_state:
    st.session_state['le_target'] = le_target

if options == "Tampilan Data":
    st.subheader("Tampilan Data Awal")
    st.write(df.head())
    st.subheader("Statistik Data")
    st.write(df.describe())
    buffer = st.info("Menampilkan informasi data...")
    st.write(df.info())
    buffer.empty()

elif options == "Analisis Data":
    st.subheader("Distribusi Data")
    st.write("Menampilkan distribusi data menggunakan histogram, pairplot untuk analisis hubungan antar variabel, dan informasi tentang nilai yang hilang.")
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.set_title("Distribusi Data", fontsize=20, fontweight='bold')
    df.hist(bins=50, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Pairplot Data")
    sns.set(style="whitegrid")
    fig = sns.pairplot(df, diag_kind='kde', plot_kws={'alpha':0.6, 's':80, 'edgecolor':'k'}, diag_kws={'shade': True})
    st.pyplot(fig)

    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)

elif options == "Modeling":
    st.subheader("Mendeteksi Outlier")

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    outliers = lof.fit_predict(df_imputed)
    df_imputed['outlier'] = outliers

    jumlah_outlier = (outliers == -1).sum()
    st.write(f"Jumlah outlier: {jumlah_outlier}")

    data_outliers = df_imputed[df_imputed['outlier'] == -1]
    st.write("Data outliers:\n", data_outliers)

    df_imputed = df_imputed.drop(columns=['outlier'])
    df_cleaned = df_imputed[outliers != -1]

    st.subheader("Boxplot Data")
    st.write("Menampilkan boxplot untuk visualisasi distribusi data yang telah dibersihkan dari outlier. Boxplot membantu dalam memahami rentang distribusi dan mendeteksi outlier baru jika ada.")
    fig, ax = plt.subplots(figsize=(20, 15))
    df_cleaned.plot(kind='box', subplots=True, layout=(5, 5), ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    for column in df_cleaned.columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mean_value = df_cleaned[column].mean()
        df_cleaned[column] = np.where((df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound), mean_value, df_cleaned[column])

    X = df_cleaned.drop(kolom_target, axis=1)
    y = df_cleaned[kolom_target]
    y = le_target.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("Data Training dan Testing")
    st.write("X_train:\n", X_train.head())
    st.write("X_test:\n", X_test.head())
    st.write("y_train:\n", y_train[:5])
    st.write("y_test:\n", y_test[:5])

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    st.subheader("Evaluasi Model")
    st.write("Akurasi:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    st.write("Classification Report:\n", classification_report(y_test, y_pred))
    
    st.write("Precision: Menunjukkan seberapa banyak prediksi positif model yang benar. Precision yang tinggi berarti lebih sedikit false positives.")
    st.write("Recall: Menunjukkan seberapa banyak kasus positif sebenarnya yang berhasil ditemukan oleh model. Recall yang tinggi berarti lebih sedikit false negatives.")
    st.write("F1-score: Harmonic mean dari precision dan recall. Berguna saat Anda membutuhkan keseimbangan antara precision dan recall.")
    st.write("Support: Jumlah kasus sebenarnya dari setiap kelas yang digunakan untuk evaluasi.")

    # Menyimpan model untuk digunakan nanti
    st.session_state['model'] = gnb

elif options == "Prediksi Data Baru":
    st.header("Input Data Baru")
    umur = st.number_input("Umur", min_value=0, max_value=120, value=30)
    jenis_kelamin = st.selectbox("Jenis Kelamin", [x for x in df['JENIS KELAMIN'].unique() if pd.notna(x)])
    kecamatan = st.selectbox("Kecamatan", [x for x in df['KECAMATAN'].unique() if pd.notna(x)])
    foto_toraks = st.selectbox("Foto Toraks", [x for x in df['FOTO TORAKS'].unique() if pd.notna(x)])
    status_hiv = st.selectbox("Status HIV", [x for x in df['STATUS HIV'].unique() if pd.notna(x)])
    riwayat_diabetes = st.selectbox("Riwayat Diabetes", [x for x in df['RIWAYAT DIABETES'].unique() if pd.notna(x)])
    hasil_tcm = st.selectbox("Hasil TCM", [x for x in df['HASIL TCM'].unique() if pd.notna(x)], key='hasil_tcm', help="Pilih hasil TCM yang valid", index=0)

    new_data = {
        'UMUR': [umur],
        'JENIS KELAMIN': [jenis_kelamin],
        'KECAMATAN': [kecamatan],
        'FOTO TORAKS': [foto_toraks],
        'STATUS HIV': [status_hiv],
        'RIWAYAT DIABETES': [riwayat_diabetes],
        'HASIL TCM': [hasil_tcm],
    }

    df_new = pd.DataFrame(new_data)

    # Memuat label encoders dan model dari session state
    if 'label_encoders' in st.session_state and 'le_target' in st.session_state:
        label_encoders = st.session_state['label_encoders']
        le_target = st.session_state['le_target']

        for col in kolom_kategori:
            if col in df_new.columns:
                le = label_encoders[col]
                df_new[col] = le.transform(df_new[col])

        # Memuat model dari session state
        if 'model' in st.session_state:
            gnb = st.session_state['model']
            new_predictions = gnb.predict(df_new)

            results_new = df_new.copy()
            results_new['Prediksi'] = new_predictions
            results_new['Prediksi'] = le_target.inverse_transform(results_new['Prediksi'])

            st.subheader("Hasil Prediksi")
            st.write(results_new[['UMUR', 'JENIS KELAMIN', 'KECAMATAN', 'FOTO TORAKS', 'STATUS HIV', 'RIWAYAT DIABETES', 'HASIL TCM', 'Prediksi']])
        else:
            st.write("Model belum dilatih. Silakan latih model terlebih dahulu di bagian 'Modeling'.")
    else:
        st.write("Label encoders belum tersedia. Silakan latih model terlebih dahulu di bagian 'Modeling'.")
