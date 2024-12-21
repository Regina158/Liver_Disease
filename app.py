from flask import Flask, request, render_template
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import csv
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Fungsi untuk klasifikasi menggunakan Naive Bayes


def classify_naive_bayes(data):
    try:
        df = pd.read_csv("uploads/Data_Training.csv")  # Pastikan file ini ada
        X = df.drop('Label', axis=1)
        y = df['Label']

        # Pembagian data untuk pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Standarisasi fitur
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model Naive Bayes
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Prediksi hasil
        prediction = model.predict([data])
        if prediction[0] == 1:
            return "Penyakit Hati"
        elif prediction[0] == 2:
            return "Tidak Ada Penyakit Hati"
        return "Label Tidak Dikenal"
    except Exception as e:
        return f"Error: {str(e)}"

# Fungsi untuk klasifikasi menggunakan RBFNN


def classify_rbf_nn(data):
    try:
        df = pd.read_csv("uploads/Data_Training.csv")  # Pastikan file ini ada
        X = df.drop('Label', axis=1)
        y = df['Label']

        # Pembagian data untuk pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Standarisasi fitur
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model RBFNN
        model = MLPClassifier(hidden_layer_sizes=(
            10,), activation='logistic', solver='lbfgs', random_state=42)
        model.fit(X_train, y_train)

        # Prediksi hasil
        prediction = model.predict([data])
        if prediction[0] == 1:
            return "Penyakit Hati"
        elif prediction[0] == 2:
            return "Tidak Ada Penyakit Hati"
        return "Label Tidak Dikenal"
    except Exception as e:
        return f"Error: {str(e)}"

# Fungsi untuk menyimpan riwayat dalam file CSV terpisah berdasarkan model yang dipilih


def save_to_csv(data, model_choice):
    if model_choice == 'naive_bayes':
        csv_filename = 'naive_bayes_history.csv'
    elif model_choice == 'rbfnn':
        csv_filename = 'rbfnn_history.csv'
    else:
        return

    file_exists = os.path.isfile(csv_filename)

    # Jika file CSV belum ada, buat file dan tambahkan header
    if not file_exists:
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Age', 'Gender', 'Bilirubin',
                            'Alkphos', 'SGPT', 'SGOT', 'Albumin', 'Label'])

    # Menambahkan data baru ke dalam file CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Fungsi untuk mengonversi gender dari 1, 2 ke Male, Female


def convert_gender(gender_num):
    if gender_num == 1:
        return 'Male'
    elif gender_num == 2:
        return 'Female'
    return 'Unknown'

# Route untuk form input


@app.route('/', methods=['GET', 'POST'])
def index():
    classification = None
    history = None
    if request.method == 'POST':
        try:
            # Ambil data dari form
            age = float(request.form['age'])
            gender = int(request.form['gender'])
            bilirubin = float(request.form['bilirubin'])
            alkphos = float(request.form['alkphos'])
            sgpt = float(request.form['sgpt'])
            sgot = float(request.form['sgot'])
            albumin = float(request.form['albumin'])
            model_choice = request.form['model_choice']

            # Data input
            data = [age, gender, bilirubin, alkphos, sgpt, sgot, albumin]

            # Pilihan metode klasifikasi
            if model_choice == 'naive_bayes':
                classification = classify_naive_bayes(data)
            elif model_choice == 'rbfnn':
                classification = classify_rbf_nn(data)

            # Menambahkan hasil ke riwayat, konversi gender ke teks (Male, Female)
            gender_text = convert_gender(gender)
            history = [age, gender_text, bilirubin, alkphos,
                       sgpt, sgot, albumin, classification]

            # Simpan riwayat ke CSV berdasarkan model yang dipilih
            save_to_csv(history, model_choice)

        except ValueError as e:
            classification = f"Error: Invalid input. {str(e)}"

    return render_template('index.html', classification=classification, history=history)


if __name__ == "__main__":
    app.run(debug=True)
