import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np

with open('final_model.pkl','rb') as file:
    Final_Model = pickle.load(file)

def main():
    # stc.html(html_temp)
    # st.title("Customer Churn Prediction App")
    st.markdown("""
            <p style="font-size: 44px; color: #023047;font-weight: bold">Customer Churn Prediction App</p>
            """, unsafe_allow_html=True)
    st.markdown("Aplikasi ini dibuat oleh tim Algowizard untuk Final Project Data Science Bootcamp Digital Skola")

    with st.sidebar:
        st.image("algowizard.jpg")

        menu = ["Overview","Machine Learning"]
        choice = st.sidebar.selectbox("Menu", menu)


    if choice == "Overview":
        st.header("Overview")
        st.markdown("Aplikasi prediksi churn memanfaatkan pembelajaran mesin dan kecerdasan buatan untuk menganalisis data pelanggan dan mengidentifikasi mereka yang berisiko pergi. Hal ini memungkinkan bisnis untuk secara proaktif melibatkan pelanggan ini dengan intervensi yang ditargetkan dan strategi retensi, meminimalkan churn dan meningkatkan nilai umur pelanggan.")

        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Sekilas tentang Dataset yang digunakan</p>
            """, unsafe_allow_html=True)

        url = "https://raw.githubusercontent.com/nuraulaola/ds-batch-32-final-project/main/datasets/Dataset1_Customer_Churn.csv"
        df = pd.read_csv(url)
        top_10_rows = df.head(10)
        st.table(top_10_rows)

        text1 = """
                Cerita Dataset:

                - Dataset ini terdiri dari total 10.000 baris (entries) dengan rentang indeks baris dari 0 hingga 9999, dan terdapat 7 kolom.
                - Variabel independen melibatkan Gender, Age, CreditScore, EstimatedSalary, dan HasCrCard, yang berisi informasi tentang pelanggan.
                - Variabel dependen adalah Exited, yang menunjukkan apakah pelanggan tersebut telah meninggalkan layanan.
                - CustomerId merupakan data integer

                Fitur:

                - CustomerId: ID unik untuk setiap pelanggan.
                - Gender: Jenis kelamin pelanggan (Female / Male).
                - Age: Usia pelanggan.
                - CreditScore: Skor kredit pelanggan.
                - EstimatedSalary: Perkiraan gaji pelanggan.
                - HasCrCard: Menunjukkan apakah pelanggan memiliki kartu kredit (1 untuk ya, 0 untuk tidak).
                - Exited: Variabel target yang menunjukkan apakah pelanggan telah keluar dari layanan (1 untuk ya, 0 untuk tidak).
                """
        
        text2 = """
                1. **Distribusi variabel dependen 'Exited' terhadap variabel independen 'Gender':**
                - Female memiliki jumlah yang lebih tinggi pada kategori '1' (Exited), yang sesuai dengan rendahnya nilai KDE pada kategori tertentu.
                - Male memiliki jumlah yang lebih tinggi pada kategori '0' (Not Exited), yang sesuai dengan tingginya nilai KDE pada kategori tertentu.

                2. **Distribusi variabel dependen 'Exited' terhadap variabel independen 'Age':**
                - Mayoritas pelanggan berada dalam kelompok usia 30 - 40 dan 40 - 50.
                - Kelompok usia 70 - 80 dan 80 - 90 memiliki jumlah pelanggan yang lebih sedikit.

                3. **Distribusi variabel dependen 'Exited' terhadap variabel independen 'CreditScore':**
                - Bentuk distribusi normal menunjukkan bahwa sebagian besar nasabah memiliki CreditScore yang berpusat di sekitar mean.
                - Namun, lonjakan di atas sumbu x pada nilai 800 mengindikasikan adanya kelompok kecil tetapi signifikan dari nasabah dengan CreditScore sangat tinggi.

                4. **Distribusi variabel dependen 'Exited' terhadap variabel independen 'EstimatedSalary':**
                - Distribusi seragam EstimatedSalary menunjukkan bahwa estimasi pendapatan nasabah cenderung stabil dan tidak mengalami variasi yang signifikan.
                - Meskipun demikian, tidak ada pola khusus atau tren yang terlihat dalam hubungannya dengan tingkat churn. Hal ini mengindikasikan bahwa faktor-faktor lain di luar estimasi pendapatan mungkin lebih berperan dalam keputusan nasabah untuk bertahan atau keluar.

                5. **Distribusi variabel dependen 'Exited' terhadap variabel independen 'HasCrCard':**
                - Terdapat perbedaan yang signifikan antara pemegang kartu kredit (HasCrCard=1) dan bukan pemegang kartu kredit (HasCrCard=0) dalam hal jumlah Exited.
                - Pemegang kartu kredit (HasCrCard=1) memiliki jumlah Exited yang lebih tinggi dibandingkan dengan yang bukan pemegang kartu kredit.
                """
        text3 = """
                1. **Gender:**
                - Sebelum SMOTE, Female memiliki jumlah yang lebih tinggi pada kategori '1' (Exited) dibandingkan dengan Male.
                - Setelah SMOTE, perbedaan jumlah antara Female dan Male pada kategori '1' (Exited) masih lebih tinggi.

                2. **Age:**
                - Sebelum SMOTE, kategori umur '30 - 50' memiliki jumlah Exited yang tinggi.
                - Setelah SMOTE, terdapat peningkatan jumlah Exited pada kategori '30 - 50'.

                3. **CreditScore:**
                - Sebelum SMOTE, skor kredit di bawah rata-rata memiliki jumlah Exited yang lebih rendah dibandingkan dengan skor kredit di atas rata-rata.
                - Setelah SMOTE, terjadi perbedaan yang sangat signifikan pada 'Exited = 1' sedangkan yang 'Exited = 0' tetap ada perubahan namun tidak signifikan.

                4. **EstimatedSalary:**
                - Sebelum SMOTE, kategori pendapatan di bawah rata-rata dan di atas rata-rata memiliki jumlah Exited yang cukup seimbang.
                - Setelah SMOTE, terjadi perbedaan yang sangat signifikan pada 'Exited = 1' sedangkan yang 'Exited = 0' tetap ada perubahan namun tidak signifikan.

                5. **HasCrCard:**
                - Sebelum SMOTE, terdapat perbedaan yang signifikan antara pemegang kartu kredit (HasCrCard=1) dan bukan pemegang kartu kredit (HasCrCard=0) dalam hal jumlah Exited.
                - Setelah SMOTE, terjadi perbedaan yang sangat signifikan pada 'Exited = 1' sedangkan yang 'Exited = 0' tetap ada perubahan namun tidak signifikan.

                SMOTE berhasil menyeimbangkan jumlah sampel antara kelas Exited (1) dan kelas Not Exited (0), mengurangi ketidakseimbangan kelas yang dapat memengaruhi kinerja model klasifikasi.
                """
        
        text4 = """
                1. **Korelasi antara Umur (Age) dan Exited:**
                - Korelasi positif menunjukkan bahwa ada hubungan yang moderat antara usia nasabah dan kecenderungan untuk keluar dari layanan.
                - Ini dapat diartikan bahwa semakin tua seseorang, semakin cenderung mereka bertahan dalam layanan.

                2. **Korelasi antara Jenis Kelamin (Gender) dan Exited:**
                - Korelasi negatif menunjukkan bahwa terdapat hubungan cukup negatif antara jenis kelamin (laki-laki) dan kecenderungan untuk keluar dari layanan.
                - Hal ini dapat diartikan bahwa nasabah perempuan mungkin cenderung lebih loyal terhadap layanan dibandingkan dengan nasabah laki-laki.

                3. **Korelasi antara Kepemilikan Kartu Kredit (HasCrCard) dan Exited:**
                - Korelasi negatif menunjukkan bahwa kepemilikan kartu kredit memiliki pengaruh cukup negatif terhadap kecenderungan keluar dari layanan.
                - Artinya, nasabah yang memiliki kartu kredit cenderung lebih setia terhadap layanan.

                4. **Korelasi antara Skor Kredit (CreditScore) dan Exited:**
                - Korelasi negatif menunjukkan bahwa terdapat hubungan yang kurang kuat antara skor kredit dan kecenderungan keluar dari layanan.
                - Hal ini mungkin menandakan bahwa nasabah dengan skor kredit yang lebih tinggi memiliki kecenderungan yang sedikit lebih rendah untuk keluar dari layanan.

                5. **Korelasi antara Estimasi Pendapatan (EstimatedSalary) dan Exited:**
                - Korelasi positif yang sangat lemah menunjukkan bahwa tidak ada korelasi yang signifikan antara estimasi pendapatan dan kecenderungan keluar dari layanan.
                - Dengan kata lain, estimasi pendapatan tidak menjadi faktor utama yang mempengaruhi keputusan nasabah untuk keluar dari layanan.

                5. **Korelasi antara Kepemilikan Kartu Kredit (HasCrCard) dan Jenis Kelamin (Gender):**
                - Korelasi positif menunjukkan bahwa ada hubungan positif yang kurang kuat antara kepemilikan kartu kredit dan jenis kelamin laki-laki.
                - Artinya, laki-laki mungkin sedikit lebih mungkin memiliki kartu kredit.
                """

        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Deskripsi Dataset</p>
            """, unsafe_allow_html=True)
        st.markdown(text1)
        st.image("output1.png")
        st.markdown(text2)
        st.image("output3.png")
        st.markdown(text3)
        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Mengatasi Imbalance Dataset</p>
            """, unsafe_allow_html=True)
        st.image("output2.png")
        st.markdown(text4)

    elif choice == "Machine Learning":
        st.header("Prediction Model")
        run_ml_app()

def run_ml_app():

    st.markdown("""
    <p style="font-size: 16px; font-weight: bold">Insert Data</p>
    """, unsafe_allow_html=True)

    left, right = st.columns((2,2))
    gender = left.selectbox('Gender',
                            ('Male', 'Female'))
    age = left.number_input('Age', 1, 100)
    credit_score = left.number_input('Credit Score',0,1000)
    estimated_salary = right.number_input('Estimated Salary',0.0,100000000.00)
    has_credit_card = right.selectbox('Credit Card',('Yes','No'))

    button = st.button('Predict')

    #if button is clicked (ketika button dipencet)
    if button:
        #make prediction
        result = predict(gender,age,credit_score,estimated_salary,has_credit_card)
        if result == 'Eligible':
            st.success(f'You have {result} from the loan')
        else:
            st.warning(f'You have {result} for the loan')

def predict(gender,age,credit_score,estimated_salary,has_credit_card):
    #processing user input
    gen = 0 if gender == 'Male' else 1
    cre = 0 if has_credit_card == 'No' else 1

    #Making prediction
    prediction = Final_Model.predict([[gen, cre, age, credit_score,estimated_salary]])
    result = 'Stayed' if prediction == 0 else 'Exited'

    return result

if __name__ == "__main__":
    main()