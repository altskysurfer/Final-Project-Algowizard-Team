import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np

with open('final_model.pkl','rb') as file:
    Final_Model = pickle.load(file)

def main():
    # stc.html(html_temp)
    st.title("Customer Churn Prediction App")
    st.caption("This app is created by Algowizard Team for Final Project of Data Science Bootcamp")

    menu = ["Home","Machine Learning"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.header("Home")
        st.caption("Aplikasi prediksi churn memanfaatkan pembelajaran mesin dan kecerdasan buatan untuk menganalisis data pelanggan dan mengidentifikasi mereka yang berisiko pergi. Hal ini memungkinkan bisnis untuk secara proaktif melibatkan pelanggan ini dengan intervensi yang ditargetkan dan strategi retensi, meminimalkan churn dan meningkatkan nilai umur pelanggan.")

        st.markdown("""
            <p style="font-size: 16px; font-weight: bold">Sekilas tentang Dataset yang digunakan</p>
            """, unsafe_allow_html=True)

        df = pd.DataFrame(np.random.randn(10, 5), columns=("col %d" % i for i in range(5)))
        st.table(df)


    elif choice == "Machine Learning":
        st.header("Prediction Model")
        run_ml_app()

    col1, col2, col3 = st.columns([1, 10, 1])  # Center column takes up most of the width
    with col2:
        images = ["1. Ola.png", "2. July.png", "3. Faza.png","4. Timmy.png",
              "5. Kemas.png", "6. Eko.png", "7. Osha.png"]
        st.image(images, width=80)  # Set width for each image

def run_ml_app():
    # design = """<div style='padding:15px;">
    #                 <h1 style='color:#fff'>Loan Eligibility Prediction</h1>
    #             </div>"""
    # st.markdown(design, unsafe_allow_html=True)

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

    # married = right.selectbox('Married', ('Yes','No'))
    # dependent = left.selectbox('Dependents', ('None', 'One', 'Two', 'Three'))
    # education = right.selectbox('Education', ('Graduate', 'Non-Graduate'))
    # self_employed = left.selectbox('Self-Employed', ('Yes', 'No'))
    # applicant_income = right.number_input('Applicant Income')
    # coApplicant_income = left.number_input(
    #     'Co - Applicant Income')
    # loan_amount = right.number_input('Loan Amount')
    # loan_amount_term = left.number_input('Loan Tenor (In Months)')
    # credit_history = right.number_input('Credit History', 0.0, 1.0)
    # property_area = st.selectbox('Property Area', ('Semiurban','Urban', 'Rural'))
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
    # mar = 0 if married == 'Yes' else 1
    # dep = float(0 if dependent == 'None' else 1 if dependent == 'One' else 2 if dependent == 'Two' else 3)
    # edu = 0 if education == 'Graduate' else 1
    # sem = 0 if self_employed == 'Yes' else 1
    # pro = 0 if property_area == 'Semiurban' else 1 if property_area == 'Urban' else 2
    # lam = loan_amount/1000
    # cap = coApplicant_income / 1000

    #Making prediction
    prediction = Final_Model.predict([[gen, cre, age, credit_score,
                                               estimated_salary]])
    result = 'Stayed' if prediction == 0 else 'Exited'

    return result

if __name__ == "__main__":
    main()