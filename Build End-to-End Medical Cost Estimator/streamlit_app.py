import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained model and preprocessor
model = joblib.load('models/final_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

st.title('Medical Cost Estimator')
st.write('Enter your details to get an estimated medical insurance cost.')

with st.form('input_form'):
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
    children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    smoker = st.selectbox('Smoker', ['yes', 'no'])
    region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

    submitted = st.form_submit_button('Estimate Cost')

    if submitted:
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }])

        # Preprocess the input data
        processed_input = preprocessor.transform(input_data)

        # Predict the cost
        predicted_cost = model.predict(processed_input)[0]
        st.success(f'Estimated Medical Cost: ${predicted_cost:.2f}')

        # SHAP explanation (for tree-based models)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_input)

            st.subheader('How features influence the prediction:')
            shap.summary_plot(shap_values, processed_input, feature_names=preprocessor.get_feature_names_out(), show=False)
            st.pyplot(plt.gcf())
        except Exception as e:
            st.warning(f'Could not generate SHAP plot: {e}')