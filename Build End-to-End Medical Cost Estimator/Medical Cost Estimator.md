# Medical Cost Estimator

This project provides an end-to-end solution for predicting medical insurance costs using supervised machine learning regression models. It includes data analysis, preprocessing, model training, and a Streamlit web application for interactive predictions.

## Project Structure

```
medical-cost-predictor/
├── app/
│   └── streamlit_app.py
├── data/
│   └── insurance.csv
├── models/
│   └── final_model.pkl
│   └── preprocessor.pkl
├── notebooks/
│   └── EDA.ipynb
│   └── Data_Preprocessing.ipynb
│   └── Model_Training.ipynb
├── Dockerfile
├── requirements.txt
├── README.md
```

## Key Steps

1.  **Exploratory Data Analysis (EDA)**: Performed in `notebooks/EDA.ipynb` to understand data distributions, correlations, and potential outliers.
2.  **Data Preprocessing**: Handled in `notebooks/Data_Preprocessing.ipynb`, including categorical encoding, numerical scaling, and train-test splitting.
3.  **Model Training & Comparison**: Implemented in `notebooks/Model_Training.ipynb`, comparing various regression algorithms (Linear, Ridge, Lasso, Random Forest, XGBoost, SVR) and evaluating them using MAE, RMSE, and R² Score. The best performing model is saved.
4.  **Model Interpretability**: SHAP values are used in `notebooks/Model_Training.ipynb` to explain the political impact of features.

## Setup and Run

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Jupyter notebooks**: Open and run the notebooks in the `notebooks/` directory in the specified order (EDA, Data_Preprocessing, Model_Training) to train the model and generate the `final_model.pkl` and `preprocessor.pkl` files in the `models/` directory.

4.  **Run Streamlit app**:
    ```bash
    streamlit run app/streamlit_app.py
    ```

## Dockerization

To containerize the Streamlit application using Docker, navigate to the project root directory and run the following commands:

```bash
    docker build -t medical-cost-estimator .
    docker run -p 8501:8501 medical-cost-estimator
```

This will make the Streamlit app accessible at `http://localhost:8501`. I will now proceed with the project. However, please note that I cannot directly execute the notebooks or run the Streamlit app in this environment. You will need to run these commands in your local environment. If you have any further questions or need assistance with specific steps, feel free to ask. Otherwise, I will consider the task complete. 

