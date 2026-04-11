import logging
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data_utils import load_data, preprocess_data
from src.model_utils import load_model, evaluate_model

# Set up basic logging
logging.basicConfig(
    filename="project.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File paths
DATA_PATH = "data/Admission.csv"
MODEL_PATH = "models/mlp_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Page setup
st.set_page_config(page_title="UCLA Admission Predictor", layout="centered")
st.title("UCLA Admission Prediction with Neural Networks")

try:
    # Load dataset
    df = load_data(DATA_PATH)

    # Preprocess dataset
    xtrain_scaled, xtest_scaled, ytrain, ytest, scaler, feature_names = preprocess_data(df)

    # Load saved model and scaler
    model, saved_scaler = load_model(MODEL_PATH, SCALER_PATH)

    # Evaluate model
    train_accuracy, test_accuracy, cm = evaluate_model(
        model,
        xtrain_scaled,
        xtest_scaled,
        ytrain,
        ytest
    )

    # Show accuracy results
    st.subheader("Model Accuracy")
    st.write(f"Training Accuracy: {train_accuracy:.4f}")
    st.write(f"Testing Accuracy: {test_accuracy:.4f}")

    # Show confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Show loss curve
    st.subheader("Loss Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(model.loss_curve_)
    ax2.set_title("Loss Curve")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Loss")
    st.pyplot(fig2)

    # User input section
    st.subheader("Predict Admission")

    gre = st.number_input("GRE Score", min_value=0, max_value=340, value=300)
    toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
    sop = st.number_input("SOP", min_value=0.0, max_value=5.0, value=3.0, step=0.5)
    lor = st.number_input("LOR", min_value=0.0, max_value=5.0, value=3.0, step=0.5)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    research = st.selectbox("Research", [0, 1])
    university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])

    if st.button("Predict"):
        # Build raw input dictionary
        input_data = {
            "GRE_Score": gre,
            "TOEFL_Score": toefl,
            "SOP": sop,
            "LOR": lor,
            "CGPA": cgpa,
            "University_Rating_1": 0,
            "University_Rating_2": 0,
            "University_Rating_3": 0,
            "University_Rating_4": 0,
            "University_Rating_5": 0,
            "Research_0": 0,
            "Research_1": 0
        }

        # Set selected university rating to 1
        input_data[f"University_Rating_{university_rating}"] = 1

        # Set selected research value to 1
        input_data[f"Research_{research}"] = 1

        # Convert to dataframe
        input_df = pd.DataFrame([input_data])

        # Reorder columns to match training data
        input_df = input_df[feature_names]

        # Scale the input
        input_scaled = saved_scaler.transform(input_df)

        # Predict class
        prediction = model.predict(input_scaled)[0]

        # Show result
        if prediction == 1:
            st.success("Prediction: Likely Admitted")
        else:
            st.error("Prediction: Not Likely Admitted")

        logging.info("Prediction made successfully")

except FileNotFoundError:
    # Handle missing files
    logging.error("Required file not found")
    st.error("Missing dataset or model files.")

except Exception as e:
    # Handle unexpected errors
    logging.error(f"App error: {e}")
    st.error(f"An error occurred: {e}")