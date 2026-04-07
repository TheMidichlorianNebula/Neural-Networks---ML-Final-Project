# UCLA Neural Networks Project – Admission Prediction

## Project Overview
This project is a modularized version of the original Jupyter Notebook for the UCLA admission prediction problem. It uses a neural network model to classify whether a student is likely to be admitted based on academic and profile-related inputs.

The original notebook was converted into a structured Python project using multiple .py files. Basic logging, error handling, and a Streamlit web application were added to improve the organization and usability of the project.

---

## Project Objectives
- Convert Jupyter Notebook code into modular Python files
- Improve code organization and readability
- Implement basic logging and error handling
- Train and save a neural network model
- Build an interactive Streamlit application
- Prepare the project for deployment on Streamlit Cloud
- Publish the project on GitHub

---

## Machine Learning Approach

This project uses a neural network classifier with Scikit-learn's MLPClassifier.

Steps:
1. Load the UCLA admissions dataset
2. Convert Admit_Chance into a binary target variable
   - 1 = 80% or higher
   - 0 = below 80%
3. Drop the Serial_No column
4. Convert University_Rating and Research into categorical variables
5. Create dummy variables
6. Split the data into training and testing sets
7. Scale the input features using MinMaxScaler
8. Train the neural network model
9. Evaluate model performance using accuracy and confusion matrix
10. Display results in a Streamlit app

---

## Project Structure

neural_networks_project/

app.py                  # Streamlit app
train_model.py          # Model training script
requirements.txt        # Dependencies
README.md               # Documentation
.gitignore

data/
  Admission.csv

models/
  mlp_model.pkl
  scaler.pkl

src/
  __init__.py
  data_utils.py         # Data loading + preprocessing
  model_utils.py        # Model training + evaluation

---

## Dataset

The dataset contains information related to student applications for admission.

Main columns include:
- Serial_No
- GRE_Score
- TOEFL_Score
- University_Rating
- SOP
- LOR
- CGPA
- Research
- Admit_Chance

The target variable is converted into a binary classification variable:
- 1 = likely admitted
- 0 = not likely admitted

---

## Installation

Install the required libraries:

pip install -r requirements.txt

---

## How to Run

Step 1: Train the model

python train_model.py

This will:
- Load the dataset
- Preprocess the data
- Split and scale the data
- Train the MLP neural network model
- Evaluate the model
- Save the model and scaler

Step 2: Run the Streamlit app

streamlit run app.py

---

## Application Features

- Displays training and testing accuracy
- Displays confusion matrix
- Displays neural network loss curve
- Allows admission prediction for new user input

---

## Logging

This project uses basic logging with Python's logging module.

- A file called project.log is automatically created
- Logs include:
  - Script execution
  - Model training
  - Predictions
  - Errors

---

## Error Handling

Basic error handling is implemented using try/except blocks.

Handles:
- Missing dataset file
- Missing columns in dataset
- Model loading errors
- Unexpected runtime errors

---

## Deployment

To deploy on Streamlit Cloud:
1. Upload the project to GitHub
2. Connect your repository to Streamlit Cloud
3. Select app.py as the main file
4. Deploy

---

## Author

Nathaniel Paquin
Algonquin College – CST2216 Individual Term Project