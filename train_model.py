import os
import logging

from src.data_utils import load_data, preprocess_data
from src.model_utils import train_model, evaluate_model, save_model

# Set up basic logging
# This creates project.log automatically
logging.basicConfig(
    filename="project.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# File paths
DATA_PATH = "data/Admission.csv"
MODEL_PATH = "models/mlp_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def main():
    try:
        # Log start of script
        logging.info("Starting neural network training script")

        # Load dataset
        df = load_data(DATA_PATH)
        logging.info("Dataset loaded successfully")

        # Preprocess data
        xtrain_scaled, xtest_scaled, ytrain, ytest, scaler, feature_names = preprocess_data(df)
        logging.info("Data preprocessing completed")

        # Train neural network model
        model = train_model(xtrain_scaled, ytrain)
        logging.info("MLP model trained successfully")

        # Evaluate model
        train_accuracy, test_accuracy, cm = evaluate_model(
            model,
            xtrain_scaled,
            xtest_scaled,
            ytrain,
            ytest
        )

        # Print results
        print("Training Accuracy:", train_accuracy)
        print("Testing Accuracy:", test_accuracy)
        print("Confusion Matrix:")
        print(cm)

        # Make sure models folder exists
        os.makedirs("models", exist_ok=True)

        # Save model and scaler
        save_model(model, scaler, MODEL_PATH, SCALER_PATH)
        logging.info("Model and scaler saved successfully")

        print("\nTraining completed successfully.")

    except FileNotFoundError:
        # Handle missing dataset
        logging.error("Dataset file not found")
        print("Error: dataset file not found.")

    except KeyError as e:
        # Handle missing columns
        logging.error(f"Missing column: {e}")
        print(f"Error: missing column: {e}")

    except Exception as e:
        # Handle any unexpected errors
        logging.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")


# Run the script
if __name__ == "__main__":
    main()