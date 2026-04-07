import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def train_model(xtrain_scaled, ytrain):
    # Create MLP neural network model
    # This is kept simple and similar to the notebook
    model = MLPClassifier(
        hidden_layer_sizes=(3, 2),
        batch_size=50,
        max_iter=200,
        random_state=123
    )

    # Train the model
    model.fit(xtrain_scaled, ytrain)

    # Return trained model
    return model


def evaluate_model(model, xtrain_scaled, xtest_scaled, ytrain, ytest):
    # Make predictions on training set
    ypred_train = model.predict(xtrain_scaled)

    # Make predictions on test set
    ypred_test = model.predict(xtest_scaled)

    # Calculate training accuracy
    train_accuracy = accuracy_score(ytrain, ypred_train)

    # Calculate test accuracy
    test_accuracy = accuracy_score(ytest, ypred_test)

    # Build confusion matrix
    cm = confusion_matrix(ytest, ypred_test)

    # Return evaluation results
    return train_accuracy, test_accuracy, cm


def save_model(model, scaler, model_path, scaler_path):
    # Save trained model
    joblib.dump(model, model_path)

    # Save scaler
    joblib.dump(scaler, scaler_path)


def load_model(model_path, scaler_path):
    # Load trained model
    model = joblib.load(model_path)

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Return both
    return model, scaler