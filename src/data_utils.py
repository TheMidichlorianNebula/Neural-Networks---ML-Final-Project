import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    # Load the dataset from CSV
    df = pd.read_csv(file_path)

    # Return dataframe
    return df


def preprocess_data(df):
    # Convert Admit_Chance into binary classes
    # 1 = admit chance is 80% or higher
    # 0 = admit chance is below 80%
    df["Admit_Chance"] = (df["Admit_Chance"] >= 0.8).astype(int)

    # Drop unnecessary column
    df = df.drop("Serial_No", axis=1)

    # Convert categorical-looking numeric columns to object
    # This matches the notebook approach
    df["University_Rating"] = df["University_Rating"].astype("object")
    df["Research"] = df["Research"].astype("object")

    # Create dummy variables
    df = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype=int)

    # Split features and target
    X = df.drop("Admit_Chance", axis=1)
    y = df["Admit_Chance"]

    # Split into training and testing sets
    xtrain, xtest, ytrain, ytest = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=123,
        stratify=y
    )

    # Initialize scaler
    scaler = MinMaxScaler()

    # Fit scaler on training data only
    scaler.fit(xtrain)

    # Scale training and test data
    xtrain_scaled = scaler.transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    # Return everything needed later
    return xtrain_scaled, xtest_scaled, ytrain, ytest, scaler, X.columns