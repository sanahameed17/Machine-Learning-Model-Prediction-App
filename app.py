from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import pandas as pd
import numpy as np

app = Flask(__name__)

# Welcome page
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Home page
@app.route('/home')
def home():
    return render_template('home.html')

# Regression page
@app.route('/regression')
def regression_page():
    return render_template('regression.html')

# Classification page
@app.route('/classification')
def classification_page():
    return render_template('classification.html')

# Function for handling regression model
@app.route('/regression_model', methods=['POST'])
def regression_model():
    if request.method == 'POST':
        # Load dataset
        file = request.files['dataset']
        data = pd.read_csv(file)

        # Preprocess the data (e.g., handle missing values)
        data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())
        
        # Encode categorical variable (one-hot encoding)
        data = pd.get_dummies(data, drop_first=True)

        # Assume the last column is the target, and the rest are features
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply the regression model
        model_type = request.form.get('model', None)
        model = None  # Initialize model variable

        if model_type == 'linear_regression':
            model = LinearRegression()

        # Check if model was set
        if model is not None:
            # Train the model
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)

            # Calculate metrics
            metric = request.form.get('metric', None)
            if metric == 'mse':
                mse = mean_squared_error(y_test, y_pred)
                result = f"Mean Squared Error: {mse:.2f}"
            elif metric == 'r2':
                r2 = r2_score(y_test, y_pred) * 100
                result = f"R² Score: {r2:.2f}%"
            else:
                result = "Please select a valid metric."
        else:
            result = "Please select a valid regression model."

        return render_template('result.html', result=result)

# Function for handling classification model
@app.route('/classification_model', methods=['POST'])
def classification_model():
    if request.method == 'POST':
        # Load dataset
        file = request.files['dataset']
        data = pd.read_csv(file)

        # Preprocess the data (e.g., handle missing values)
        # Fill missing values with the mode (for categorical columns)
        data.fillna(data.mode().iloc[0], inplace=True)

        # Check if there are still NaN values after filling
        missing_values = data.isnull().sum()
        if missing_values.any():
            missing_info = missing_values[missing_values > 0]  # Get only columns with missing values
            missing_details = missing_info.to_frame().reset_index()
            missing_details.columns = ['Column Name', 'Missing Values']
            return render_template('missing_values.html', missing_details=missing_details.to_html(index=False))

        # Encode categorical variables (one-hot encoding)
        data = pd.get_dummies(data, drop_first=True)

        # Assume the last column is the target, and the rest are features
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply the classification model
        model_type = request.form.get('model', None)
        model = None  # Initialize model variable

        if model_type == 'logistic_regression':
            model = LogisticRegression(max_iter=1000)

        # Check if model was set
        if model is not None:
            # Train the model
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)

            # Calculate metrics
            metric = request.form.get('metric', None)
            if metric == 'accuracy':
                accuracy = accuracy_score(y_test, y_pred) * 100
                result = f"Accuracy: {accuracy:.4f}%"
            elif metric == 'precision':
                precision = precision_score(y_test, y_pred, average='weighted') * 100
                result = f"Precision: {precision:.4f}%"
            elif metric == 'recall':
                recall = recall_score(y_test, y_pred, average='weighted') * 100
                result = f"Recall: {recall:.4f}%"
            elif metric == 'f1':
                f1 = f1_score(y_test, y_pred, average='weighted') * 100
                result = f"F1 Score: {f1:.4f}%"
            else:
                result = "Please select a valid metric."
        else:
            result = "Please select a valid classification model."

        return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)