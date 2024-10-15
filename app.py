from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import csv




with open('HDPN.pkl', 'rb') as file:
    data = pickle.load(file)
    rf_classifier = data['random_forest']
    knn_classifier = data['knn']
    linear_model = data['linear_regression']
    scaler = data['scaler']  

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('heart.html')

@app.route('/take-test')
def take_test():
    return render_template('heart.html')

@app.route('/predict', methods=['POST'])
def predict():
    columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_mappings = {
        'Sex': ['Sex_M'],
        'ExerciseAngina': ['ExerciseAngina_Y'],
        'ChestPainType': ['ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA'],
        'RestingECG': ['RestingECG_Normal', 'RestingECG_ST'],
        'ST_Slope': ['ST_Slope_Flat', 'ST_Slope_Up']
    }
    columns_order = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 
                     'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'Sex_M', 
                     'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']

    if request.method == 'POST':
        user_input = [
            int(request.form.get('age', 0)),
            request.form.get('sex', ''),
            request.form.get('cp', ''),
            int(request.form.get('trestbps', 0)),
            int(request.form.get('chol', 0)),
            int(request.form.get('fbs', 0)),
            request.form.get('restecg', ''),
            int(request.form.get('thalach', 0)),
            request.form.get('exang', ''),
            float(request.form.get('oldpeak', 0.0)),
            request.form.get('slope', '')
        ]
        
        # Apply preprocessing
        processed_input = preprocess_input(user_input, columns, categorical_mappings, numerical_columns, columns_order)
        
        # Reorder columns in the processed input DataFrame to match the training order
        processed_input_scaled = scaler.transform(processed_input)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
            linear_probs = sigmoid(linear_preds)[0]

        # Make predictions
        rf_probs = rf_classifier.predict_proba(processed_input_scaled)[0][1]
        knn_probs = knn_classifier.predict_proba(processed_input_scaled)[0][1]
        linear_preds = linear_model.predict(processed_input_scaled)
        linear_probs = sigmoid(linear_preds)[0]

        
        
        # Decision rule based on threshold
        threshold = 0.75
        heart_disease_prediction = 1 if (rf_probs > threshold or knn_probs > threshold or linear_probs > threshold) else 0

        # Format the response
        result_message = "At risk of heart failure." if heart_disease_prediction == 1 else "Not at risk of heart failure."

        # Append the data with the prediction to heart.csv
        with open('heart.csv', mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(user_input + [heart_disease_prediction])

        return render_template('result.html', prediction=heart_disease_prediction)


if __name__ == '__main__':
    app.run(debug=True)
