from flask import Flask, jsonify, render_template, request
from whitenoise import WhiteNoise  # Import WhiteNoise
import pandas as pd
import joblib
import shap
import os
import base64
import io
import pickle
import bz2
import matplotlib.pyplot as plt
import tensorflow as tf

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/') 

def print_directory_structure(directory):
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir():
                print("📁 " + entry.path)  # Print the directory path
                print_directory_structure(entry.path)  # Recursively print contents of the subdirectory
            else:
                print("📃 " + entry.path)  # Print the file path

@app.route('/')
def index():
    # Get the current working directory
    current_directory = os.getcwd()

    # Print the entire directory structure
    print_directory_structure(current_directory)

    return render_template('index.html')

@app.route("/predictgood", methods=['POST'])
def do_prediction_good():
    #load test.txt and print it out
    with open('static/test.txt') as f:
        print(f.read())

    json_data = request.get_json()
    json_data = {
        "age": json_data["age"],
        "hypertension": json_data["hypertension"],
        "heart_disease": json_data["heart_disease"],
        "bmi": json_data["bmi"],
        "HbA1c_level": json_data["HbA1c_level"],
        "blood_glucose_level": json_data["blood_glucose_level"],
        "gender_encoded": json_data["gender_encoded"],
        "smoking_history_encoded": json_data["smoking_history_encoded"]
    }
    print(json_data)
    df = pd.DataFrame(json_data, index=[0])

    # Print current working directory
    print("Current directory: " + os.getcwd())
    
    # predict
    with bz2.BZ2File("static/explainer_good.bz2", "rb") as file:
        explainer = pickle.load(file)
    shap_values = explainer.shap_values(df)

    print("IT REACHED HERE")
    print("Explainer loaded" + shap_values)

    model = tf.keras.models.load_model("static/diabetes_good_model.h5")
    print("Model loaded")
        
    y_pred = model.predict(df)
    pred_diabetes = int(y_pred[0])
    
    # shap

    i = 0
    shap.force_plot(explainer.expected_value[i], shap_values[i], df.iloc[i], matplotlib=True, show=False, plot_cmap=['#77dd77', '#f99191'])

    # Save the plot as a Base64 encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode("utf-8")

    # Return the Base64-encoded image string in the response
    result_map = {0: False, 1: True}
    print(result_map[pred_diabetes])
    return jsonify({'diabetes': result_map[pred_diabetes], 'image_base64': base64_image})

@app.route("/predictbad", methods=['POST'])
def do_prediction_bad():
    json_data = request.get_json()
    json_data = {
        "age": json_data["age"],
        "hypertension": json_data["hypertension"],
        "heart_disease": json_data["heart_disease"],
        "bmi": json_data["bmi"],
        "HbA1c_level": json_data["HbA1c_level"],
        "blood_glucose_level": json_data["blood_glucose_level"],
        "gender_encoded": json_data["gender_encoded"],
        "smoking_history_encoded": json_data["smoking_history_encoded"]
    }
    df = pd.DataFrame(json_data, index=[0])

    # predict

    model = tf.keras.models.load_model('static/diabetes_bad_model.keras')
    y_pred = model.predict(df)
    pred_diabetes = int(y_pred[0])
    
    # shap
    with bz2.BZ2File("static/explainer_bad.bz2", "rb") as file:
        explainer = pickle.load(file)
    shap_values = explainer.shap_values(df)

    i = 0
    shap.force_plot(explainer.expected_value[i], shap_values[i], df.iloc[i], matplotlib=True, show=False, plot_cmap=['#77dd77', '#f99191'])

    # Save the plot as a Base64 encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode("utf-8")

    result_map = {0: False, 1: True}
    return jsonify({'diabetes': result_map[pred_diabetes], 'image_base64': base64_image})

if __name__ == '__main__':
  app.run(port=5000)
