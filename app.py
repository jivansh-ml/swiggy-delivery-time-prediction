import joblib
import pandas as pd
from flask import Flask,render_template,request
import mlflow
import json
from mlflow import MlflowClient
from sklearn import set_config
from scripts.data_cleaning_utils import perform_data_cleaning
from sklearn.pipeline import Pipeline

set_config(transform_output='pandas')

import dagshub
import mlflow.client

dagshub.init(repo_owner='jivanshs51', repo_name='swiggy-delivery-time-prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/jivanshs51/swiggy-delivery-time-prediction.mlflow")


app=Flask(__name__)

#columns to preprocess
num_cols=['age','ratings','pickup_time_minutes','distance']
nominal_cat_cols = ['weather','type_of_order','type_of_vehicle','festival','city_type','is_weekend','order_time_of_day']
ordinal_cat_cols = ['traffic','distance_type']

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info

def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

# load the model info to get the model name
model_name = load_model_information("run_information.json")['model_name']

client = MlflowClient()

model = joblib.load('models/model.joblib')

preprocessor_path = 'models/preprocessor.joblib'
preprocessor = load_transformer(preprocessor_path)

model_pipe = Pipeline(steps=[
      ('preprocess',preprocessor),
      ('regressor',model)
])
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    try:
            # Check if any field is empty
            fields = ["age", "ratings", "weather", "traffic", "vehicle_condition", 
                      "type_of_order", "type_of_vehicle", "multiple_deliveries", 
                      "festival", "city_type", "is_weekend", "pickup_time_minutes", 
                      "order_time_of_day", "distance", "distance_type"]
    
            for field in fields:
                value = request.form.get(field)
                if value is None or value == "":
                    return render_template('index.html', error=f"Please fill in: {field}")
    
            # 1. Get all 15 features from the HTML form
            data = {
                "age": [float(request.form.get("age"))],
                "ratings": [float(request.form.get("ratings"))],
                "weather": [request.form.get("weather")],
                "traffic": [request.form.get("traffic")],
                "vehicle_condition": [int(request.form.get("vehicle_condition"))],
                "type_of_order": [request.form.get("type_of_order")],
                "type_of_vehicle": [request.form.get("type_of_vehicle")],
                "multiple_deliveries": [int(request.form.get("multiple_deliveries"))],
                "festival": [request.form.get("festival")],
                "city_type": [request.form.get("city_type")],
                "is_weekend": [int(request.form.get("is_weekend"))],
                "pickup_time_minutes": [float(request.form.get("pickup_time_minutes"))],
                "order_time_of_day": [request.form.get("order_time_of_day")],
                "distance": [float(request.form.get("distance"))],
                "distance_type": [request.form.get("distance_type")]
            }

            df = pd.DataFrame(data)
            # get the predictions directly since the form already provides cleaned data
            predictions = model_pipe.predict(df)[0]

            return render_template('index.html', result=round(predictions, 2))
    
    except Exception as e:
                return render_template('index.html', error=str(e))

# Start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)