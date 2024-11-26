from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask('predict_car_performance')

def load_model(model_name):
    with open(f'{model_name}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

random_forest_model = load_model('random_forest')
svr_model = load_model('svr')

@app.route('/')
def home():
    return 'Hello! Welcome to the Car Performance Prediction API.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_df = pd.DataFrame([data])

    data_encoded = random_forest_model.named_steps['preprocessor'].transform(input_df)

    rf_prediction = random_forest_model.named_steps['regressor'].predict(data_encoded)
    svr_prediction = svr_model.named_steps['regressor'].predict(data_encoded)

    return jsonify({
        'RandomForest_Prediction': rf_prediction[0],
        'SVR_Prediction': svr_prediction[0]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
