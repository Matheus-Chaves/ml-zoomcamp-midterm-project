import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

DATA_PATH = os.path.join('data', 'car_data.csv')

car_data = pd.read_csv(DATA_PATH)
car_data = car_data.dropna()
car_data = car_data.drop_duplicates()

X = car_data.drop(columns=['combination_mpg'])
y = car_data['combination_mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

def tune_model(model, X_train, y_train, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

param_grid_rf = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5]
}

param_grid_svr = {
    'regressor__C': [1, 10, 100],
    'regressor__epsilon': [0.01, 0.1, 0.2],
    'regressor__kernel': ['linear', 'rbf']
}

models = {
    'RandomForest': Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', RandomForestRegressor(random_state=42))]),
    'SVR': Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', SVR())])
}

best_models = {}

for model_name, model in models.items():
    print(f'Training and tuning {model_name}...')
    if model_name == 'RandomForest':
        best_model, best_params = tune_model(model, X_train, y_train, param_grid_rf)
    else:
        best_model, best_params = tune_model(model, X_train, y_train, param_grid_svr)
    
    best_models[model_name] = {'model': best_model, 'best_params': best_params}

def save_model(model, model_name):
    with open(f'{model_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

save_model(best_models['RandomForest']['model'], 'random_forest')
save_model(best_models['SVR']['model'], 'svr')

print("Models trained, evaluated, and saved successfully.")