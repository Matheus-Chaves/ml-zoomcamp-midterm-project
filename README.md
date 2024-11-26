# Car Performance Prediction: Fuel Efficiency

## Project Overview

This project aims to predict **fuel efficiency** (measured as `combination_mpg`) of cars based on various features such as engine specifications, fuel type, drivetrain, and more. The main goal is to build a regression model that predicts how efficient a car is in terms of fuel consumption. This can be useful for car buyers, manufacturers, and environmental researchers who are interested in understanding and comparing the fuel efficiency of various car models.

The dataset used for this project contains detailed information on 550 car models, including key features like city and highway fuel efficiency, engine configurations, car make, model, and production year.

## Problem Description

The problem is a **regression problem**, where the target variable is `combination_mpg`, which represents the car's combined fuel efficiency in miles per gallon (mpg). We aim to develop a machine learning model capable of predicting this target variable based on other car features.

Key features in the dataset include:

- **city_mpg**: Fuel efficiency in miles per gallon for city driving.
- **highway_mpg**: Fuel efficiency in miles per gallon for highway driving.
- **cylinders**: Number of cylinders in the car's engine.
- **displacement**: Engine displacement.
- **drive**: Type of drivetrain (e.g., FWD, AWD).
- **fuel_type**: Type of fuel used (e.g., gasoline, electric).
- **make & model**: Car manufacturer and model.
- **year**: Year of production.

## Data Exploration and Preprocessing

### Dataset Overview

- **Shape**: 550 rows and 12 columns.
- **Data Cleaning**: Rows with missing or duplicate values were removed prior to analysis.
  
### Exploratory Data Analysis (EDA)

1. **Univariate Analysis**: Distribution of features such as `city_mpg`, `highway_mpg`, `year`, and more.
2. **Bivariate Analysis**: Investigated the relationships between `combination_mpg` and other features (e.g., `city_mpg`, `highway_mpg`, `cylinders`, `displacement`).
3. **Correlation Analysis**: Analyzed correlations between numerical features to identify potential linear or non-linear relationships.

## Machine Learning Models

The following machine learning models were trained and evaluated for the task:

1. **Random Forest Regressor**: An ensemble model based on decision trees, suitable for capturing complex relationships in the data.
2. **Support Vector Regression (SVR)**: A model that works well for both linear and non-linear regression tasks.
3. **Linear Regression**: A simple yet effective model for linear relationships between features.

Each model was evaluated using the following metrics:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

## Hyperparameter Tuning

Hyperparameter tuning was performed using **Grid Search** for both the Random Forest and SVR models. The best parameters were selected based on cross-validation performance, ensuring optimal model configurations.

## Future Work

- **Feature Engineering**: Further exploration of feature interactions and transformations may improve model performance.
- **Model Ensembling**: Combining predictions from different models could potentially lead to better results.
- **Deployment**: The trained models could be deployed in a web application or API to predict fuel efficiency for new car models.

## Installation

To run this project locally, clone this repository and run it with Docker:

```bash
docker build -t car-performance-api . 
```

Start the API by running the container:

```bash
docker run -p 9696:9696 car-performance-api
```

Then, make a post request passing car data to the route `http://localhost:9696/predict`:

```bash
curl -X POST http://localhost:9696/predict \
-H "Content-Type: application/json" \
-d '{
  "city_mpg": 25,
  "class": "midsize car",
  "cylinders": 4.0,
  "displacement": 2.5,
  "drive": "fwd",
  "fuel_type": "gas",
  "highway_mpg": 36,
  "make": "mazda",
  "model": "6",
  "transmission": "m",
  "year": 2014
}'
```

You will receive a response predicting the fuel efficiency of this car, like below:

```json
{
  "RandomForest_Prediction": 28.71,
  "SVR_Prediction": 29.057080495117557
}
```
