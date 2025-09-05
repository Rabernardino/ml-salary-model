import pandas as pd
from sklearn import model_selection, pipeline, linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from dotenv import load_dotenv
import mlflow
import os

#Loading environment variables
load_dotenv()

uri = os.getenv('uri')

#Importing data
raw_data = pd.read_excel('../../data/raw_data.xlsx')
data = raw_data[raw_data['JOB_TITLE'].str.contains('Technical Specialist')]


#
X = data[['GENDER','ETHNICITY','JOB_TITLE','CONTRACT_TYPE','EDUCATION','PWD','CONTRACT_TIME','AGE_RANGE','CONTRACT_REGIME']]
y = data['SALARY']


#Splitting the variables in trainning and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)


#Filtered the categorical and numerical variable types for preprocessing
categorical = ['GENDER','ETHNICITY','JOB_TITLE','CONTRACT_TYPE','EDUCATION','CONTRACT_TIME','AGE_RANGE','CONTRACT_REGIME']
numerical = ['PWD']


#Instantiated the onehot encoder and the linear regression model
onehot = OneHotEncoder()
regressor = linear_model.LinearRegression()


#Setted the columns transformation
preprocessing = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('numerical', 'passthrough', numerical)
    ]
)

#Setted the model pipeline
model = pipeline.Pipeline(
    steps=[
        ('preprocessor', preprocessing),
        ('algorithm',regressor)
    ]
)

#Fitting the model
mlflow.set_tracking_uri(uri)
mlflow.set_experiment(experiment_id=1)


#Making the predictions using the trainning set

with mlflow.start_run():

    mlflow.sklearn.autolog()

    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    results_r2_score_train = r2_score(y_train, y_train_predict)
    results_r2_score_test = r2_score(y_test, y_test_predict)

    results_mean_sq_error_train = mean_absolute_error(y_train, y_train_predict)
    results_mean_sq_error_test = mean_absolute_error(y_test, y_test_predict)

    mlflow.log_metric("r2_Train", results_r2_score_train)
    mlflow.log_metric("r2_Test", results_r2_score_test)
    mlflow.log_metric("MAE_Train", results_mean_sq_error_train)
    mlflow.log_metric("MAE_Test", results_mean_sq_error_test)

