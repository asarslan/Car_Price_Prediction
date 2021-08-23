from airflow import DAG
from airflow.operators.python import task
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR

from datetime import datetime
import pandas as pd
import numpy as np


dag_config = Variable.get("variables_config", deserialize_json=True)

training_data = dag_config["training_data_path"]
testing_data = dag_config["testing_data_path"]

train_x_a = dag_config["train_x_a"]
train_y_a = dag_config["train_y_a"]
test_x = dag_config["test_x"]
test_y = dag_config["test_y"]

scaled_training_data = dag_config["scaled_training_data"]
scaled_testing_data = dag_config["scaled_testing_data"]

transformed_data = dag_config["transformed_data"]

LR_results = dag_config["LR_results"]
PR_results = dag_config["PR_results"]
Lasso_results = dag_config["Lasso_results"]
SVR_results = dag_config["SVR_results"]



def preprocessing_training_testing_data(_file_name= training_data, _test_file= testing_data, **kwargs):
    train_data = pd.read_csv(_file_name) # reading training_data.csv file
    train_data  = train_data.dropna(subset=['price']) # dropping the rows completely from training_data that has null prices

    test_data = pd.read_csv(_test_file) # reading testing_data.csv file
    test_data = test_data.dropna(subset=['price']) # dropping the rows completely from testing_data that has null prices

    train_data.to_csv(training_data, index=False)
    test_data.to_csv(testing_data, index=False)

    print(train_data.head(2))
    print(test_data.head(2))

def splitting_training_testing_data(_file_name= training_data, _test_file= testing_data, **kwargs):
    processed_train_data = pd.read_csv(_file_name)
    x_train = processed_train_data.drop(columns='price') # containing all input features except the price column
    y_train = processed_train_data[['price']].copy() # containing the target label "price" only
    
    processed_test_data = pd.read_csv(_test_file)
    x_test = processed_test_data.drop(columns='price') # containing all input features except the price column
    y_test = processed_test_data[['price']] # containing the target label "price" only

    print(x_train.head(2))
    print(y_train.head(2))
    
    x_train.to_csv(train_x_a, index=False)
    y_train.to_csv(train_y_a, index=False)
    
    x_test.to_csv(test_x, index=False)
    y_test.to_csv(test_y, index=False)
    
    nan_columns = processed_train_data.columns[processed_train_data.isna().any()].tolist() # finding the features (columns) that contain missing (NaN) values and then returning all of them in a list
    nan_test_columns = processed_test_data.columns[processed_test_data.isna().any()].tolist() # finding the features (columns) that contain missing (NaN) values and then returning all of them in a list

    print("nan columns for training data: ", nan_columns)
    print("nan columns for testing data: ", nan_test_columns)

def handling_missing_values(_file_name= train_x_a, _test_file= test_x, **kwargs):
    train_x_b = pd.read_csv(_file_name)
    test_x_b = pd.read_csv(_test_file)
    
    train_x_b = train_x_b.fillna(train_x_b.median()) # filling the missing values with the median value of the corresponding feature
    test_x_b = test_x_b.fillna(test_x_b.median()) # filling the missing values with the median value of the corresponding feature

    train_x_b.to_csv(train_x_a, index=False)
    test_x_b.to_csv(test_x, index=False)

    print(train_x_b.head(2))
    print(test_x_b.head(2))

def handling_categorical_columns(_file_name= train_x_a, _test_file= test_x, **kwargs):
    train_x_c = pd.read_csv(_file_name)
    test_x_c = pd.read_csv(_test_file)
    
    categorical_columns = train_x_c.select_dtypes(include='object').columns.tolist() # finding all features (columns) that contain categorical values ('object') and storing their column names into list categorical_columns
    categorical_test_columns = test_x_c.select_dtypes(include='object').columns.tolist() # finding all features (columns) that contain categorical values ('object') and storing their column names into list categorical_columns

    # One hot encoding function that uses pandas get_dummies.
    train_x_c = pd.get_dummies(train_x_c, columns=categorical_columns) # modifying train_x_c by replacing categorical columns with their one-hot encoding representations
    test_x_c = pd.get_dummies(test_x_c,columns=categorical_test_columns) # modifying test_x_c by replacing categorical columns with their one-hot encoding representations
    
    print(categorical_columns)
    print(categorical_test_columns)

    print(train_x_c.head(2))
    print(test_x_c.head(2))

    train_x_c.to_csv(train_x_a, index=False)
    test_x_c.to_csv(test_x, index=False)

def standardizing_training_test_data(_file_name= train_x_a, _test_file= test_x, **kwargs):
    train_x_c = pd.read_csv(_file_name)
    test_x_c = pd.read_csv(_test_file)
    
    # scaling all columns in train_x_c with standardization with a transformer called StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_x_c)
    
    train_scaler = scaler.transform(train_x_c)
    train_x_c = pd.DataFrame(data= train_scaler, index= train_x_c.index, columns= train_x_c.columns) # converting the output of scaler numpy array into dataframe after standardization also adding the original data frame’s indices and columns to the new data frame
    
    test_scaler = scaler.transform(test_x_c) # scaling all columns in test_x_c with standardization 
    test_x_c = pd.DataFrame(data=test_scaler,index= test_x_c.index, columns= test_x_c.columns)
    
    print(train_x_c.head(2))
    print(test_x_c.head(2))

    train_x_c.to_csv(scaled_training_data, index=False)
    test_x_c.to_csv(scaled_testing_data, index=False)

def evaluate(test_y_a, preds, test_train): # calculating the metrics
    MSE_test = mean_squared_error(test_y_a, preds)
    RMSE_test = np.sqrt(MSE_test)
    Adjusted_RSquare_test = r2_score(test_y_a, preds)
    MAE = mean_absolute_error(test_y_a, preds)
    
    print('Model Performance for {:s}'.format(test_train))
    print('Mean Sqaure Error(MSE): {:0.4f}.'.format(MSE_test))
    print('Root Mean Sqaure Error(RMSE): {:0.4f}.'.format(RMSE_test))
    print('Adjusted R Square = {:0.2f}.'.format(Adjusted_RSquare_test))
    print('MAE = {:0.2f}.'.format(MAE))
    return {"MSE": MSE_test, "RMSE" : RMSE_test, "Adjusted_RSquare_test": Adjusted_RSquare_test, "MAE": MAE}

def linear_regression_model(scaled_training_data, train_y_a, test_x, test_y):
    scaled_data = pd.read_csv(scaled_training_data)
    train_y_b =  pd.read_csv(train_y_a)
    test_x_a =  pd.read_csv(test_x)
    test_y_a =  pd.read_csv(test_y)
    
    lr_model = LinearRegression() # linear regression model is created
    lr_model.fit(scaled_data, train_y_b) # training the model with scaled_data and train_y_b
    
    preds = lr_model.predict(test_x_a) # getting predictions

    test_results = evaluate(test_y_a, preds, 'Linear Regression')
    test_results = pd.DataFrame.from_dict(test_results, orient='index',columns=['Test Values'])
    test_results.to_csv(LR_results)
    print("Test Values: ", test_results)

def polynomial_regression_model(scaled_training_data, train_y_a, test_x, test_y):
    scaled_data = pd.read_csv(scaled_training_data)
    train_y_b =  pd.read_csv(train_y_a)
    test_x_a =  pd.read_csv(test_x)
    test_y_a =  pd.read_csv(test_y)
    
    poly_features = PolynomialFeatures(degree=2, include_bias=False) # running a polynomial transformation with degree 2 on scaled_data
    
    train_x_poly = pd.DataFrame(poly_features.fit_transform(scaled_data))
    test_x_poly = pd.DataFrame(poly_features.fit_transform(test_x_a))
    
    lr_model_poly  = LinearRegression() # creating LR for polynomial
    lr_model_poly.fit(train_x_poly, train_y_b)
    
    preds = lr_model_poly.predict(test_x_poly) # getting predictions

    train_x_poly.to_csv(transformed_data, index=False)
    test_results = evaluate(test_y_a, preds, 'Polynomial Regression')
    test_results = pd.DataFrame.from_dict(test_results, orient='index',columns=['Test Values'])
    test_results.to_csv(PR_results)
    print("Test Values: ", test_results)

def lasso_regression_model(scaled_training_data, train_y_a, test_x, test_y):
    scaled_data = pd.read_csv(scaled_training_data)
    train_y_b =  pd.read_csv(train_y_a)
    test_x_a =  pd.read_csv(test_x)
    test_y_a =  pd.read_csv(test_y)
    
    lasso = Lasso() # created lasso regression model
    lasso.fit(scaled_data, train_y_b)
    
    preds = lasso.predict(test_x_a) # getting predictions

    print(scaled_data.shape)
    print(test_x_a.shape)

    test_results = evaluate(test_y_a, preds, 'Lasso Regression')
    test_results = pd.DataFrame.from_dict(test_results, orient='index',columns=['Test Values'])
    test_results.to_csv(Lasso_results)
    print("Test Values: ", test_results)

def SVR_model(scaled_training_data, train_y_a, test_x, test_y):
    scaled_data = pd.read_csv(scaled_training_data)
    train_y_b =  pd.read_csv(train_y_a)
    test_x_a =  pd.read_csv(test_x)
    test_y_a =  pd.read_csv(test_y)
    
    svr = SVR() # created SVR model
    svr.fit(scaled_data, train_y_b)
    
    preds = svr.predict(test_x_a) # getting predictions

    test_results = evaluate(test_y_a, preds, 'SVR')
    test_results = pd.DataFrame.from_dict(test_results, orient='index',columns=['Test Values'])
    test_results.to_csv(SVR_results)
    print("Test Values: ", test_results)

def model_evaluation(LR_results, PR_results, Lasso_results, SVR_results):
    LR = pd.read_csv(LR_results)
    PR =  pd.read_csv(PR_results)
    Lasso =  pd.read_csv(Lasso_results)
    SVR =  pd.read_csv(SVR_results)

    lr_mse = LR['Test Values'].iloc[0]
    pr_mse =  PR['Test Values'].iloc[0]
    lasso_mse = Lasso['Test Values'].iloc[0]
    svr_mse = SVR['Test Values'].iloc[0]

    minMSE = min([lr_mse, pr_mse, lasso_mse, svr_mse])

    print('The best model according to MSE scores is SVR with MSE score: ', minMSE)

dag = DAG('ml_regression_pipeline',
           schedule_interval='0 12 * * *',
           start_date=datetime(2021, 8, 19), catchup=False)

task_1 = PythonOperator(
    task_id='preprocessing_training_testing_data',
    python_callable=preprocessing_training_testing_data,
    dag=dag,
)

task_2 = PythonOperator(
    task_id='splitting_training_testing_data',
    python_callable=splitting_training_testing_data,
    dag=dag,
)

task_3 = PythonOperator(
    task_id='handling_missing_values',
    python_callable=handling_missing_values,
    dag=dag,
)

task_4 = PythonOperator(
    task_id='handling_categorical_columns',
    python_callable=handling_categorical_columns,
    dag=dag,
)

task_5 = PythonOperator(
    task_id='standardizing_training_test_data',
    python_callable=standardizing_training_test_data,
    dag=dag,
)

task_6_1 = PythonOperator(
    task_id='linear_regression_prediction',
    python_callable=linear_regression_model,
    dag=dag,
    op_kwargs={'scaled_training_data': scaled_training_data, 'train_y_a': train_y_a, 'test_x': test_x,'test_y': test_y},
)

task_6_2 = PythonOperator(
    task_id='polynomial_regression_prediction',
    python_callable=polynomial_regression_model,
    dag=dag,
    op_kwargs={'scaled_training_data': scaled_training_data, 'train_y_a': train_y_a, 'test_x': test_x,'test_y': test_y},
)

task_6_3 = PythonOperator(
    task_id='lasso_regression_prediction',
    python_callable=lasso_regression_model,
    dag=dag,
    op_kwargs={'scaled_training_data': scaled_training_data, 'train_y_a': train_y_a, 'test_x': test_x,'test_y': test_y},
)

task_6_4 = PythonOperator(
    task_id='svr_prediction',
    python_callable=SVR_model,
    dag=dag,
    op_kwargs={'scaled_training_data': scaled_training_data, 'train_y_a': train_y_a, 'test_x': test_x,'test_y': test_y},
)

task_7 = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation,
    dag=dag,
    op_kwargs={'LR_results': LR_results, 'PR_results': PR_results, 'Lasso_results': Lasso_results,'SVR_results': SVR_results},
)

task_1 >> task_2 >> task_3 >> task_4 >> task_5 >> [task_6_1, task_6_2, task_6_3, task_6_4] >> task_7 #pipeline is created
