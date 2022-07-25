import numpy as nm
import pandas as pd
import matplotlib.pyplot as plot
import gradio as gr
def rain_prediction(Year, Temperature):
    # importing data
    dataset = pd.read_csv('raindata3.csv')
    X = dataset.iloc[:, :-1].values # reading columns with indices 0 to -1 i.e the column just before the last -1 refers to last column.
    Y = dataset.iloc[:, -1].values #dependent colmn is only the last column.
    
    year = int(Year)
    temperature = float(Temperature)


    # Filling missing values -- Imputing strategy used is 'median'

    from sklearn.impute import SimpleImputer 
    from sklearn.preprocessing import LabelEncoder
    imputer = SimpleImputer(missing_values=nm.nan, strategy='median')
    imputer = imputer.fit(X[:, 1:14]) # fit is a method
    X[:, 1:14] = imputer.transform(X[:, 1:14])
    #print(X)
    #print(Y)

    # Encoding data

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    X = nm.array(ct.fit_transform(X))
    #print(X)

    # Splitting dataset into training and testing tests

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X ,Y, test_size=0.2, random_state = 1)

    # Model -- Multiple linear regression for rainfall of year 

    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit(X_train, Y_train)


    # Creating dictionary for areas and numbers as keys

    area_map = {1:'TELANGANA'}
    X[:, 0] = map(X[: ,0],area_map)

    # Predictions

    #y_pred = reg.predict(X_test)

    #print(y_pred)
    #print(Y_test)
    # In input list as user cannot give rainfall values of all months they are substituted by mean values of respective months.
    # This creates a uncertainity of predicted values and the error is of range -150 to 300
    input_list = nm.array([[1,year,9.58,11.68,15.6,20.18,27.37,145.12,249.59,218.05,177.50,77.22,21.85,9.14,temperature]]) 
    y1_pred = reg.predict(input_list[:1])
    #print(y1_pred)

    # Accuracy testing through R2 method

    #import sklearn.metrics as sm
    #print("Mean absolute error = ",round(sm.mean_absolute_error(Y_test, y_pred), 2)) = 0.08
    #print("R2 score = ",round(sm.r2_score(Y_test, y_pred),10)) = 0.9999996684

    # returning output
    string = str(y1_pred)
    str1 = string[1:-1]
    return str1

# user interface using gradio

interface = gr.Interface(fn = rain_prediction, inputs = ['text', 'text'],outputs = 'text')
interface.launch(share=True)
