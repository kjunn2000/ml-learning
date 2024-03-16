import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression
import joblib

def main():
    df_sal = pd.read_csv('Salary_Data.csv')
    # regressor = joblib.load('train_salary.pkl')
    
    x = df_sal.iloc[:, :1]
    y = df_sal.iloc[:, 1:]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    

    y_pred_test = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    plt.scatter(X_train, y_train, color = 'lightcoral')
    plt.plot(X_test, y_pred_test, color = 'firebrick')
    plt.title('Salary vs Experience (Training Set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend(['X_test/Pred(y_test)', 'X_train/y_train'], title = 'Sal/Exp', loc='best', facecolor='white')
    plt.box(False)
    plt.show()
    
    print(regressor.score(X_test, y_test))
    
    print(f'Coefficient: {regressor.coef_}')
    print(f'Intercept: {regressor.intercept_}')
     
    # joblib.dump(regressor, 'train_salary.pkl')

if __name__=="__main__": 
    main() 