import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from math import log
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

df = pd.read_excel("dataset_tupras.xlsx")

warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# data info
def check(dataframe,head=5):
    print(dataframe.shape)
    print("********************************************************************")
    print(dataframe.dtypes)
    print("********************************************************************")
    print(dataframe.info())
    print("********************************************************************")
    print(dataframe.describe().T)
    print("********************************************************************")
    print(dataframe.columns)
    print("********************************************************************")
    print(dataframe.head(head))
    print("********************************************************************")
    print(dataframe.tail(head))
    print("********************************************************************")
    print(dataframe.isnull().sum())
    print("********************************************************************")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

##  Numerik ve kategorik değişken yakalanması

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

## Numerik değişken analizi
## sayısal degiskenlerin çeyreklik değerlerini gösterir ve histogram grafiği oluşturur

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df,col,True)

## ozellik cikarimi
def return_value(df,col_name):
    i = 0
    val_arr = [0]
    for num in range(0,len(df)-1):
        val = (-1)*log(df[col_name][i]/df[col_name][i+1])
        i += 1
        val_arr.append(val)
    return val_arr

def return_r_value(df,col_name2,col_name1="RFAIZ"):
    i = 0
    val_arr = []
    for num in range(0,len(df)):
        val =(df[col_name2][i] - df[col_name1][i])*100
        i += 1
        val_arr.append(val)
    return val_arr

df["RTUPRS"] = return_value(df,"TUPRS")
df["RBIST100"] = return_value(df,"BIST100")
df["RKUR"] = return_value(df,"USD/TRY")
df["RPETROL"] = return_value(df,"PETROL")
df["RFAIZ"] = return_value(df,"FAIZ")
df["RSUE"] = return_value(df,"SUE")
df["RALTIN"] = return_value(df,"ALTIN")
df["RM2"] = return_value(df,"M2")
df["RM3"] = return_value(df,"M3")
df["ENF"] = return_value(df,"TUFE")
df["ERTUPRS"] = return_r_value(df,"RTUPRS")
df["ERBIST100"] = return_r_value(df,"RBIST100")


## Model
y = df[["ERTUPRS"]]
X = df[["ERBIST100",
        "RKUR",
        "RPETROL",
        "RFAIZ",
        "RSUE",
        "RALTIN",
        "RM2",
        "RM3",
        "ENF"]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=1)
reg_model = LinearRegression().fit(X_train,y_train)

print(reg_model.intercept_)
print(reg_model.coef_)

# train RMSE
y_pred = reg_model.predict(X_train)
print("train RMSE ",np.sqrt(mean_squared_error(y_train,y_pred)))

# train RKARE
print("train RKARE ",reg_model.score(X_train,y_train))

# test RMSE
y_pred = reg_model.predict(X_test)
print("test RMSE ",np.sqrt(mean_squared_error(y_test,y_pred)))

# test RKARE
print("test RKARE ",reg_model.score(X_test,y_test))

# 10 katlı CV RMSE
print("10 katlı CV RMSE ",np.mean(np.sqrt(-cross_val_score(reg_model,X,y,cv=10,scoring="neg_mean_squared_error"))))

# 5 katlı CV RMSE
print("5 katlı CV RMSE ",np.mean(np.sqrt(-cross_val_score(reg_model,X,y,cv=5,scoring="neg_mean_squared_error"))))

