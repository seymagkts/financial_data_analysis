import numpy as np 
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from sklearn.linear_model import LinearRegression

train_data = pd.read_excel("train.xlsx")

dependent_var = train_data[["ERTUPRS"]]
independent_var = train_data[["ERBIST100","RKUR","RPETROL","RFAIZ","RSUE","RALTIN","RM2","RM3","RTUFE"]]

lm = LinearRegression()
model = lm.fit(independent_var,dependent_var)

test_data = pd.read_excel("test.xlsx")

TUPRS_values = []
BIST100_values = []
kur_values = []
petrol_values = []
altin_values = []
tufe_values = []
m2_values = []
m3_values = []
sue_values = []
faiz_values = []

def read_value(arr,txt,dataset):
    arr_ = dataset[txt].values
    for i in arr_:
        arr.append(i)
        
read_value(TUPRS_values,"ERTUPRS",test_data)
read_value(BIST100_values,"ERBIST100",test_data)
read_value(kur_values,"RKUR",test_data)
read_value(petrol_values,"RPETROL",test_data)
read_value(altin_values,"RALTIN",test_data)
read_value(tufe_values,"RTUFE",test_data)
read_value(m2_values,"RM2",test_data)
read_value(m3_values,"RM3",test_data)
read_value(sue_values,"RSUE",test_data)
read_value(faiz_values,"RFAIZ",test_data)

def test_model(bist100,kur,petrol,faiz,sue,altin,m2,m3,tufe): # predict
    return (model.intercept_) + (model.coef_[0][0]*bist100) + (model.coef_[0][1]*kur) + (model.coef_[0][2]*petrol) + (model.coef_[0][3]*faiz) + (model.coef_[0][4]*sue) + (model.coef_[0][5]*altin) + (model.coef_[0][6]*m2) + (model.coef_[0][7]*m3) + (model.coef_[0][8]*tufe)
 
arr = []
for i in range(len(BIST100_values)):
    test_values = test_model(BIST100_values[i],kur_values[i],petrol_values[i],faiz_values[i],sue_values[i],altin_values[i],m2_values[i],m3_values[i],tufe_values[i])
    arr.append(test_values)
df_result = pd.DataFrame(arr).T
df_real = pd.DataFrame(TUPRS_values).T
 
fig = plt.figure(figsize=(11,8))
plt.style.use("classic")
plt.plot(arr,
         color="#cd3333",
         linewidth=3,
         linestyle="--",
         marker="o",
         markersize=8,
         markerfacecolor="yellow",
         markeredgewidth=2,
         label="Model return")
plt.plot(TUPRS_values,
         "#3d59ab",
         linewidth=3,
         linestyle="--",
         marker="o",
         markersize=8,
         label="Real return")
plt.ylabel("Model")
plt.xlabel("Real")
plt.legend(shadow=True)
# plt.savefig(quality=95,fname="chart.png",facecolor="white")
plt.show()
