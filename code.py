"""
Financial modeling
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from  sklearn.linear_model import LinearRegression

train_data = pd.read_excel("train.xlsx")
test_data = pd.read_excel("test.xlsx")
dep_var = train_data[["ERTUPRS"]]
indep_var = train_data[["ERBIST100",
                        "RKUR",
                        "RPETROL",
                        "RFAIZ",
                        "RSUE",
                        "RALTIN",
                        "RM2",
                        "RM3",
                        "RTUFE"]]


class Analysis:
    """
    Necessary for analysis
    """
    def read_value(self,arr,txt, dataset):
        """
         Read excel
        """
        arr_ = dataset[txt].values
        for i in arr_:
            arr.append(i)

    def test_model(self,dict_model):
        """
        Model equation
        """
        return (model.intercept_) + (
                model.coef_[0][0]*dict_model["bist100"]) + (
                model.coef_[0][1]*dict_model["kur"]) + (
                model.coef_[0][2]*dict_model["petrol"]) + (
                model.coef_[0][3]*dict_model["faiz"]) + (
                model.coef_[0][4]*dict_model["sue"]) + (
                model.coef_[0][5]*dict_model["altin"]) + (
                model.coef_[0][6]*dict_model["arzm2"]) + (
                model.coef_[0][7]*dict_model["arzm3"]) + (
                model.coef_[0][8]*dict_model["tufe"])

class Model:
    """
    Creating two different models
    """
    def stats_model(self, dependent, independent):
        """
        For Stats model
        """
        const = sm.add_constant(independent)
        model_arb = sm.OLS(dependent, const).fit()
        return model_arb.summary()

    def sk_model(self, dependent, independent):
        """"
        For Scikit model
        """
        linear_reg = LinearRegression()
        return linear_reg.fit(independent, dependent)

model_obj = Model()
analysis_obj = Analysis()

model = model_obj.sk_model(dep_var,indep_var)

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


analysis_obj.read_value(TUPRS_values, "ERTUPRS", test_data)
analysis_obj.read_value(BIST100_values, "ERBIST100", test_data)
analysis_obj.read_value(kur_values, "RKUR", test_data)
analysis_obj.read_value(petrol_values, "RPETROL", test_data)
analysis_obj.read_value(altin_values, "RALTIN", test_data)
analysis_obj.read_value(tufe_values, "RTUFE", test_data)
analysis_obj.read_value(m2_values, "RM2", test_data)
analysis_obj.read_value(m3_values, "RM3", test_data)
analysis_obj.read_value(sue_values, "RSUE", test_data)
analysis_obj.read_value(faiz_values, "RFAIZ", test_data)

COUNT = len(BIST100_values)
arr_result = []

for index in range(COUNT):
    func_param = {"bist100": BIST100_values[index],
            "kur": kur_values[index],
            "petrol": petrol_values[index],
            "faiz": faiz_values[index],
            "sue":sue_values[index],
            "altin":altin_values[index],
            "arzm2":m2_values[index],
            "arzm3":m3_values[index],
            "tufe" :tufe_values[index]}
    test_values = analysis_obj.test_model(func_param)
    arr_result.append(test_values)
    func_param.clear()

df_result = pd.DataFrame(arr_result).T
df_real = pd.DataFrame(TUPRS_values).T

# Graphic
fig = plt.figure(figsize=(11, 8))
plt.style.use("classic")
plt.plot(arr_result,
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

