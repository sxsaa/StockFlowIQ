import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# import dataset
store_sales = pd.read_csv("C:\\Users\\Kavindu Sasanka\\Documents\\Academics\\University\\Semester 6\\Data Management Project\\Objective Models\\Sales Forecasting\\ARIMA Model\\SeriesReport-Not Seasonally Adjusted Sales - Monthly (Millions of Dollars).csv", header=0)
# drop all the rows if contain any null values
print(store_sales.head(10))
monthly_sales = store_sales.dropna()
# print(monthly_sales.info())


# convert period from object type to dateTime format
monthly_sales['Period'] = pd.to_datetime(monthly_sales['Period'])
# print(monthly_sales.info())
# print(monthly_sales.head(10))

# visualization 
plt.figure(figsize=(15,7))
plt.plot(monthly_sales['Period'], monthly_sales['Value'])
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Monthly Customer Sales")
plt.show()

# call the difference on the sales columns to make the sales data stationary
monthly_sales['sales_diff'] = monthly_sales['Value'].diff()
monthly_sales = monthly_sales.dropna()
print(monthly_sales.head(10))   

# visualization non-stationary data
# plt.figure(figsize=(15,7))
# plt.plot(monthly_sales['Period'], monthly_sales['sales_diff'])
# plt.xlabel("Date")
# plt.ylabel("Sales Difference")
# plt.title("Monthly Customer Sales Difference")
# plt.show()

# droping Period and Value columns from the dataset
supervised_data = monthly_sales.drop(['Period', 'Value'], axis=1)

# preparing the supervised data
for i in range(1,13):
    col_name = 'month_' + str(i)
    supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop=True)
print(supervised_data.head(10))
# print(supervised_data.info())

# split the data into Train and Test
train_data = supervised_data[:-12]
test_data = supervised_data[-12:]
print("Train Data Shape : ", train_data.shape)
print("Test Data Shape : " , test_data.shape)

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

x_train, y_train = train_data[:, 1:], train_data[:, 0:1]
x_test, y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()

print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)

# make prediction data frame to merge the predicted sales prices of all trained algs
sales_dates = monthly_sales['Period'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)

act_sales = monthly_sales['Value'][-13:].to_list()
print(act_sales)

# create the linear regression model and predicted output
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_pre = lr_model.predict(x_test)

lr_pre = lr_pre.reshape(-1, 1)
# this is a set matrix - contains the input features of the test data and also the predicted output. 
lr_pre_test_set = np.concatenate([lr_pre, x_test], axis=1)
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

result_list = []
for index in range(0, len(lr_pre_test_set)):
    result_list.append(lr_pre_test_set[index][0] + act_sales[index])
lr_pre_series = pd.Series(result_list, name="Linear Prediction")
predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index=True)

lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'], monthly_sales['Value'][-12:]))
lr_mae = mean_absolute_error(predict_df['Linear Prediction'	], monthly_sales['Value'][-12:])
lr_r2 = r2_score(predict_df['Linear Prediction'], monthly_sales['Value'][-12:])

print("Linear Regression MSE: ", lr_mse)
print("Linear Regression MAE: ", lr_mae)
print("Linear Regression R2: ", lr_r2)

# # visulaization of the prediction against the actual sales
plt.figure(figsize=(15,7))
# actual values 
plt.plot(monthly_sales['Period'], monthly_sales['Value'])
# predicted values
plt.plot(predict_df['Period'], predict_df['Linear Prediction'])
plt.title("Customer sales Forcast using LR Model")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(['Actual Sales', 'Predicted Sales'])
plt.show()

print(predict_df)
print(act_sales)

# Filter the predict_df for April 2017
april_sales = predict_df[predict_df['Period'] == '2019-01-06']['Linear Prediction'].values[0]

print(f"Predicted sales for April 2017: {april_sales}")