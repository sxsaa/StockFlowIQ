import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from pandas.tseries.offsets import DateOffset

from pmdarima import auto_arima
import statsmodels.api as sm 

# import dataset
store_sales = pd.read_csv("C:\\Users\\Kavindu Sasanka\\Documents\\Academics\\University\\Semester 6\\Data Management Project\\Objective Models\\Sales Forecasting\\ARIMA Model\\SeriesReport-Not Seasonally Adjusted Sales - Monthly (Millions of Dollars).csv", header=0)
# print(store_sales.head(100))
print(store_sales.tail(10))
store_sales.dropna(inplace=True)

# check null values in the dataset
print(store_sales.info())

# dropping store and item columns
## store_sales = store_sales.drop(['store', 'item'], axis=1)
# print(store_sales.info())

# convert date from object data type to dateTime datatype
store_sales['Period'] = pd.to_datetime(store_sales['Period'])
# print(store_sales.info())

# result = adfuller(store_sales['Value'])

# # Extract and print the test results
# print("ADF Statistic:", result[0])
# print("p-value:", result[1])
# print("Critical Values:", result[4])

# # Interpretation of results
# if result[1] < 0.05:
#     print("The data is stationary (reject the null hypothesis)")
# else:
#     print("The data is not stationary (fail to reject the null hypothesis)")

store_sales['Seasonal First Difference']=store_sales['Value']-store_sales['Value'].shift(12)
# print(store_sales.head(20))

# result = adfuller(store_sales['Seasonal First Difference'].dropna())

# # Extract and print the test results
# print("ADF Statistic:", result[0])
# print("p-value:", result[1])
# print("Critical Values:", result[4])

# # Interpretation of results
# if result[1] < 0.05:
#     print("The data is stationary (reject the null hypothesis)")
# else:
#     print("The data is not stationary (fail to reject the null hypothesis)")

# def adfuller_test(sales):
#     result=adfuller(sales)
#     labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
#     for value,label in zip(result,labels):
#         print(label+' : '+str(value) )
#     if result[1] <= 0.05:
#         print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
#     else:
#         print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

# adfuller_test(store_sales['Value'])
# adfuller_test(store_sales['Seasonal First Difference'].dropna())

# plt.figure(figsize=(15,7))
# plt.plot(store_sales['Period'], store_sales['Seasonal First Difference'])
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.title("Monthly Customer Sales")
# plt.show()


# Automatic search for best p, d, q values
# auto_model = auto_arima(store_sales['Value'], seasonal=False, trace=True)
# print(auto_model.summary())

model=sm.tsa.statespace.SARIMAX(store_sales['Value'],order=(0, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()

# import matplotlib.pyplot as plt

# Assuming 'results' is already defined from fitting a SARIMA model
store_sales['forecast'] = results.predict(start=327, end=340, dynamic=True)

# Plotting the actual values and the forecasted values
store_sales[['Value', 'forecast']].plot(figsize=(12, 8))

# Display the plot
plt.show()

import matplotlib.pyplot as plt

# Plot only the part of data that includes actual values and forecasted values (from index 327 to 340)
fig, ax = plt.subplots(figsize=(12, 8))

# Plot actual data and forecasted data
store_sales.loc[327:340, ['Value', 'forecast']].plot(ax=ax)

# Set title and labels
ax.set_title('Actual vs Forecasted Sales (Zoomed in)', fontsize=16)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Sales', fontsize=12)

# Optionally, adjust y-axis limits if you want to zoom on the values
ax.set_ylim([min(store_sales.loc[327:340, ['Value', 'forecast']].min()), 
             max(store_sales.loc[327:340, ['Value', 'forecast']].max())])

# Show the plot
plt.show()


from pandas.tseries.offsets import DateOffset
# import matplotlib.pyplot as plt

# Step 1: Ensure the 'store_sales' DataFrame index is datetime
store_sales.index = pd.to_datetime(store_sales.index)

print(store_sales)

# Step 2: Get the last date from the existing data (ensure it's a datetime object)
last_date = store_sales.index[-1]

print(last_date)

# Step 3: Create future dates (next 24 months)
future_dates = [last_date + DateOffset(months=x) for x in range(1, 25)]  # 24 months ahead

# Step 4: Print future dates to verify
print(future_dates)


# converting date to a Month period and then sum of the number of items in each month
##store_sales['date'] = store_sales['date'].dt.to_period("M")
##monthly_sales = store_sales.groupby('date').sum().reset_index()
# converting resulting date column to timstamp datatype
##monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
# print(monthly_sales)
# print(monthly_sales.info())

# setting date column as an index 
##monthly_sales.set_index('date', inplace=True)
# print(monthly_sales.head(10))

# visulaiztion 
# plt.figure(figsize=(15,7))
# plt.plot(store_sales['Period'], store_sales['Value'])
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.title("Monthly Customer Sales")
# plt.show()

##monthly_sales['sales_diff'] = monthly_sales['sales'] - monthly_sales['sales'].shift(12)
##print(monthly_sales)

# check whether sales data is stationary or non-stationary
##result_adfuller = adfuller(monthly_sales['sales_diff'].dropna())
# extracting ADF results
##adf_stat = result_adfuller[0]
##p_value = result_adfuller[1]
##critical_values = result_adfuller[4]

##print(f'p-value : {p_value}')






