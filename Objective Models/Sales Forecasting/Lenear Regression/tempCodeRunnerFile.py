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