
## 📘 Project Description

**Intelligent Stock Management and Customer Insights** is an AI-driven, full-stack system designed to help supermarket chains manage stock levels, forecast sales, and understand customer sentiment in real-time — all from a unified platform.

### 🧠 What It Does
- **Sales Forecasting**: Predicts future sales using the SARIMA model to optimize restocking and reduce waste.
- **Customer Review System**: Collects and classifies customer feedback using a fine-tuned RoBERTa model.
- **Outlet Dashboard**: Manages inventory, tracks low-stock items, and visualizes daily/weekly/monthly sales.
- **Head Office Dashboard**: Aggregates data from all outlets for centralized analytics like income by city, top-selling products, and payment methods.
- **POS System**: Handles daily billing, offline sales recording, and automatic sync with the outlet database.



### ⚙️ Why These Technologies?
- **Node.js + Express.js** – Fast, scalable backend for POS and dashboards.
- **Python + Machine Learning** – Powerful for time series forecasting (SARIMA) and NLP (RoBERTa sentiment analysis).
- **PostgreSQL** – Reliable relational DB for managing structured inventory and sales data.
- **Google OAuth2 + BCrypt** – Ensures secure user authentication and session handling.

### 🚧 Challenges Faced
- Handling non-stationary sales data with traditional models like Linear Regression and Random Forest, which led to the adoption of SARIMA for better accuracy.
- Training and integrating a sentiment analysis model that performs well across mixed, noisy customer feedback.
- Ensuring smooth sync between offline POS systems and centralized databases without data loss or duplication.
- Building dashboards that visualize complex sales data in a clean, readable format.

### 🌱 Future Enhancements
- **Customer segmentation** using clustering to target promotions.
- **Mobile-friendly dashboards** for remote monitoring.
- **Real-time sync** between POS and Head Office even during network interruptions.
- **Deep learning** for even more accurate forecasting and advanced NLP sentiment classification.

This project shows how machine learning and thoughtful system design can transform traditional retail operations into a smart, data-driven experience.


## 🛠️ How to Install and Run the Project

Follow these steps to set up the project on your local machine.

---

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/sxsaa/StockFlowIQ.git
cd StockFlowIQ
```

### ✅ 2. Set Up Environment Variables

Create a .env file in the root directory and add the following environment variables:

`GOOGLE_CLIENT_ID`= your_google_client_id
`GOOGLE_CLIENT_SECRET`= your_google_client_secret
`SESSION_SECRET` = your_secret
`PG_USER` = postgres
`PG_HOST` = localhost
`PG_DATABASE` = db_name
`PG_PASSWORD` = your_postgres_password
`PG_PORT` = 5432

### ✅ 3. Install Backend Dependencies (Python)

```bash
cd backend
pip install -r requirements.txt
```

### ✅ 4. Install Frontend / Node Dependencies

```bash
npm install
```

### ✅ 5. Run the Project

```bash
cd backend
python app.py
```

```bash
nodemon index.js
```

Once the server is running, open your browser and go to:
```bash
http://localhost:3000
```