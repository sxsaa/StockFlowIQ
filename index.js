import express from "express";
import bodyParser from "body-parser";
import pg from "pg";
import bcrypt from "bcrypt";
import passport from "passport";
import { Strategy } from "passport-local";
import GoogleStrategy from "passport-google-oauth2";
import session from "express-session";
import env from "dotenv";
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;
const saltRounds = 10;
env.config();

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

app.use(
  session({
    secret: process.env.SESSION_SECRET,
    resave: false,
    saveUninitialized: true,
  })
);
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static("public"));

app.use(passport.initialize());
app.use(passport.session());

const db = new pg.Client({
  user: process.env.PG_USER,
  host: process.env.PG_HOST,
  database: process.env.PG_DATABASE,
  password: process.env.PG_PASSWORD,
  port: process.env.PG_PORT,
});
db.connect();

app.get("/", (req, res) => {
  res.render("home.ejs");
});

app.get("/login", (req, res) => {
  res.render("login.ejs");
});

app.get("/register", (req, res) => {
  res.render("register.ejs");
});

app.get("/logout", (req, res) => {
  req.logout(function (err) {
    if (err) {
      return next(err);
    }
    res.redirect("/");
  });
});

app.get("/outlet_dashboard", async (req, res) => {
  console.log(req.user);

  ////////////////UPDATED GET SECRETS ROUTE/////////////////
  if (req.isAuthenticated()) {
    if (req.isAuthenticated()) {
      res.render("outlet_dashboard.ejs");
    } else {
      res.redirect("/login");
    }
  } 
});

// Nested route for customer reviews under outlet dashboard
app.get('/outletdata_dashboard/customer_reviews', (req, res) => {
  res.render("customer_reviews.ejs"); // Ensure 'customer_reviews.ejs' exists in 'views'
});


////////////////SUBMIT GET ROUTE/////////////////
app.get("/submit", function (req, res) {
  if (req.isAuthenticated()) {
    res.render("submit.ejs");
  } else {
    res.redirect("/login");
  }
});

// add_stocks GET ROUTE
app.get("/add_stocks", function (req, res) {
  if (req.isAuthenticated()) {
    res.render("add_stocks.ejs", { activePage: 'add_stocks' });
  } else {
    res.redirect("/login");
  }
});

app.get("/return_stocks", function (req, res) {
  if (req.isAuthenticated()) {
    res.render("return_stocks.ejs", { activePage: 'return_stocks' });
  } else {
    res.redirect("/login");
  }
});

app.get("/update_stocks", function (req, res) {
  if (req.isAuthenticated()) {
    res.render("update_stocks.ejs", { activePage: 'update_stocks' });
  } else {
    res.redirect("/login");
  }
});

app.get("/order_stocks", function (req, res) {
  if (req.isAuthenticated()) {
    res.render("update_stocks.ejs", { activePage: 'order_stocks' });
  } else {
    res.redirect("/login");
  }
});

app.get(
  "/auth/google",
  passport.authenticate("google", {
    scope: ["profile", "email"],
  })
);

app.get(
  "/auth/google/secrets",
  passport.authenticate("google", {
    successRedirect: "/outlet_dashboard",
    failureRedirect: "/login",
  })
);

app.post(
  "/login",
  passport.authenticate("local", {
    successRedirect: "/outlet_dashboard",
    failureRedirect: "/login",
  })
);

app.post("/register", async (req, res) => {
  const email = req.body.username;
  const password = req.body.password;

  try {
    const checkResult = await db.query("SELECT * FROM users WHERE email = $1", [
      email,
    ]);

    if (checkResult.rows.length > 0) {
      req.redirect("/login");
    } else {
      bcrypt.hash(password, saltRounds, async (err, hash) => {
        if (err) {
          console.error("Error hashing password:", err);
        } else {
          const result = await db.query(
            "INSERT INTO users (email, password) VALUES ($1, $2) RETURNING *",
            [email, hash]
          );
          const user = result.rows[0];
          req.login(user, (err) => {
            console.log("success");
            res.redirect("/outlet_dashboard");
          });
        }
      });
    }
  } catch (err) {
    console.log(err);
  }
});

app.post("/add_stocks", async (req, res) => {
  const { item_code, item_name, description, quantity, price, discount, mfd_date, exp_date } = req.body;

  try {
    await db.query(
      "INSERT INTO stocks (item_code, item_name, description, quantity, price, discount, mfd_date, exp_date) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
      [item_code, item_name, description, quantity, price, discount, mfd_date, exp_date]
    );
    res.redirect("/add_stocks");
  } catch (err) {
    res.status(500).send("Error inserting data: " + err.message);
  }
});

passport.use(
  "local",
  new Strategy(async function verify(username, password, cb) {
    try {
      const result = await db.query("SELECT * FROM users WHERE email = $1 ", [
        username,
      ]);
      if (result.rows.length > 0) {
        const user = result.rows[0];
        const storedHashedPassword = user.password;
        bcrypt.compare(password, storedHashedPassword, (err, valid) => {
          if (err) {
            console.error("Error comparing passwords:", err);
            return cb(err);
          } else {
            if (valid) {
              return cb(null, user);
            } else {
              return cb(null, false);
            }
          }
        });
      } else {
        return cb("User not found");
      }
    } catch (err) {
      console.log(err);
    }
  })
);

passport.use(
  "google",
  new GoogleStrategy(
    {
      clientID: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      callbackURL: "http://localhost:3000/auth/google/secrets",
      userProfileURL: "https://www.googleapis.com/oauth2/v3/userinfo",
    },
    async (accessToken, refreshToken, profile, cb) => {
      try {
        const result = await db.query("SELECT * FROM users WHERE email = $1", [
          profile.email,
        ]);
        if (result.rows.length === 0) {
          const newUser = await db.query(
            "INSERT INTO users (email, password) VALUES ($1, $2)",
            [profile.email, "google"]
          );
          return cb(null, newUser.rows[0]);
        } else {
          return cb(null, result.rows[0]);
        }
      } catch (err) {
        return cb(err);
      }
    }
  )
);
passport.serializeUser((user, cb) => {
  cb(null, user);
});

passport.deserializeUser((user, cb) => {
  cb(null, user);
});

app.post("/return_stocks", async (req, res) => {
  const { item_code, quantity, rdate, reason } = req.body;

  try {
    // Start transaction
    await db.query("BEGIN");

    // Insert return record
    await db.query(
      "INSERT INTO return_stocks (item_code, quantity, return_date, reason) VALUES ($1, $2, $3, $4)",
      [item_code, quantity, rdate, reason]
    );

    // Update stock quantity in the `stocks` table
    await db.query(
      "UPDATE stocks SET quantity = quantity + $1 WHERE item_code = $2",
      [quantity, item_code]
    );

    // Commit transaction
    await db.query("COMMIT");

    res.redirect("/return_stocks");
  } catch (err) {
    // Rollback in case of an error
    await db.query("ROLLBACK");
    res.status(500).send("Error processing return: " + err.message);
  }
});

app.post("/update_stocks", async (req, res) => {
  const { item_code, quantity, update_date, price, discount } = req.body;

  try {
    // Step 1: Check if the item exists in the stocks table
    const result = await db.query("SELECT item_name, quantity FROM stocks WHERE item_code = $1", [item_code]);

    if (result.rows.length === 0) {
      // If the item is not found in the stocks table
      return res.status(404).send("Item not found.");
    }

    // Step 2: Get the current stock details
    const currentQuantity = result.rows[0].quantity;

    // Step 3: Log the update in the update_stocks table
    await db.query(
      "INSERT INTO update_stocks (item_code, quantity, price, discount, update_date) VALUES ($1, $2, $3, $4, $5)",
      [item_code, quantity, price, discount, update_date]
    );

    // Step 4: Update the quantity, price, and discount in the stocks table
    await db.query(
      "UPDATE stocks SET quantity = quantity + $1, price = $2, discount = $3 WHERE item_code = $4",
      [quantity, price, discount, item_code]
    );

    res.redirect("/update_stocks");  // Redirect back to the update_stocks page
  } catch (err) {
    console.error("Error updating stock:", err);
    res.status(500).send("Error updating stock: " + err.message);
  }
});

app.get('/outletdata_dashboard', async (req, res) => {
  try {
    // Queries for daily, weekly, and monthly data
    const endDate = '2024-01-25';

    const dailyQuery = `
      SELECT 
          TO_CHAR(created_at, 'YYYY-MM-DD') AS date,
          SUM(gross_amount) AS sales,
          SUM(profit) AS profit
      FROM invoice_header
      WHERE created_at BETWEEN '2024-01-18' AND '2024-01-25'
      GROUP BY date
      ORDER BY date ASC;
    `;

    const weeklyQuery = `
      SELECT 
          DATE_TRUNC('week', created_at) AS week,
          SUM(gross_amount) AS sales,
          SUM(profit) AS profit
      FROM invoice_header
      WHERE created_at <= '${endDate}' AND created_at >= '${endDate}'::DATE - INTERVAL '5 weeks'
      GROUP BY week
      ORDER BY week DESC;
    `;

    const monthlyQuery = `
      SELECT 
          DATE_TRUNC('month', created_at) AS month,
          SUM(gross_amount) AS sales,
          SUM(profit) AS profit
      FROM invoice_header
      WHERE created_at <= '${endDate}' AND created_at >= '${endDate}'::DATE - INTERVAL '5 months'
      GROUP BY month
      ORDER BY month DESC;
    `;

    // Query to count status values in cust_reviews table
    const pieChartQuery = `
      SELECT status, COUNT(*) AS count
      FROM cust_reviews
      GROUP BY status;
    `;

    // Fetch results
    const dailyResult = await db.query(dailyQuery);
    const weeklyResult = await db.query(weeklyQuery);
    const monthlyResult = await db.query(monthlyQuery);
    const pieChartResult = await db.query(pieChartQuery);

    // Format data for frontend
    const dailyChartData = [['Date', 'Sales', 'Profit']];
    dailyResult.rows.forEach(row => {
      dailyChartData.push([row.date, parseFloat(row.sales), parseFloat(row.profit)]);
    });

    const weeklyChartData = [['Week', 'Sales', 'Profit']];
    weeklyResult.rows.forEach(row => {
      weeklyChartData.push([row.week.toISOString().slice(0, 10), parseFloat(row.sales), parseFloat(row.profit)]);
    });

    const monthlyChartData = [['Month', 'Sales', 'Profit']];
    monthlyResult.rows.forEach(row => {
      monthlyChartData.push([row.month.toISOString().slice(0, 7), parseFloat(row.sales), parseFloat(row.profit)]);
    });

    // Pie chart data preparation
    const totalReviews = pieChartResult.rows.reduce((sum, row) => sum + parseInt(row.count), 0);
    const pieChartData = [['Status', 'Percentage']];
    pieChartResult.rows.forEach(row => {
      const percentage = (row.count / totalReviews) * 100;
      pieChartData.push([`Status ${row.status}`, percentage]);
    });

    // Pass data to the frontend
    res.render('outletdata_dashboard.ejs', {
      dailyChartData: JSON.stringify(dailyChartData),
      weeklyChartData: JSON.stringify(weeklyChartData),
      monthlyChartData: JSON.stringify(monthlyChartData),
      pieChartData: JSON.stringify(pieChartData)
    });

  } catch (error) {
    console.error('Error fetching data:', error);
    res.status(500).send('Server Error');
  }
});

app.post("/outletdata_dashboard/customer_reviews", async (req, res) => {
  console.log("Received Data:", req.body); // Debugging
  console.log("Request Body Type:", typeof req.body); // Check if parsed correctly

  let { customerID, feedback } = req.body; // Ensure customerID is extracted

  if (!customerID || !feedback) {
    return res.status(400).send("Customer ID and feedback are required.");
  }

  try {
    // Step 1: Send feedback to FastAPI for sentiment analysis
    let fastApiResponse = await fetch("http://127.0.0.1:8000/analyze/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer_id: customerID, feedback_text: feedback }) // ✅ FIXED
    });

    let fastApiData = await fastApiResponse.json();
    console.log("FastAPI Response:", fastApiData); // Debugging: Log API response

    if (!fastApiResponse.ok) {
      throw new Error(`FastAPI Error: ${fastApiData.detail || "Unknown error"}`);
    }

    let status = fastApiData.status; // Get sentiment result (0 = Neutral, 1 = Negative, 2 = Positive)

    // Step 2: Insert data into the database with sentiment status
    await db.query(
      "INSERT INTO cust_reviews (customer_id, feedback, status) VALUES ($1, $2, $3)",
      [customerID, feedback, status]
    );

    console.log("Review inserted successfully!");
    res.redirect("/outletdata_dashboard/customer_reviews");
  } catch (err) {
    console.error("Error:", err);
    res.status(500).send("Internal Server Error");
  }
});

app.get('/head-office-dashboard', async (req, res) => {
  const incomeQuery = `
      SELECT 
          TO_CHAR(sale_date, 'YYYY-MM') AS month,
          SUM(gross_income) AS total_gross_income
      FROM supermarket_sales
      GROUP BY month
      ORDER BY month ASC;
  `;

  const paymentQuery = `
      SELECT 
          payment, COUNT(*) AS count 
      FROM supermarket_sales 
      GROUP BY payment;
  `;

  const productLineQuery = `
      SELECT 
          product_line, COUNT(*) AS count 
      FROM supermarket_sales 
      GROUP BY product_line;
  `;

  const cityIncomeQuery = `
      SELECT 
          TO_CHAR(sale_date, 'YYYY-MM') AS month,
          city,
          SUM(gross_income) AS total_gross_income
      FROM supermarket_sales
      WHERE city IN ('Mandalay', 'Naypyitaw', 'Yangon')
      GROUP BY month, city
      ORDER BY month ASC;
  `;

  try {
      const incomeResult = await db.query(incomeQuery);
      const paymentResult = await db.query(paymentQuery);
      const productLineResult = await db.query(productLineQuery);
      const cityIncomeResult = await db.query(cityIncomeQuery);

      // Convert data for Google Charts
      const chartData = incomeResult.rows.map(row => [row.month, parseFloat(row.total_gross_income)]);
      const paymentData = paymentResult.rows.map(row => [row.payment, parseInt(row.count)]);
      const productLineData = productLineResult.rows.map(row => [row.product_line, parseInt(row.count)]);

      // Transform data for Area Chart (Group by City)
      let cityIncomeMap = {};
      cityIncomeResult.rows.forEach(row => {
          if (!cityIncomeMap[row.month]) {
              cityIncomeMap[row.month] = { Mandalay: 0, Naypyitaw: 0, Yangon: 0 };
          }
          cityIncomeMap[row.month][row.city] = parseFloat(row.total_gross_income);
      });

      const cityIncomeData = Object.entries(cityIncomeMap).map(([month, cities]) => [
          month,
          cities.Mandalay,
          cities.Naypyitaw,
          cities.Yangon
      ]);

      console.log("Chart Data Sent to Client:", chartData);
      console.log("Payment Data Sent to Client:", paymentData);
      console.log("Product Line Data Sent to Client:", productLineData);
      console.log("City Income Data Sent to Client:", cityIncomeData);

      // Render the EJS template with all datasets
      res.render('headoffice_dashboard', { 
          chartData: JSON.stringify(chartData),
          paymentData: JSON.stringify(paymentData),
          productLineData: JSON.stringify(productLineData),
          cityIncomeData: JSON.stringify(cityIncomeData)
      });

  } catch (err) {
      console.error("Database error:", err);
      res.status(500).send("Database error");
  }
});

app.get('/sales_pred', (req, res) => {
  res.render('sales_pred');
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});

