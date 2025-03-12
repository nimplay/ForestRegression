# Project: Random Forest Regression App

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

This is a Streamlit-based web application that uses Random Forest Regression to predict product sales based on features like price, marketing budget, and product age. The app also includes visualizations to help users understand the data and model performance.

---

## **Features**

- Dataset Preview: Displays the first few rows of the dataset.

- Model Performance: Shows the Mean Squared Error (MSE) of the Random Forest Regression model.

- Feature Importance Plot: Visualizes the importance of each feature in the model.

- Scatter Plots: Shows the relationship between each feature (Price, Marketing, Age) and the target variable (Units_Sold).

- Predictions vs Actual Values Plot: Compares the model's predictions with the actual values from the test set.

- Prediction Interface: Allows users to input values for price, marketing budget, and product age to get a prediction for units sold.
---
## **Requirements**

To run this app, you need the following Python libraries:

 - streamlit

 - pandas

 - numpy

 - scikit-learn

 - matplotlib

 - seaborn

You can install the required libraries using the following command:

  pip install streamlit pandas numpy scikit-learn matplotlib seaborn

---

## How to Run the App
1- Clone this repository or download the app.py file.

2- Navigate to the folder where app.py is located.

3- Run the app using the following command: streamlit run app.py

4- The app will open in your browser. You can interact with it as follows:

   - View the dataset and model performance.

   - Explore feature importance and scatter plots.

   - Input values for price, marketing budget, and product age to get a prediction for units sold.

## Dataset

The dataset used in this app is synthetic and generated randomly. It contains the following columns:

   - Price: The price of the product (between 10 and 100).

   - Marketing: The marketing budget allocated for the product (between 1000 and 5000).

   - Age: The age of the product in years (between 1 and 5).

   - Units_Sold: The number of units sold (target variable).

The dataset is saved as product_sales.csv.


## Code Structure
app.py: The main Streamlit application file.

   - Loads the dataset.

   - Trains a Random Forest Regression model.

   - Displays visualizations and allows user input for predictions.


## Author
   -  Nimrod Acosta

   - Email: nimrod7day@gimail.com

   - GitHub: [nimplay](https://github.com/nimplay)

   - Linkedin: [nimrod-acosta](https://www.linkedin.com/in/nimrod-acosta/)

Thanks for using this app! 🚀

Year: 2025

# Live Version


