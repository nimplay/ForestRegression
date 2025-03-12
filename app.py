import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Title of the app
st.title("Random Forest Regression App")
st.write("""
This app uses Random Forest Regression to predict product sales based on price, marketing budget, and product age.
""")

# Load the dataset
@st.cache_data  # Cache the data to improve performance
def load_data():
    data = pd.read_csv('product_sales.csv')
    return data

data = load_data()

# Display the dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Define features (X) and target variable (y)
X = data[['Price', 'Marketing', 'Age']]
y = data['Units_Sold']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
st.subheader("Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")

# Feature importance
st.subheader("Feature Importances")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
st.write("**Feature Importance Plot**")
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, palette='viridis')
ax.set_title("Feature Importances")
st.pyplot(fig)

# Scatter plots for feature relationships
st.subheader("Feature Relationships")
st.write("Scatter plots to visualize the relationship between features and units sold.")

# Plot for Price vs Units Sold
fig, ax = plt.subplots()
sns.scatterplot(x='Price', y='Units_Sold', data=data, ax=ax, color='blue')
ax.set_title("Price vs Units Sold")
st.pyplot(fig)

# Plot for Marketing vs Units Sold
fig, ax = plt.subplots()
sns.scatterplot(x='Marketing', y='Units_Sold', data=data, ax=ax, color='green')
ax.set_title("Marketing vs Units Sold")
st.pyplot(fig)

# Plot for Age vs Units Sold
fig, ax = plt.subplots()
sns.scatterplot(x='Age', y='Units_Sold', data=data, ax=ax, color='red')
ax.set_title("Age vs Units Sold")
st.pyplot(fig)

# Predictions vs Actual Values Plot
st.subheader("Predictions vs Actual Values")
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax, color='purple')
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Predictions vs Actual Values")
st.pyplot(fig)

# User input for prediction
st.subheader("Make a Prediction")
st.write("Enter the values for price, marketing budget, and product age to predict units sold.")

price = st.number_input("Price", min_value=10.0, max_value=100.0, value=50.0)
marketing = st.number_input("Marketing Budget", min_value=1000.0, max_value=5000.0, value=3000.0)
age = st.number_input("Product Age (years)", min_value=1, max_value=5, value=3)

# Predict button
if st.button("Predict"):
    input_data = np.array([[price, marketing, age]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Units Sold: {prediction[0]:.2f}")

# Footer
st.markdown("---")
st.markdown("**Nimrod Acosta 2025**")
