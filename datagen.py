import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate random data
n_samples = 1000  

# Features
price = np.random.uniform(10, 100, n_samples)
marketing = np.random.uniform(1000, 5000, n_samples)
age = np.random.randint(1, 5, n_samples)

# Target variable: Units sold
# Assume sales depend on a linear combination of features + some noise
units_sold = 50 * price + 0.05 * marketing - 30 * age + np.random.normal(0, 50, n_samples)

# Create a DataFrame
data = pd.DataFrame({
    'Price': price,
    'Marketing': marketing,
    'Age': age,
    'Units_Sold': units_sold
})

# Save the DataFrame to a CSV file
data.to_csv('product_sales.csv', index=False)

print("CSV file generated successfully! Check the file 'product_sales.csv'.")
