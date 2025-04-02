import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = pd.read_csv("Advertising.csv")
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("best_random_forest_model.pkl", "wb") as file:
    pickle.dump(model, file)
