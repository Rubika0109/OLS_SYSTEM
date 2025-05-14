import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
df = pd.read_csv(r"C:\project\ols-regression-challenge\capped_data.csv")  # Replace with your actual file

# Replace with actual features and target from the dataset
features = ['popest2015', 'pctpubliccoveragealone', 'pctempprivcoverage', 'state_ District of Columbia', 'pctmarriedhouseholds', 'pctprivatecoveragealone', 'avgdeathsperyear', 'lower_bound', 'pctblack', 'upper_bound', 'median', 'pctprivatecoverage', 'medianagefemale']
target = 'target_deathrate'  # <-- Replace with the actual target column name

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model to file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")
