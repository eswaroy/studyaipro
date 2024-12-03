import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
df = pd.read_excel('test1.xlsx')
X = df[['Attendance (%)', 'Study Hours (per week)', 'Midterm 1 Score (%)', 'Midterm 2 Score (%)','Assignment Completion (%)']].values
y = df['Pass/Fail'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model and scaler
with open('trained_model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)
