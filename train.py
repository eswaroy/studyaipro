# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import RandomOverSampler
# import pickle

# # Load the dataset
# df = pd.read_excel('test1.xlsx')

# # Convert Pass/Fail to binary labels
# df['Pass/Fail'] = (df['Pass/Fail'] == 'Pass').astype(int)

# # Split features and target
# X = df.drop(columns=['Pass/Fail']).values
# y = df['Pass/Fail'].values

# # Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Oversample to balance classes
# ros = RandomOverSampler()
# X_resampled, y_resampled = ros.fit_resample(X_scaled, y)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# # Define models
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# # Combine models using VotingClassifier
# ensemble = VotingClassifier(estimators=[
#     ('Random Forest', rf),
#     ('Gradient Boosting', gb)
# ], voting='soft')  # Use 'soft' for probabilities, 'hard' for majority voting

# # Train the ensemble model
# ensemble.fit(X_train, y_train)

# # Predictions
# y_pred = ensemble.predict(X_test)

# # Evaluate performance
# print("Classification Report for Combined Model:")
# print(classification_report(y_test, y_pred))
# with open('trained_model.pkl', 'wb') as f:
#     pickle.dump((df, scaler), f)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import pickle

# Load the dataset
df = pd.read_excel('test.xlsx')

# Convert Pass/Fail to binary labels
df['Pass/Fail'] = (df['Pass/Fail'] == 'Pass').astype(int)

# Split features and target
X = df.drop(columns=['Pass/Fail']).values
y = df['Pass/Fail'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Oversample to balance classes
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Combine models using VotingClassifier
ensemble = VotingClassifier(estimators=[('Random Forest', rf), ('Gradient Boosting', gb)], voting='soft')

# Train the ensemble model
ensemble.fit(X_train, y_train)

# Predictions
y_pred = ensemble.predict(X_test)

# Evaluate performance
print("Classification Report for Combined Model:")
print(classification_report(y_test, y_pred))

# Save the trained model and scaler
with open('trained_model.pkl', 'wb') as f:
    pickle.dump((ensemble, scaler), f)
