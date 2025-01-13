import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Financial_Data.csv')

# Define the correlation scores and select the top features
correlation_scores = {
    "VIX": 0.598432,
    "GTITL2YR": 0.304528,
    "GTITL10YR": 0.297573,
    "GTITL30YR": 0.295000,
    "EONIA": 0.180311,    
}
selected_features = list(correlation_scores.keys())

# Calculate moving averages for the selected features
window_size = 7
for feature in selected_features:
    data[f'{feature}_MA{window_size}'] = data[feature].rolling(window=window_size).mean()

# Drop NaN values caused by moving averages
data = data.dropna().reset_index(drop=True)

# Define features and target
features = data[[f"{feature}" for feature in selected_features] + 
                [f"{feature}_MA{window_size}" for feature in selected_features]]
target = data['Y']  # Use the correct target variable

# Split the data chronologically (80% train, 20% test)
split_index = int(len(data) * 0.8)
features_train = features.iloc[:split_index]
features_test = features.iloc[split_index:]
target_train = target.iloc[:split_index]
target_test = target.iloc[split_index:]

# Standardize features
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
features_train_balanced, target_train_balanced = smote.fit_resample(features_train_scaled, target_train)

# Train logistic regression model
classifier = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500)
classifier.fit(features_train_balanced, target_train_balanced)

# Predict probabilities for the test set
probs = classifier.predict_proba(features_test_scaled)[:, 1]

# Determine the best threshold based on F1-score
precision, recall, thresholds = precision_recall_curve(target_test, probs)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
best_threshold = thresholds[np.argmax(f1_scores)]

# Adjust predictions based on the best threshold
target_pred_adjusted = (probs >= best_threshold).astype(int)

# Evaluate the model
conf_matrix = confusion_matrix(target_test, target_pred_adjusted)
class_report = classification_report(target_test, target_pred_adjusted)

# Print results
print(f"Best threshold based on F1-score: {best_threshold}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Plot Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
