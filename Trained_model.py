import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
# Load the data
data = pd.read_csv('Financial_Data.csv')

#1, LOGISTIC REGRESSION MODEL 

# Define the correlation scores and select the top features
correlation_scores = {
    "VIX": 0.598432,
    "GTITL2YR": 0.304528,
    "GTITL10YR": 0.297573,
    "GTITL30YR": 0.295000,
    "EONIA": 0.180311,
    "XAU BGNL": -0.011535,
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
split_index = int(len(data) * 0.9)
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
classifier_main = LogisticRegression(random_state=0, class_weight='balanced', C = 0.65, max_iter=1000,solver = "liblinear")
classifier_main.fit(features_train_balanced, target_train_balanced)

# Predict probabilities for the test set
probs = classifier_main.predict_proba(features_test_scaled)[:, 1]

# Determine the best threshold based on F1-score
precision, recall, thresholds = precision_recall_curve(target_test, probs)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
best_threshold = thresholds[np.argmax(f1_scores)]
#0.488599511390968

# Adjust predictions based on the best threshold
target_pred_adjusted = (probs >= best_threshold).astype(int)


#2,XGBOOST model

# Define the correlation scores and select the top features
correlation_scores = {
    "VIX": 0.598432,
    "GTITL30YR": 0.295000,
    "XAU BGNL": -0.011535,
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
split_index = int(len(data) * 0.7)
features_train = features.iloc[:split_index]
features_test = features.iloc[split_index:]
target_train = target.iloc[:split_index]
target_test = target.iloc[split_index:]

# Standardize features
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

smote = SMOTE(random_state=42)
features_train_balanced, target_train_balanced = smote.fit_resample(features_train_scaled, target_train)

negative_class = np.sum(target_train_balanced == 0)
positive_class = np.sum(target_train_balanced == 1)
ratio = negative_class / positive_class

# Train XGBoost model
xgb_model = XGBClassifier(
    objective='binary:logistic',
    max_depth= 5,
    learning_rate=0.1,
    n_estimators=100,
    alpha=10,
    scale_pos_weight=ratio,
    random_state=42
)
xgb_model.fit(features_train_balanced, target_train_balanced)

# Predict probabilities for the test set
probs = xgb_model.predict_proba(features_test_scaled)[:, 1]

# Determine the best threshold based on F1-score
precision, recall, thresholds = precision_recall_curve(target_test, probs)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
best_threshold = 0.7
thresholds[np.argmax(f1_scores)]

# Adjust predictions based on the best threshold
target_pred_adjusted = (probs >= best_threshold).astype(int)