#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#loading dataset
data = pd.read_csv('cdd.csv')

#splitting data
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Retrain the model after scaling
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Taking input from the user
print("Please enter values for the features to predict the class:")
feature_names = X.columns.tolist()
user_input = []
for feature in feature_names:
    value = float(input(f"Enter value for {feature}: "))
    user_input.append(value)

# Apply same scaling to user input
user_input_scaled = scaler.transform([user_input])

# Predicting using the trained model
predicted_class = random_forest_model.predict(user_input_scaled)
print(f"\nThe predicted class is: {predicted_class[0]}")
if predicted_class[0] == 1:
    print("⚠️ This transaction is Fraudulent.")
else:
    print("✅ This transaction is Not Fraudulent.")

