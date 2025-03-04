from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters
n_trees = 10
max_depth = None
max_features = None

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, max_features=max_features, random_state=42)

# Train the model
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"accuracy: {accuracy * 100:.2f}")
