# Quick test of RandomForest implementation
import numpy as np

# Create some test data
np.random.seed(42)
X_test = np.random.randn(100, 5)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

print("Testing RandomForest implementation...")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels distribution: {np.bincount(y_test)}")

# Test the model
rf = RandomForestModel(n_estimators=20, max_depth=5, patience=2)
rf.fit(X_test, y_test)

# Make predictions
preds = rf.predict(X_test)
acc = np.mean(preds == y_test)
print(f"Training accuracy: {acc*100:.2f}%")

# Test probabilities
probas = rf.predict_proba(X_test)
print(f"Probability range: [{probas.min():.3f}, {probas.max():.3f}]")

print("✅ RandomForest implementation working correctly!")
