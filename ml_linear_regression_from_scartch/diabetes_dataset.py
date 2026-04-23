import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LinearRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.m = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.m = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.m) + self.b

            error = y_pred - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.m -= self.lr * dw
            self.b -= self.lr * db

            loss = np.mean(error ** 2)
            self.losses.append(loss)

    def predict(self, X):
        return np.dot(X, self.m) + self.b


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegressionScratch(lr=0.01, epochs=1000)
model.fit(X_train, y_train)


predictions = model.predict(X_test)


mse = np.mean((y_test - predictions) ** 2)
print("Mean Squared Error:", mse)


plt.plot(model.losses)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()

plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Regression Plot")
plt.show()
# Loss curve plot
plt.plot(model.losses)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("MSE")

plt.plot(model.losses)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("MSE")


plt.plot(model.losses)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("MSE")

plt.savefig("ml_linear_regression_from_scratch/loss_curve.png")
plt.close()