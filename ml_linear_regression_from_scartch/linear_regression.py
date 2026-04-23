import numpy as np
import matplotlib.pyplot as plt
X = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])  # y = 2x
class LinearRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.m = 0
        self.b = 0
        self.losses = []

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            y_pred = self.m * X + self.b
            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)
            self.m -= self.lr * dm
            self.b -= self.lr * db
            loss = np.mean((y - y_pred)**2)
            self.losses.append(loss)

    def predict(self, X):
        return self.m * X + self.b

model = LinearRegressionScratch(lr=0.01, epochs=1000)
model.fit(X, y)
print("Slope:", model.m, "Intercept:", model.b)

plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.title("Linear Regression Scratch")
plt.show()


plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.title("Linear Regression Scratch")



plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.title("Linear Regression Scratch")


plt.savefig("ml_linear_regression_from_scratch/regression_plot.png")
plt.close()

