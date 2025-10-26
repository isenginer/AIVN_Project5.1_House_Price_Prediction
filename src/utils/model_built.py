import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LRRegularizationVectorized:

    def __init__(self,
                 method = "gradient",
                 learning_rate = 0.01,
                 n_iterators = 500,
                 verbose=False,
                 fit_intercept=True,
                 random_state = 42,
                 alpha = 0.0, # for RidgeRegression, 0.0 is not applicable
                 tolerance = 1e-4 # the expected deviation at last for gradient
                 ):
        """
        Linear Regression với vectorized operations - Implementation from scratch

        Features:
        - Vectorized computations cho performance cao
        - Multiple solving methods (Normal Equation, Gradient Descent)
        - Built-in metrics và visualization
        - Regularization support (Ridge and Lasso)
        - Scikit-learn compatible API

        :param method: choose method for model train: normal equation, gradient descent or regularization
        :param learning_rate: learning rate for gradient descent
        :param n_iterators: the iterators of training data
        :param verbose: True/False for verbose output
        :param fit_intercept: apply bias for model train
        :param random_state: random seed for splitting
        :param alpha: Parameter for regularization of Ridge , 0.0 is default value if not applicable
        :param tolerance: limit tolerance for gradient descent
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterators = n_iterators
        self.verbose = verbose
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.alpha = alpha # using this parameter for RidgeRegression
        self.tolerance = tolerance

        self.coef_ = None
        self.intercept_ = None
        self.cost_history_ = [] # saving loss values
        self.n_features_ = None
        self.n_samples_ = None

        if random_state is not None:
            self.random_state = np.random.seed(random_state)

    def _add_intercept(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def _prepare_data(self, X: np.ndarray, y: np.ndarray=None):
        """
        Prepare data for model training, X, y shall be numpy array for calculation
        :return: X numpy array and y numpy array if not None
        """
        # Convert X, y to array
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)

        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fit_intercept:
            X = self._add_intercept(X)

        if y is not None:
            return X, y
        return X

    def _normal_equation(self, X: np.ndarray, y: np.ndarray):
        """
        Linear Regression using normal equation with regularization
        Note that this method is standard of "Closed-form equation" of Linear Regression
        """
        # Regulation term
        try:
            XTX = X.T @ X
            n = X.shape[0] # len of XTX
            if self.alpha > 0:
                reg_matrix = self.alpha * np.eye(XTX.shape[0]) # np.eye - diagonal matrix with one (create diagonal matrix with alpha)
                if self.fit_intercept:
                    reg_matrix[0, 0] = 0
                XTX += reg_matrix

            # Calculate theta
            theta = np.linalg.inv(XTX) @ X.T @ y
            return theta
        except np.linalg.LinAlgError:
            warnings.warn("Matrix is singular matrix, using pseudo inverse") # sử dụng giả nghịch đảo
            return np.linalg.pinv(X) @ y

    def _gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """
        Gradient descent with regularization, in this case we use Batch GD
        Note:
        SGD: Sample < 1000
        Batch GD: Sample 1000 - 100,000
        Mini Batch GD: Sample > 100,000
        :return:
        optimum theta for GD
        """
        m, n = X.shape
        theta = np.random.randn(n) * 0.01
        self.cost_history = []

        for i in range(self.n_iterators): # note from 1 because the intercept is at location 1
            # Compute the output
            y_hat = X @ theta
            errors = y_hat - y

            # Compute the cost with Ridge Regulation losses^2
            # If use Lasso Regulation, the losses = |losses|
            cost = (1/(2*m)) * np.sum(errors**2)
            if self.alpha > 0:
                l2_penalty = self.alpha * np.sum(theta[1:]**2) if self.fit_intercept else self.alpha * np.sum(theta**2)
                cost += l2_penalty

            self.cost_history_.append(cost)

            # Calculate the gradient
            gradients = (1/m) * X.T @ errors
            # Compensate the gradient by regulator reg_gradient
            if self.alpha > 0:
                # Calculate the regulated gradients, create matrix 0 with dim = dim(theta)
                reg_gradient = np.zeros_like(theta)
                if self.fit_intercept:
                    reg_gradient[1:] = self.alpha * theta[1:]
                else:
                    reg_gradient = self.alpha * theta
                gradients += reg_gradient # gradient after regulating

            theta -= self.learning_rate * gradients # theta = theta - lr* {gradient after regulating}
            # print(f"fitting at iteration {i}, cost = {cost}, theta = {theta}")
            # Check convergence of loop based on the tolerance, if reached then break the iterators
            if i > 0 and abs(self.cost_history_[-2] - self.cost_history_[-1]) < self.tolerance:
                print(f"⚠️ Converged at iteration {i+1} iterations")
                break

        return theta

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit linear regression model
        Note that fit method always return the `self` of X and y for transformation
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        X_array, y_array = self._prepare_data(X, y)
        self.n_samples_, self.n_features_ = X_array.shape

        if self.fit_intercept:
            self.n_features_ -= 1 # remove intercept or bias of features (by first value adding at 1)

        try:
            if self.method == "normal":
                theta = self._normal_equation(X_array, y_array)
            if self.method == "gradient":
                theta = self._gradient_descent(X_array, y_array)
        except ValueError:
            print(f"❌ Unknown method: {self.method}")

        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_= 0.0
            self.coef_ = theta
        print("✅ Fitting data completion!")
        return self

    def predict(self, X:np.ndarray):
        if self.coef_ is None:
            raise ValueError("❌ Model not fitted, please fit before predicting")

        X_array = self._prepare_data(X)
        # to calulate the result, remove the intercept if available
        if self.fit_intercept:
            X_features = X_array[:, 1:]
            return X_features @ self.coef_ + self.intercept_
        else:
            return X_array @ self.coef_

    def score(self, X:np.ndarray, y:np.ndarray):
        """
        Calculate score of model R^2
        :param X:
        :param y:
        :return:
        """
        y_pred = self.predict(X)
        sum_of_squre_error = np.sum((y_pred - y) ** 2)
        sum_of_square_total = np.sum((y - np.mean(y)) ** 2)
        return 1 - (sum_of_squre_error/sum_of_square_total)

    def mean_squared_error(self, X:np.ndarray, y:np.ndarray):
        y_pred = self.predict(X)
        return np.mean((y_pred - y)**2)

    def plot_cost_history(self) -> None:
        """Plot cost history for gradient descent"""
        if not self.cost_history_:
            print("No cost history available. Use method='gradient_descent' to track cost.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history_)
        plt.title('Cost Function During Training')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()

    def summary(self) -> None:
        """Print model summary"""
        if self.coef_ is None:
            print("Model not fitted yet.")
            return

        print("📊 LINEAR REGRESSION MODEL SUMMARY")
        print("=" * 50)
        print(f"Method: {self.method}")
        print(f"Features: {self.n_features_}")
        print(f"Samples: {self.n_samples_}")
        print(f"Fit Intercept: {self.fit_intercept}")

        if self.alpha > 0:
            print(f"Regularization (α): {self.alpha}")

        print(f"\nIntercept: {self.intercept_:.6f}")
        print("Coefficients:")
        for i, coef in enumerate(self.coef_):
            print(f"  Feature {i + 1}: {coef:.6f}")

        if self.cost_history_:
            print(f"\nFinal Cost: {self.cost_history_[-1]:.6f}")
            print(f"Iterations: {len(self.cost_history_)}")

        print("=" * 50)


if __name__ == "__main__":
    data = np.random.randn(100, 4)
    X = data[:, :-1]
    y = data[:, -1]
    LRR = LRRegularizationVectorized(
        method="gradient",
        n_iterators=500,
        alpha=0.009,
    )
    LRR.fit(X, y)

    y_pred = LRR.predict(X)
    LRR.summary()
    print(LRR.score(X, y))