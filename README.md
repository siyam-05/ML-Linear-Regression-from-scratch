#  ML Linear Regression from Scratch

This project implements **Linear Regression** using **NumPy** (without using scikit-learn’s regression model).
It applies the model to the **Diabetes dataset** from scikit-learn to predict disease progression.

The main goal is to understand the fundamentals of:

* Gradient Descent
* Matrix operations
* Model evaluation



##  Features

* Linear Regression implemented from scratch
* Gradient Descent optimization
* Evaluation using Mean Squared Error (MSE)
* Dataset: Diabetes (scikit-learn)
* Visual outputs:

  * Loss Curve
  * Regression Plot



##  Project Structure



  ml-linear-regression-from-scratch

## Project Structure

```text
ml-linear-regression-from-scratch
├── src/
│   ├── __init__.py
│   ├── linear_regression.py
│   └── diabetes_dataset.py
├── outputs/
├── notebooks/
├── tests/
├── requirements.txt
├── README.md
└── setup.py
```


##  Requirements

* Python 3.x
* NumPy
* scikit-learn
* Matplotlib
* Jupyter Notebook (optional)

Install dependencies:

bash 
pip install -r requirements.txt




##  Usage

Run the project:


 python diabetes_dataset.py




##  Output

* Console:


Mean Squared Error: ~2800–3000


* Generated plots (saved in `outputs/`):

  * `loss_curve.png` → shows training convergence
  * `regression_plot.png` → predicted vs actual values



##  Notes

* This project is built for **learning purposes**
* Focus is on understanding the **math behind linear regression**
* No high-level ML models are used



##  Future Improvements

* Add Regularization (Ridge / Lasso)
* Compare with scikit-learn model
* Try Polynomial Regression
* Use additional datasets



##  License

This project is licensed under the **MIT License**.



##  Author

MD.Tanvir Hasan Siyam

Machine Learning Learner

