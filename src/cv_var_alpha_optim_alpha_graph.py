import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

import matplotlib.pyplot as plt

from utils import XyScaler

def rss(y, y_hat):
    return np.mean((y  - y_hat)**2)

def cv(X, y, base_estimator, n_folds, random_seed=154):
    """Estimate the in and out-of-sample error of a model using cross validation.
    
    Parameters
    ----------
    
    X: np.array
      Matrix of predictors.
      
    y: np.array
      Target array.
      
    base_estimator: sklearn model object.
      The estimator to fit.  Must have fit and predict methods.
      
    n_folds: int
      The number of folds in the cross validation.
      
    random_seed: int
      A seed for the random number generator, for repeatability.
    
    Returns
    -------
      
    train_cv_errors, test_cv_errors: tuple of arrays
      The training and testing errors for each fold of cross validation.
    """
    kf = KFold(n_splits=n_folds, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X)):
        # Split into train and test
        X_cv_train, y_cv_train = X[train], y[train]
        X_cv_test, y_cv_test = X[test], y[test]
        # Standardize data.
        standardizer = XyScaler()
        standardizer.fit(X_cv_train, y_cv_train)
        X_cv_train_std, y_cv_train_std = standardizer.transform(X_cv_train, y_cv_train)
        X_cv_test_std, y_cv_test_std = standardizer.transform(X_cv_test, y_cv_test)
        # Fit estimator
        estimator = clone(base_estimator)
        estimator.fit(X_cv_train_std, y_cv_train_std)
        # Measure performance
        y_hat_train = estimator.predict(X_cv_train_std)
        y_hat_test = estimator.predict(X_cv_test_std)
        # Calclate the error metrics
        train_cv_errors[idx] = rss(y_cv_train_std, y_hat_train)
        test_cv_errors[idx] = rss(y_cv_test_std, y_hat_test)
    return train_cv_errors, test_cv_errors

def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs):
    """Train a regularized regression model using cross validation at various values of alpha.
    
    Parameters
    ----------
    
    X: np.array
      Matrix of predictors.
      
    y: np.array
      Target array.
      
    model: sklearn model class
      A class in sklearn that can be used to create a regularized regression object.  Options are `Ridge` and `Lasso`.
      
    alphas: numpy array
      An array of regularization parameters.
      
    n_folds: int
      Number of cross validation folds.
      
    Returns
    -------
    
    cv_errors_train, cv_errors_test: tuple of DataFrame
      DataFrames containing the training and testing errors for each value of 
      alpha and each cross validation fold.  Each row represents a CV fold,
      and each column a value of alpha.
    """
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                    columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                    columns=alphas)
    for alpha in alphas:
        train_fold_errors, test_fold_errors = cv(X, y, model(alpha=alpha, **kwargs), n_folds=n_folds)
        cv_errors_train.loc[:, alpha] = train_fold_errors
        cv_errors_test.loc[:, alpha] = test_fold_errors
    return cv_errors_train, cv_errors_test

def get_optimal_alpha(mean_cv_errors_test):
    alphas = mean_cv_errors_test.index
    optimal_idx = np.argmin(mean_cv_errors_test.values)
    optimal_alpha = alphas[optimal_idx]
    return optimal_alpha


if __name__ == '__main__':
    # Before running any code below the next comment
    # we must import data and create X, y
    equip_train = pd.read_csv('../data/train_clean.csv')
    equip_test = pd.read_csv('../data/test_clean.csv')

    scat_matrx = scatter_matrix(equip_train, figsize=(20, 20))
    fig, ax = plt.subplots()
    equip_hist = ax.hist(equip_train['SalePrice'], bins=50)

    target_column = 'SalePrice'

    X_train, y_train = equip_train.drop(target_column, axis=1), 
                    equip_train[target_column]
    X_test, y_test = equip_test.drop(target_column, axis=1),
                    eqiup_test[target_column]

    ## FOR RIDGE REGRESSION ------------------------------
    # Calling the cross validation function - for RIDGE
    n_folds = 10
    train_cv_errors, test_cv_errors = cv(X_train.values, y_train.values, 
                                    Ridge(alpha=0.5), n_folds=n_folds)

    print(f'Cross Validation training error: {train_cv_errors}, test error: {test_cv_errors}')

    '''
    we need to re-assign X_train, y_train, X_test and y_test after crossvalidation
    '''

    # Calling the various alphas function - for RIDGE
    ridge_alphas = np.logspace(-2, 4, num=250)

    ridge_cv_errors_train, ridge_cv_errors_test = train_at_various_alphas(
    X_train.values, y_train.values, Ridge, ridge_alphas)

    # Calling for the optimal Alpha - for RIDGE
    ridge_mean_cv_errors_train = ridge_cv_errors_train.mean(axis=0)
    ridge_mean_cv_errors_test = ridge_cv_errors_test.mean(axis=0)

    ridge_optimal_alpha = get_optimal_alpha(ridge_mean_cv_errors_test)

    # Plot the Regression Train and Test MSE - for RIDGE

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(np.log10(ridge_alphas), ridge_mean_cv_errors_train)
    ax.plot(np.log10(ridge_alphas), ridge_mean_cv_errors_test)
    ax.axvline(np.log10(ridge_optimal_alpha), color='grey')
    ax.set_title("Ridge Regression Train and Test MSE")
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_ylabel("MSE")

    ## FOR LASSO REGRESSION ------------------------------

    lasso_alphas = np.logspace(-3, 1, num=250)
    # Calling the various alpha function - for LASSO
    lasso_cv_errors_train, lasso_cv_errors_test = train_at_various_alphas(
        X_train.values, y_train.values, Lasso, lasso_alphas, max_iter=5000)

    # Calling for the optimal Alpha - for LASSO
    lasso_mean_cv_errors_train = lasso_cv_errors_train.mean(axis=0)
    lasso_mean_cv_errors_test = lasso_cv_errors_test.mean(axis=0)

    lasso_optimal_alpha = get_optimal_alpha(lasso_mean_cv_errors_test)

    # Plot the Regression Train and Test MSE - for LASSO
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.log10(lasso_alphas), lasso_mean_cv_errors_train)
    ax.plot(np.log10(lasso_alphas), lasso_mean_cv_errors_test)
    ax.axvline(np.log10(lasso_optimal_alpha), color='grey')
    ax.set_title("LASSO Regression Train and Test MSE")
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_ylabel("MSE")

    plt.show()

    
