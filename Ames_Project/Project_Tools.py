import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import math
import copy




'''

Functions implemented throughout all analyses.

***Note*** The order of these functions generally reflect how they were implemented throughout the analyses. The functions below are organized by:


Functions explicitly implemented in the finished analyses:

    Model Training Functions
        - Functions for Training with Gradient Descent
        - Functions for Training with Tree-Based Methods

    Visualization Functions:

    Other Functions:


Functions not explicitly implemented in the finished analyses:

    Model Training Functions
        - Functions for Training with Gradient Descent
        - Functions for Training with Tree-Based Methods

    Visualization Functions:

'''





'''

Functions explicitly implemented in the finished analyses:

'''

    
    
'''

Model Training Functions:

'''
    

'Functions for Training with Gradient Descent.'

def compute_regularized_cost(X, y, w, b, lambda_):
    """
    Computes the regularized squared error cost function over all examples with complete vectorization.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w: Weight parameters(numpy.ndarray).  
    - b: Bias parameter(scalar).
    - lambda_: Controls strength of regularization(scalar).
    Returns:
    - total_cost: Value of the regularized squared error cost function over all examples(scalar).
    """

    m = len(y)
    
    #Compute predictions.
    predictions = np.dot(X, w) + b
    
    #Compute the squared error cost term.
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    
    #Compute the regularization term. 
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    
    #Sum these terms. 
    total_cost = cost + reg_cost
    return total_cost





 
def compute_regularized_gradient(X, y, w, b, lambda_): 
    """
    Computes the regularized gradient for each weight parameter and the unregularized gradient for the bias parameter with complete vectorization. 
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w: Weight parameters(ndarray).  
    - b: Bias parameter(scalar).
    - lambda_: Controls strength of regularization(scalar).
    Returns:
      dj_db: The gradient of the cost with respect to the bias parameter(scalar). 
      dj_dw: The gradient of the cost with respect to each weight parameter(numpy.ndarray).  
    """
    m = X.shape[0]
    predictions = np.dot(X, w) + b
    error = predictions - y
    
    #Regularized gradient for weights.
    dj_dw = (1 / m) * (np.dot(X.T, error)) + (lambda_ / m) * w
    
    #Unregularized gradient for bias.
    dj_db = (1 / m) * np.sum(error)
    
    return dj_db, dj_dw




def gradient_descent_reg_with_convergence_check(X, y, w, b, cost_function, gradient_function, alpha, lambda_, num_iters, tol): 
    """
    Performs gradient descent to learn the weight(w) and bias(b) parameters. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha and regualrization term lambda_. 
    Gradient descent stops when parameter gradients converge pass threshold or prints warning if there was not convergence. 
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w: Initial values of weight parameters(numpy.ndarray).
    - b: Initial value of parameter(scaler).
    - cost_function: Function to compute regularized cost.
    - gradient_function: Function to compute regularized gradients.
    - alpha: Learning rate(scalar).
    - lambda_: Controls strength of regularization(scalar).
    - num_iters: Number of iterations for gradient descent(scalar).
    - tol: Convergence threshold(scalar).
    Returns:
    - w: Updated values of weight parameters after running gradient descent or until convergence if applicable(numpy.ndarray).
    - b: Updated value of bias parameter after running gradient descent or until convergence if applicable(scalar).
    - J_history: List of cost function outputs across training intervals.
    """
    
    #Number of training examples.
    m = X.shape[0]
    
    #An array to store cost J at each iteration primarily for graphing later.
    J_history = []
    
    w = copy.deepcopy(w) #Avoid modifying global w within function.
    
    
    save_interval = np.ceil(num_iters/100) #Prevent resource exhaustion for long runs.

    for i in range(num_iters):

        #Compute the gradient for each parameter.
        dj_db,dj_dw = gradient_function(X, y, w, b, lambda_)   

        #Update Parameters using w, b, alpha and gradient.
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db  
        
        #Compute both gradient norms for convergence checking.
        grad_norm_w = np.linalg.norm(dj_dw)
        grad_norm_b = np.abs(dj_db)  

        #Save regularized squared error cost J at intervals.
        if i == 0 or i % save_interval == 0:
            J_history.append(cost_function(X, y, w , b, lambda_))

        #Print regularized squared error cost, mse, gradients, and parameter values 10 times throughout training. 
        if i% math.ceil(num_iters/10) == 0:
            y_pred = np.dot(X, w) + b
            mse = np.mean((y - y_pred) ** 2)
            print(f"Iteration {i:4}: Cost {cost_function(X, y, w , b, lambda_):0.2e} MSE {mse:0.2e} ",
                  f"dj_dw: {dj_dw}, dj_db: {dj_db}  ",
                  f"w: {w}, b:{b}")
        #Stop if both gradients are very small (i.e., convergence)
        if grad_norm_w < tol and grad_norm_b < tol:
            print(f"Converged at iteration {i}")
            break
            
    #Print warning if gradients did not converge.
    if grad_norm_w >= tol or grad_norm_b >= tol:
        print(f"Warning: Did not converge within {num_iters} iterations")
    
    return w, b, J_history    #Return w,b and history for graphing





def gradient_descent_reg_with_convergence_check_for_tracking(X, y, w, b, gradient_function, alpha, lambda_reg, num_iter, tol):
    """
    Performs gradient descent to learn the weight(w) and bias(b) parameters. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha and regualrization term lambda_. 
    Returns the updated w_new and b_new paramters when converged and True further indicating convergence. If the w and b paramters did not converge returns those updated parameters and False further indicating non-convergence.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w: Initial values of weight parameters(numpy.ndarray).
    - b: Initial value of parameter(scaler).
    - gradient_function: Function to compute regularized gradients.
    - alpha: Learning rate(scalar).
    - lambda_reg: Controls strength of regularization(scalar).
    - num_iter: Number of iterations for gradient descent(scalar).
    - tol: Convergence threshold(scalar).
    Returns:
    - w_new: Updated values of weight parameters at convergence(numpy.ndarray).
    - b_new: Updated value of bias parameter at convergence(scalar).
    - True: Further indicating convergence.
    
    or
    
    - w: Updated values of weight parameters after running gradient descent which did not converge(numpy.ndarray).
    - b: Updated value of bias parameter after running gradient descent which did not converge(scalar).
    - False: Further indicating non-convergence.
    """
    
    w = copy.deepcopy(w) #Avoid modifying global w within function.
    b = b
    
    for i in range(num_iter):
        # Compute gradients
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_reg)
        
        #Update parameters.
        w_new = w - alpha * dj_dw
        b_new = b - alpha * dj_db

        #Check for convergence based on the norm of the gradients.
        if np.linalg.norm(dj_dw) < tol and abs(dj_db) < tol:
            print(f"Converged at iteration {i}")
            return w_new, b_new, True  #Return True indicating convergence.
        
        #Update the parameters for the next iteration.
        w, b = w_new, b_new
    
    #If not converged.
    print("Did not converge within iteration limit")
    return w, b, False  # Return False indicating no convergence





def k_fold_cross_validation_with_scaler(X, y, w_int, b_int, scaler_function, cost_function, gradient_function, gradient_descent_function, alpha, num_iters, lambda_, k, tol):
    """
    Performs gradient descent within K-fold cross validation with one scaling method. The gradient descent function used in this function should have convergence checking functionality.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_int: Initial values of weight parameters(numpy.ndarray).
    - b_int: Initial value of parameter(scaler).
    - scaler_function: Scaler class to initialize scaler instance for feature scaling.
    - cost_function: Function to compute regularized cost.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    - alpha: Learning rate(scalar).
    - num_iters: Number of iterations for gradient descent(scalar).
    - lambda_: Controls strength of regularization(scalar).
    - k: Number of folds for cross validation.
    - tol: Convergence threshold(scalar).
    Returns:
    - best_w: Updated values of weight parameters after running gradient descent or until convergence if applicable(numpy.ndarray).
    - best_b: Updated value of bias parameter after running gradient descent or until convergence if applicable(scalar).
    - fold_mses: List of mean squared error values from each fold.
    - J_history: List of cost function outputs across training intervals.
    """
    
    #K-Folds with shuffling and random seed.
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    #Initialize scaler function and lists.
    scaler = scaler_function()
    fold_mses = []
    fold_params = []
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        
        #Split the data into train and test for this fold.
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Apply RobustScaler to the training data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # Scale test data using the same scaler

        #Perform gradient descent on the scaled training data
        w, b, J_history = gradient_descent_function(X_train_scaled, y_train, w_int, b_int, cost_function, gradient_function, alpha, lambda_, num_iters, tol)
        
        
        #Make predictions on the test set. 
        y_predicted = np.dot(X_test_scaled, w) + b 

        #Compute the mean squared error on the test set.  
        mse = np.mean((y_test - y_predicted) ** 2)
        fold_mses.append(mse)
        fold_params.append((w, b))  #Store the parameters after convergence for this fold.

        print(f"Fold {fold_idx+1}, MSE: {mse:.4f}")
    
    #Find the fold with the lowest MSE
    best_fold_idx = np.argmin(fold_mses)
    best_w, best_b = fold_params[best_fold_idx]
    
    print(f"Average MSE across all folds: {np.mean(fold_mses)}")
    print(f"Best Fold: {best_fold_idx + 1}, Best MSE: {fold_mses[best_fold_idx]:.4f}")
    print(f"Best parameters: w = {best_w}, b = {best_b}")

    return best_w, best_b, fold_mses, J_history




def k_fold_cross_validation_with_combined_scaler(X, y, w_int, b_int, robust_scaler_function, minmax_scaler_function, cost_function, gradient_function, gradient_descent_function, alpha, num_iters, lambda_, k, tol):
    """
    Performs gradient descent within K-fold cross validation with combined scaling methods. The gradient descent function used in this function should have convergence checking functionality.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_int: Initial values of weight parameters(numpy.ndarray).
    - b_int: Initial value of parameter(scaler).
    - robust_scaler_function: RobustScaler class to initialize scaler instance for feature scaling.
    - minmax_scaler_function: MinMaxScaler class to initialize scaler instance for feature scaling.
    - cost_function: Function to compute regularized cost.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    - alpha: Learning rate(scalar).
    - num_iters: Number of iterations for gradient descent(scalar).
    - lambda_: Controls strength of regularization(scalar).
    - k: Number of folds for cross validation.
    - tol: Convergence threshold(scalar).
    Returns:
    - best_w: Updated values of weight parameters after running gradient descent or until convergence if applicable(numpy.ndarray).
    - best_b: Updated value of bias parameter after running gradient descent or until convergence if applicable(scalar).
    - fold_mses: List of mean squared error values from each fold.
    - J_history: List of cost function outputs across training intervals.
    """
    
    #K-Folds with shuffling and random seed.
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    #Initialize lists.
    fold_mses = []
    fold_params = []
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        
        #Split the data into train and test for this fold.
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Scaling all feature examples(non splits) with robustscaler. 
        scaler_robust = robust_scaler_function()
        X_train_scaled_robust = scaler_robust.fit_transform(X_train)
        X_test_scaled_robust = scaler_robust.transform(X_test)
    
        #Scaling with MinMaxscaler to obtain combined scaler.
        scaler_minmax = minmax_scaler_function()
        X_train_scaled_combined = scaler_minmax.fit_transform(X_train_scaled_robust)
        X_test_scaled_combined = scaler_minmax.transform(X_test_scaled_robust)

        #Perform gradient descent on the scaled training data
        w, b, J_history = gradient_descent_function(X_train_scaled_combined, y_train, w_int, b_int, cost_function, gradient_function, alpha, lambda_, num_iters, tol)
        
        
        #Make predictions on the test set. 
        y_predicted = np.dot(X_test_scaled_combined, w) + b 

        #Compute the mean squared error on the test set.  
        mse = np.mean((y_test - y_predicted) ** 2)
        fold_mses.append(mse)
        fold_params.append((w, b))  # Store the parameters after convergence for this fold

        print(f"Fold {fold_idx+1}, MSE: {mse:.4f}")
    
    #Find the fold with the lowest MSE
    best_fold_idx = np.argmin(fold_mses)
    best_w, best_b = fold_params[best_fold_idx]
    
    print(f"Average MSE across all folds: {np.mean(fold_mses)}")
    print(f"Best Fold: {best_fold_idx + 1}, Best MSE: {fold_mses[best_fold_idx]:.4f}")
    print(f"Best parameters: w = {best_w}, b = {best_b}")
    

    return best_w, best_b, fold_mses, J_history



def cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X, y, w_in, b_in, scaler_function, gradient_function, gradient_descent_function, k, tol):
    """
    Performs gradient descent within K-fold cross validation, with one scaling method, over different gradient descent hyperparameters to obtain the optimal combination based on the lowest average mse across all folds. This function also tracks and stores converging and non-converging hyperparamter combinations. The "gradient_descent_reg_with_convergence_check_for_tracking" gradient descent function or a similar variant is required for this function. This is because the functionality of this function depends on the boolean objects returned in "gradient_descent_reg_with_convergence_check_for_tracking".
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_in: Initial values of weight parameters(numpy.ndarray).
    - b_in: Initial value of parameter(scaler).
    - scaler_function: Scaler class to initialize scaler instance for feature scaling.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: The "gradient_descent_reg_with_convergence_check_for_tracking" function or a similar variant.
    - k: Number of folds for cross validation.
    - tol: Convergence threshold(scalar).
    Returns:
    - best_params: Dictionary including the gradient descent hyperparameters that result in the lowest average mse across all folds.
    - results: List of dictionaries including converging gradient descent hyperparameter combinations.
    - non_converging_combinations: List of dictionaries including non-converging gradient descent hyperparameter combinations.
    """
    
    
    #Hyperparameter grid.
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.8]
    num_iters = [1000, 5000, 10000, 15000, 20000]
    lambda_regs = [0.01, 0.1, 1.0, 10.0, 25.0, 50.0, 75.0, 100.0]
    results = []
    non_converging_combinations = []  #To track non-converging hyperparameter combinations.
    best_params = None
    lowest_avg_mse = float('inf')  #Initialize with a high value to find the lowest MSE.
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    #Iterate over hyperparameter grid.
    for lr in learning_rates:
        for num_iter in num_iters:
            for lambda_reg in lambda_regs:
                mse_scores = []
                converged = True #Assume the hyperparameter combination will result in parameter convergence.
                
                #Perform K-fold cross-validation.
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    #Scale training and test splits. 
                    scaler = scaler_function()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    #Initialize weights and bias.
                    w = copy.deepcopy(w_in)
                    b = b_in

                    #Run gradient descent with convergence check.
                    weights, bias, did_converge = gradient_descent_function(X_train_scaled, y_train, w, b, gradient_function, lr, lambda_reg, num_iter, tol)

                    if not did_converge:
                        converged = False #Change this convergence assumption to False i.e., that hyperparameter combination did not result in parameter convergence across all folds within cross validation.
                        break  #Exit this hyperparameter combination if it didn't result in parameter convergence i.e., stop training on other folds within cross validation.

                    #Calculate MSE on the test set.
                    y_pred = np.dot(X_test_scaled, weights) + bias
                    mse = np.mean((y_test - y_pred) ** 2)
                    mse_scores.append(mse)

                if converged:
                    #Calculate the average MSE(average mse across all folds) for this hyperparameter combination which results in parameter convergence.
                    avg_mse = np.mean(mse_scores)
                    results.append({
                        'alpha': lr,
                        'num_iters': num_iter,
                        'lambda_': lambda_reg,
                        'avg_mse': avg_mse
                    })
                    
                    #Track the best hyperparameters based on the lowest average MSE.
                    if avg_mse < lowest_avg_mse:
                        lowest_avg_mse = avg_mse
                        best_params = {'alpha': lr, 'num_iters': num_iter, 'lambda_': lambda_reg, 'avg_mse': avg_mse}
                
                else:
                    #Track non-converging hyperparameter combinations.
                    non_converging_combinations.append({
                        'alpha': lr,
                        'num_iters': num_iter,
                        'lambda_': lambda_reg
                    })
                    print(f"Non-converging combination: alpha={lr}, num_iters={num_iter}, lambda_={lambda_reg}")

    #Return the best hyperparameters, list of dictionaries including converging combinations(results), and list of dictionaries including non-converging combinations.
    return best_params, results, non_converging_combinations




def cross_validation_and_grid_search_with_convergence_tracking_with_combined_scaler(X, y, w_in, b_in, robust_scaler_function, minmax_scaler_function, gradient_function, gradient_descent_function, k, tol):
    """
    Performs gradient descent within K-fold cross validation, with combined scaling methods, over different gradient descent hyperparameters to obtain the optimal combination based on the lowest average mse across all folds. This function also tracks and stores converging and non-converging hyperparamter combinations. The "gradient_descent_reg_with_convergence_check_for_tracking" gradient descent function or a similar variant is required for this function. This is because the functionality of this function depends on the boolean objects returned in "gradient_descent_reg_with_convergence_check_for_tracking".
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_in: Initial values of weight parameters(numpy.ndarray).
    - b_in: Initial value of parameter(scaler).
    - robust_scaler_function: RobustScaler class to initialize scaler instance for feature scaling.
    - minmax_scaler_function: MinMaxScaler class to initialize scaler instance for feature scaling.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: The "gradient_descent_reg_with_convergence_check_for_tracking" function or a similar variant.
    - k: Number of folds for cross validation.
    - tol: Convergence threshold(scalar).
    Returns:
    - best_params: Dictionary including the gradient descent hyperparameters that result in the lowest average mse across all folds.
    - results: List of dictionaries including converging gradient descent hyperparameter combinations.
    - non_converging_combinations: List of dictionaries including non-converging gradient descent hyperparameter combinations.
    """
    
    
    #Hyperparameter grid.
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.8]
    num_iters = [1000, 5000, 10000, 15000, 20000]
    lambda_regs = [0.01, 0.1, 1.0, 10.0, 25.0, 50.0, 75.0, 100.0]
    results = []
    non_converging_combinations = []  #To track non-converging hyperparameter combinations.
    best_params = None
    lowest_avg_mse = float('inf')  #Initialize with a high value to find the lowest MSE.
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    #Iterate over hyperparameter grid.
    for lr in learning_rates:
        for num_iter in num_iters:
            for lambda_reg in lambda_regs:
                mse_scores = []
                converged = True    #Assume the hyperparameter combination will result in parameter convergence.
                
                #Perform K-fold cross-validation.
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    #Step 1: Apply Robust Scaler.
                    scaler_robust = robust_scaler_function()
                    X_train_scaled_robust = scaler_robust.fit_transform(X_train)
                    X_test_scaled_robust = scaler_robust.transform(X_test)

                    #Step 2: Apply Min-Max Scaler.
                    scaler_minmax = minmax_scaler_function()
                    X_train_scaled_combined = scaler_minmax.fit_transform(X_train_scaled_robust)
                    X_test_scaled_combined = scaler_minmax.transform(X_test_scaled_robust)

                    #Initialize weights and bias.
                    w = copy.deepcopy(w_in)
                    b = b_in

                    #Run gradient descent with convergence check.
                    weights, bias, did_converge = gradient_descent_function(X_train_scaled_combined, y_train, w, b, gradient_function, lr, lambda_reg, num_iter, tol)

                    if not did_converge:
                        converged = False   #Change this convergence assumption to False i.e., that hyperparameter combination did not result in parameter convergence across all folds within cross validation.
                        break  #Exit this hyperparameter combination if it didn't result in parameter convergence i.e., stop training on other folds within cross validation.

                    #Calculate MSE on the test set.
                    y_pred = np.dot(X_test_scaled_combined, weights) + bias
                    mse = np.mean((y_test - y_pred) ** 2)
                    mse_scores.append(mse)

                if converged:
                    #Calculate the average MSE(average mse across all folds) for this hyperparameter combination which results in parameter convergence.
                    avg_mse = np.mean(mse_scores)
                    results.append({
                        'alpha': lr,
                        'num_iters': num_iter,
                        'lambda_': lambda_reg,
                        'avg_mse': avg_mse
                    })
                    
                    #Track the best hyperparameters based on the lowest average MSE.
                    if avg_mse < lowest_avg_mse:
                        lowest_avg_mse = avg_mse
                        best_params = {'alpha': lr, 'num_iters': num_iter, 'lambda_': lambda_reg, 'avg_mse': avg_mse}
                
                else:
                    #Track non-converging hyperparameter combinations.
                    non_converging_combinations.append({
                        'alpha': lr,
                        'num_iters': num_iter,
                        'lambda_': lambda_reg
                    })
                    print(f"Non-converging combination: alpha={lr}, num_iters={num_iter}, lambda_={lambda_reg}")

    #Return the best hyperparameters, list of dictionaries including converging combinations(results), and list of dictionaries including non-converging combinations.
    return best_params, results, non_converging_combinations


def retrain_on_full_data_with_scaling(X, y, optimal_w, optimal_b, scaler_function, cost_function, gradient_function, gradient_descent_function, alpha, lambda_, num_iters, tol):
    """
    Performs gradient descent on all examples, with one scaling method, using optimal hyperparameters. The gradient descent function used in this function should have convergence checking functionality.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - optimal_w: Optimal values of weight parameters obtained from other training methods(numpy.ndarray).
    - optimal_b: Optimal value of parameter obtained from other training methods(scaler).
    - scaler_function: Scaler class to initialize scaler instance for feature scaling.
    - cost_function: Function to compute regularized cost.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    - alpha: Learning rate(scalar).
    - lambda_: Controls strength of regularization(scalar).
    - num_iters: Number of iterations for gradient descent(scalar).
    - tol: Convergence threshold(scalar).
    Returns:
    - w: Updated values of weight parameters after running gradient descent or until convergence if applicable(numpy.ndarray).
    - b: Updated value of bias parameter after running gradient descent or until convergence if applicable(scalar).
    - mse: Mean squared error value from training on all examples(scalar).
    - J_history: List of cost function outputs across training intervals.
    """
    
    #Apply scaler to the entire dataset.
    scaler = scaler_function()
    X_scaled = scaler.fit_transform(X)

    #Perform gradient descent on the scaled data.
    w, b, J_history = gradient_descent_function(X_scaled, y, optimal_w, optimal_b, cost_function, gradient_function, alpha, lambda_, num_iters, tol)
    
    #Make predictions on the entire dataset.
    y_pred = np.dot(X_scaled, w) + b

    #Compute MSE on the entire dataset.
    mse = np.mean((y - y_pred) ** 2)

    return w, b, mse, J_history




def retrain_on_full_data_with_combined_scaling(X, y, optimal_w, optimal_b, robust_scaler_function, minmax_scaler_function, cost_function, gradient_function, gradient_descent_function, alpha, lambda_, num_iters, tol):
    """
    Performs gradient descent on all examples, with combined scaling methods, using optimal hyperparameters. The gradient descent function used in this function should have convergence checking functionality.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - optimal_w: Optimal values of weight parameters obtained from other training methods(numpy.ndarray).
    - optimal_b: Optimal value of parameter obtained from other training methods(scaler).
    - robust_scaler_function: RobustScaler class to initialize scaler instance for feature scaling.
    - minmax_scaler_function: MinMaxScaler class to initialize scaler instance for feature scaling.
    - cost_function: Function to compute regularized cost.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    - alpha: Learning rate(scalar).
    - lambda_: Controls strength of regularization(scalar).
    - num_iters: Number of iterations for gradient descent(scalar).
    - tol: Convergence threshold(scalar).
    Returns:
    - w: Updated values of weight parameters after running gradient descent or until convergence if applicable(numpy.ndarray).
    - b: Updated value of bias parameter after running gradient descent or until convergence if applicable(scalar).
    - mse: Mean squared error value from training on all examples(scalar).
    - J_history: List of cost function outputs across training intervals.
    """
    
    #Apply robust scaler to the entire dataset.
    robust_scaler = robust_scaler_function()
    X_robust_scaled = robust_scaler.fit_transform(X)
    
    #Apply Min-Max scaler to obtain the combined scaler. 
    minmax_scaler = minmax_scaler_function()
    X_scaled_combined = minmax_scaler.fit_transform(X_robust_scaled)
    

    #Perform gradient descent on the scaled data.
    w, b, J_history = gradient_descent_function(X_scaled_combined, y, optimal_w, optimal_b, cost_function, gradient_function, alpha, lambda_, num_iters, tol)
    
    #Make predictions on the entire dataset.
    y_pred = np.dot(X_scaled_combined, w) + b

    #Compute MSE on the entire dataset.
    mse = np.mean((y - y_pred) ** 2)

    return w, b, mse, J_history





'Functions for Training with Tree-Based Methods'

def decision_tree_with_grid_search_within_cross_validation(X, y, k):
    """
    Performs grid search within cross-validation for a DecisionTreeRegressor.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - k: Number of folds for cross-validation.
    Returns:
    - best_params: Dictionary including decision tree hyperparameters that result in the lowest average mse across all folds.
    - best_mse: The lowest average mse across all folds obtained from cross-validation(scalar).
    """
    #Hyperparameter grid.
    max_depths = [None, 2, 5, 10, 20, 30, 40, 50]
    min_samples_splits = [2, 5, 10, 20, 30, 40, 50]
    min_samples_leafs = [1, 2, 4, 10, 20]
    max_features_options = [None, 'sqrt', 'log2', 0.5]
    max_leaf_nodes_options = [None, 10, 20, 30, 40, 50, 80, 100]
    
    best_mse = float('inf')
    best_params = None
    
    #Cross-validation setup.
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    #Loop through hyperparameter grid.
    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            for min_samples_leaf in min_samples_leafs:
                for max_features in max_features_options:
                    for max_leaf_nodes in max_leaf_nodes_options:
                        #Print which combinations are being tested.
                        print(f'Testing combination: max_depth={max_depth}, '
                              f'min_samples_split={min_samples_split}, '
                              f'min_samples_leaf={min_samples_leaf}, '
                              f'max_features={max_features}, '
                              f'max_leaf_nodes={max_leaf_nodes}')
                        
                        mse_per_fold = []
                        
                        #Cross-validation loop.
                        for train_idx, test_idx in kf.split(X):
                            X_train, X_test = X[train_idx], X[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]
                            
                            #Initialize DecisionTreeRegressor with current hyperparameters.
                            model = DecisionTreeRegressor(
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features,
                                max_leaf_nodes=max_leaf_nodes,
                                random_state=42  #Fixed random_state for consistency.
                            )
                            
                            #Train the model.
                            model.fit(X_train, y_train)
                            
                            #Predict and calculate MSE.
                            y_pred = model.predict(X_test)
                            mse = np.mean((y_test - y_pred) ** 2)
                            mse_per_fold.append(mse)
                        
                        #Average MSE across all folds.
                        avg_mse = np.mean(mse_per_fold)
                        
                        #Update best hyperparameters if the current combination results in a lower average mse.
                        if avg_mse < best_mse:
                            best_mse = avg_mse
                            best_params = {
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'max_features': max_features,
                                'max_leaf_nodes': max_leaf_nodes,
                                'average mse': best_mse
                            }
    
    print(f'Optimal hyperparameters: {best_params}')
    print(f'Best average MSE: {best_mse}')
    
    return best_params, best_mse
    
    

def train_decision_tree_regression_within_CV_and_outside_on_all_examples(X, y, k, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, random_state):
    """
    Trains a model using a DecisionTreeRegressor within k-fold cross-validation and also calculates the final mse from training on all examples.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - k: Number of folds for cross-validation(scalar).
    - max_depth: Maximum depth of the tree(scalar).
    - min_samples_split: Minimum number of samples required to split a node(scalar).
    - min_samples_leaf: Minimum number of samples required at a leaf node(scalar).
    - max_features: Maximum number of features considered when looking for the best split(scalar).
    - max_leaf_nodes: Maximum number of leaf nodes in each tree(scalar).
    - random_state: Seed for reproducibility(scalar).  
    Returns:
    - avg_mse: The average mse across all folds(scalar).
    - mse_per_fold: List of mse values for each fold.
    - final_mse: Mean squared error from training on all examples.
    - trained_model: Decision Tree model trained on all examples.
    """
    
    #Initialize K-fold cross-validation.
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    mse_per_fold = []
    
    #Iterate through each fold.
    for train_index, test_index in kf.split(X):
        # Split the data into training and test sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Create and fit the decision tree model.
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=random_state)
        model.fit(X_train, y_train)
        
        #Make predictions on the test set.
        y_pred = model.predict(X_test)
        
        #Calculate mean squared error for each fold.
        mse = np.mean((y_test - y_pred) ** 2)
        mse_per_fold.append(mse)
    
    #Average MSE across all folds.
    avg_mse = np.mean(mse_per_fold)
    
    #Train the model on the entire dataset (without cross-validation)
    trained_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
    trained_model.fit(X, y)
    
    #Calculate the final MSE on the entire dataset.
    y_pred_final = trained_model.predict(X)
    final_mse = np.mean((y - y_pred_final) ** 2)
    
    #Output the results and return the final model and final MSE.
    print(f"Average MSE across {k} folds: {avg_mse}")
    print(f"MSE per fold: {mse_per_fold}")
    print(f"Final MSE from training on all examples: {final_mse}")
    
    return avg_mse, mse_per_fold, final_mse, trained_model



def train_random_forest_regression_within_CV_and_outside_on_all_examples(X, y, k, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes, random_state):
    """
    Trains a model using a RandomForestRegressor within k-fold cross-validation and also calculates the final mse from training on all examples.

    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - k: Number of folds for cross-validation(scalar).
    - n_estimators: Number of trees in the forest(scalar).
    - max_depth: Maximum depth of the trees(scalar).
    - min_samples_split: Minimum number of samples required to split an internal node(scalar).
    - min_samples_leaf: Minimum number of samples required at a leaf node(scalar).
    - max_features: Maximum number of features considered when looking for the best split(scalar).
    - max_leaf_nodes: Maximum number of leaf nodes in each tree(scalar).
    - random_state: Seed for reproducibility(scalar).
    Returns:
    - avg_mse: The average mean squared error across all folds
    - mse_per_fold: List of MSE values for each fold
    - final_mse: MSE of the model trained on the entire dataset
    - trained_model: Random Forest model trained on the entire dataset
    """
    
    #Initialize K-fold cross-validation.
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    mse_per_fold = []
    
    #Iterate through each fold.
    for train_index, test_index in kf.split(X):
        #Split the data into training and test sets for this fold.
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Create and fit the random forest model.
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=random_state)
        model.fit(X_train, y_train)
        
        #Make predictions on the test set.
        y_pred = model.predict(X_test)
        
        #Calculate mean squared error.
        mse = np.mean((y_test - y_pred) ** 2)
        mse_per_fold.append(mse)
    
    #Average MSE across all folds.
    avg_mse = np.mean(mse_per_fold)
    
    #Train the model on the entire dataset (without cross-validation).
    trained_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=random_state)
    trained_model.fit(X, y)
    
    #Calculate the final MSE on the entire dataset.
    y_pred_final = trained_model.predict(X)
    final_mse = np.mean((y - y_pred_final) ** 2)
    
    #Output the results and return the final model and final MSE.
    print(f"Average MSE across {k} folds: {avg_mse}")
    print(f"MSE per fold: {mse_per_fold}")
    print(f"Final MSE from training on all examples: {final_mse}")
    
    return avg_mse, mse_per_fold, final_mse, trained_model





'''

Visualization Functions:

'''


def plot_feature_vs_target_scatter(X, y, feature_names, target_name, title):
    """
    Plots the coordinate distribution of each feature with the target using a scatter plot.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - feature_names: List of strings for feature labels.
    - target_name: String for target label.
    - title: String for title of entire figure.
    """

    
    #Number of features to plot.
    n_features = X.shape[1]
    
    #Create subplots (1 row, n columns based on the number of features).
    fig, ax = plt.subplots(1, n_features, figsize=(18, 5), sharey=True)
    
    #Plot each feature against the target.
    for i in range(n_features):
        ax[i].scatter(X[:, i], y, label='(x, y) coordinate', s=2)
        ax[i].set_xlabel(feature_names[i])
        ax[i].legend()
    
    #Label the first subplot with the target name.
    ax[0].set_ylabel(target_name)
    
    #Add a title to the entire figure.
    fig.suptitle(title)
    
    #Show the plot.
    plt.show()

    


def plot_isolated_feature_regressions_from_gradient_descent_model_with_polys_and_individuals_with_rowsandcols(X_poly, y, w, b, feature_labels, title, original_indices, poly_index, n_cols=3):
    """
    Plots isolated feature regressions for a linear regression model trained with gradient descent.
    The effects of other features are held constant at their mean. This function plots the combined effect of the original feature
    and its polynomial transformation, while plotting other features individually.
    
    Parameters:
    - X_poly: Feature matrix(numpy.ndarray) with polynomial transformations.
    - y: Target array(numpy.ndarray).
    - w: Weight vector(numpy.ndarray) from the gradient descent model.
    - b: Bias term from the gradient descent model.
    - original_indices: List of indices for the original features.
    - poly_index: Index of the polynomial term's original feature (None if no polynomial terms).
    - n_cols: Number of subplots per row (default is 3).
    """
    num_features = len(original_indices)  # Number of original features (not counting polynomial terms)
    
    # We're plotting individual features, and combining one feature with its polynomial term
    num_subplots = len(original_indices)

    # Determine the number of rows needed for the given number of columns
    n_rows = (num_subplots + n_cols - 1) // n_cols  # This ensures enough rows for all subplots

    X_mean = X_poly.mean(axis=0)  # Calculate the mean of the features
    
    # Create a plot with the correct number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6))
    axes = axes.flatten()  # Flatten to easily index subplots

    subplot_idx = 0  # Keep track of the subplot index
    for orig_idx in original_indices:
        X_plot = np.linspace(X_poly[:, orig_idx].min(), X_poly[:, orig_idx].max(), 100)
        y_pred_plot = []

        # Prepare to predict while holding all other features constant at their means
        for val in X_plot:
            X_example = X_mean.copy()
            X_example[orig_idx] = val
            
            # If this is the feature with a polynomial term, also set the polynomial term's value
            if orig_idx == poly_index:
                poly_term_idx = poly_index + 1  # Assuming poly feature is right after the original
                X_example[poly_term_idx] = val ** 2  # Assuming quadratic term for now
            
            # Predict using the formula y_pred = X @ w + b
            y_pred = X_example @ w + b
            y_pred_plot.append(y_pred)
        
        # Plot actual data points
        axes[subplot_idx].scatter(X_poly[:, orig_idx], y, color='blue', label='Actual', s=2)

        # Plot predictions for the feature (and combined polynomial term if applicable)
        axes[subplot_idx].plot(X_plot, y_pred_plot, color='red', label='Predicted')
        
        # Label the axes
        if orig_idx == poly_index:
            axes[subplot_idx].set_xlabel(f'Linear and Quadratic Contributions of {feature_labels[orig_idx]}')
        else:
            axes[subplot_idx].set_xlabel(feature_labels[orig_idx])
        
        axes[subplot_idx].legend()

        subplot_idx += 1  # Move to the next subplot

    # Remove any extra empty subplots (if applicable)
    for i in range(subplot_idx, len(axes)):
        fig.delaxes(axes[i])  # Remove the extra empty subplots
    
    #Label y axis for left most subplots.
    axes[0].set_ylabel('Sale Price($dollars)')
    axes[3].set_ylabel('Sale Price($dollars)')
    
    #Add a title to the entire figure.
    fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()

    
    
def plot_isolated_feature_regressions_from_tree_based_model_with_polyfeature_combined(X_poly, y, model, feature_labels, title, x3_index, poly_index):
    """
    Plots isolated feature regressions for a decision tree or random forest model. The effects of other features
    are held constant at their mean. This function plots the combined effect of the original feature (x3)
    and its polynomial transformation (x3^2), while plotting other features individually.
    
    Parameters:
    - X_poly: Feature matrix(numpy.ndarray) with polynomial transformations.
    - y: Target array(numpy.ndarray).
    - model: Trained decision tree or random forest model.
    - feature_labels: List of strings for feature labels.
    - title: String for title of entire figure.
    - x3_index: Index of the original feature (e.g., x3).
    - poly_index: Index of the polynomial feature (e.g., x3^2).
    """
    num_features = X_poly.shape[1]  #Total number of features including polynomial terms.
    
    #One fewer subplot since x3 and x3_poly will be combined.
    num_subplots = num_features - 1

    X_mean = X_poly.mean(axis=0)  #Calculate the mean of the polynomial features.
    
    #Create a plot with the correct number of subplots.
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4))
    
    subplot_idx = 0  #To track subplot index since x3 and x3_poly will not be plotted individually.
    
    for i in range(num_features):
        #Skip the individual plots of x3 and x3_poly, since they will be plotted together.
        if i == x3_index or i == poly_index:
            continue
        
        #Set up a range for the i-th feature.
        X_plot = np.linspace(X_poly[:, i].min(), X_poly[:, i].max(), 100)
        
        #Prepare to predict while holding all other features constant at their means.
        y_pred_plot = []
        for val in X_plot:
            X_example = X_mean.copy()
            X_example[i] = val
            y_pred = model.predict([X_example])
            y_pred_plot.append(y_pred[0])
        
        #Plot actual data points.
        axes[subplot_idx].scatter(X_poly[:, i], y, color='blue', label='Actual', s=2)
        
        #Plot model predictions for the i-th feature.
        axes[subplot_idx].plot(X_plot, y_pred_plot, color='red', label='Predicted')
        
        #Label the axes.
        axes[subplot_idx].set_xlabel(feature_labels[i])
        axes[subplot_idx].legend()
        
        subplot_idx += 1  #Increment the subplot index.
    
    #Set the y-axis label only for the first subplot.
    axes[0].set_ylabel('Sale Price($dollars)')
    
    #Plot combined regression of both x3 and x3_poly together.
    X_plot_combined = np.linspace(X_poly[:, x3_index].min(), X_poly[:, x3_index].max(), 100)
    y_pred_combined = []
    
    for val in X_plot_combined:
        X_example = X_mean.copy()
        X_example[x3_index] = val
        X_example[poly_index] = val ** 2  #Apply quadratic transformation.
        y_pred_combined.append(model.predict([X_example])[0])
    
    #Plot the combined polynomial feature regression in the last subplot.
    axes[subplot_idx].scatter(X_poly[:, x3_index], y, color='blue', label='Actual', s=2)
    axes[subplot_idx].plot(X_plot_combined, y_pred_combined, color='red', label='Predicted')
    axes[subplot_idx].set_xlabel(f'{feature_labels[x3_index]} with Original and Squared Terms')
    axes[subplot_idx].legend()
    
    #Add a title to the entire figure.
    fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()


    
    
def plot_isolated_feature_regressions_from_tree_based_model_with_polyfeature_combined_with_rowsandcols(X_poly, y, model, feature_labels, title, original_indices, poly_index, n_cols=3):
    """
    Plots isolated feature regressions for a tree-based model (e.g., Decision Tree or Random Forest).
    The effects of other features are held constant at their mean. This function plots the combined effect of the original feature
    and its polynomial transformation, while plotting other features individually.
    
    Parameters:
    - X_poly: Feature matrix(numpy.ndarray) with polynomial transformations.
    - y: Target array(numpy.ndarray).
    - model: The trained tree-based model (e.g., DecisionTree or RandomForest).
    - feature_labels: List of strings for feature labels.
    - title: String for title of entire figure.
    - original_indices: List of indices for the original features.
    - poly_index: Index of the polynomial term's original feature (None if no polynomial terms).
    - n_cols: Number of subplots per row (default is 3).
    """
    num_features = X_poly.shape[1]  #Total number of features including polynomial terms.
    
    #One fewer subplot since x3 and x3_poly will be combined.
    num_subplots = num_features

    #Determine the number of rows needed for the given number of columns.
    n_rows = (num_subplots + n_cols - 1) // n_cols  #Ensures enough rows for all subplots.

    X_mean = X_poly.mean(axis=0)  #Calculate the mean of the features.
    
    #Create a plot with the correct number of rows and columns.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 2))
    axes = axes.flatten()  #Flatten the axes array for easier indexing.

    subplot_idx = 0  #Keep track of the subplot index in case there is polynomial features.
    for orig_idx in original_indices:
        X_plot = np.linspace(X_poly[:, orig_idx].min(), X_poly[:, orig_idx].max(), 100)
        y_pred_plot = []

        #Prepare to predict while holding all other features constant at their means.
        for val in X_plot:
            X_example = X_mean.copy()
            X_example[orig_idx] = val
            
            #Check if this feature has a polynomial term.
            if poly_index is not None and orig_idx == poly_index:
                #Get the index of the corresponding polynomial term (assumed right after the original feature)
                poly_term_idx = poly_index + 1  #Assuming poly feature is right after the original.
                X_example[poly_term_idx] = val ** 2  #Apply quadratic transformation.
            
            #Predict using the trained tree-based model.
            y_pred = model.predict(X_example.reshape(1, -1))[0] 
            y_pred_plot.append(y_pred)
        
        #Plot actual data points.
        axes[subplot_idx].scatter(X_poly[:, orig_idx], y, color='blue', label='Actual', s=2)

        #Plot predictions for the feature (and combined polynomial term if applicable).
        if poly_index is not None and orig_idx == poly_index:
            axes[subplot_idx].set_xlabel(f'{feature_labels[orig_idx]} with Original and Squared Terms')
        else:
            axes[subplot_idx].set_xlabel(feature_labels[orig_idx])
        
        axes[subplot_idx].plot(X_plot, y_pred_plot, color='red', label='Predicted')
        axes[subplot_idx].legend()

        subplot_idx += 1  #Move to the next subplot.

    #Remove any extra empty subplots (if applicable).
    for i in range(subplot_idx, len(axes)):
        fig.delaxes(axes[i])  #Remove the extra empty subplots.
    
    #Set the y-axis label.
    axes[0].set_ylabel('Sale Price($dollars)')
    axes[3].set_ylabel('Sale Price($dollars)')
    axes[6].set_ylabel('Sale Price($dollars)')
    axes[9].set_ylabel('Sale Price($dollars)')
    
    #Add a title to the entire figure.
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()

    

def plot_feature_importance_for_gradient_descent(X_train, feature_names, weight_coefficients):
    """
    Plots feature importances based on the magnitude of weight coefficients using a bar plot 
    from a model trained with gradient descent (e.g., linear regression).
    
    Parameters:
    - X_train: Training feature matrix(numpy.ndarray).
    - feature_names: List of strings for feature names.
    - weight_coefficients: List of weight coefficients(Assumes these weight coefficients were calculated with appropriate scaling if needed).
    Returns:
    - sorted_indices: List of indices of the features sorted by their absolute coefficient values.
    """
    
    #Get the absolute values of the weight coefficients for sorting.
    abs_weight_coefficients = np.abs(weight_coefficients)
    
    #Sort the weight coefficients in descending order of importance (magnitude).
    sorted_indices = np.argsort(abs_weight_coefficients)[::-1]
    
    #Plot the feature importances based on the weight coefficient magnitudes.
    plt.figure(figsize=(12, 7))
    plt.title('Figure 9: Bar Plot of Feature Importance from the Optimal Standard Scaled Gradient Descent Model')
    plt.bar(range(X_train.shape[1]), abs_weight_coefficients[sorted_indices], align='center')
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in sorted_indices], rotation=90)
    plt.ylabel('Absolute Weight Coefficient(unitless)')
    plt.tight_layout()
    plt.show()
    
    #Print the feature importance based on weight coefficient magnitudes.
    print("Feature importance based on weight coefficient magnitudes:")
    for i in sorted_indices:
        print(f'{feature_names[i]}, Weight Coefficient: {weight_coefficients[i]}, Absolute Weight Coefficient: {abs_weight_coefficients[i]}')
    
    return sorted_indices





def compute_feature_importance_for_random_forest(X_train, feature_names, model):
    """
    Computes and plots feature importances with a bar plot 
    using a model trained with a RandomForestRegressor.
    
    Parameters:
    - X_train: Training feature matrix(numpy.ndarray).
    - feature_names: List of strings for feature names.
    - model: A pre-trained model using a RandomForestRegressor.
    Returns:
    - importances: list of feature importance scores.
    - indices: List of sorted feature indices by importance.
    """
    
    #Get feature importances.
    importances = model.feature_importances_
    
    #Sort feature importances in descending order.
    indices = np.argsort(importances)[::-1]
    
    #Plot the feature importances.
    plt.figure(figsize=(10, 7))
    plt.title('Figure 10: Bar Plot of Feature Importance Scores from the Optimal Unscaled Random Forest Model')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.ylabel('Feature Importance Scores(unitless)')
    plt.tight_layout()
    plt.show()
    
    #Print feature importance scores.
    print("Feature importance scores:")
    for i in indices:
        print(f'{feature_names[i]}, Importance Score: {importances[i]}')
    
    return importances, indices




'''

Other functions:

'''

def fill_missing_values_with_interpolation(arr):
    """
    Fills missing values in a numpy array using linear interpolation.

    Parameters: 
    - arr: Feature array(numpy.ndarray).
    Returns: 
    - arr: Inputted array with missing values filled by interpolation, if needed.
    """
    # Check if input is a numpy array.
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    #Count the number of NaN values.
    nan_count = np.isnan(arr).sum()
    
    if nan_count == 0:
        print("No NaN values found in the array.")
        return arr
    
    print(f"Number of NaN values: {nan_count}")
    
    #Find the indices of valid values (not NaN).
    valid = ~np.isnan(arr)
    x = np.arange(len(arr))

    #Interpolate the missing values.
    arr[np.isnan(arr)] = np.interp(x[np.isnan(arr)], x[valid], arr[valid])

    return arr



def analyze_data(df):
    '''
    Performs the D’Agostino and Pearson’s test for normality and counts the number of outliers and prints these statistics for each feature in a dataframe . Dataframe should include column labels.
    
    Parameters:
    - df: Dataframe which includes column labels.
    '''
    for column in df.columns:
        print(f"\nAnalyzing {column}:\n")
        
        #Perform normality test (D’Agostino and Pearson’s test for normality).
        stat, p = normaltest(df[column])
        print(f"Normality test p-value: {p:.4f}")
        if p > 0.05:
            print(f"{column} is likely normally distributed (p > 0.05)")
        else:
            print(f"{column} is not normally distributed (p <= 0.05)")

        #Detect outliers using IQR.
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        print(f"Number of outliers detected in {column}: {len(outliers)}")


      

        
        
'''

Functions not explicitly implemented in the finished analyses: 

***Note*** These functions at one point were implemented but were either improved, modified, or removed from the analyses. These scripts can provide insight to the programming iterations these analyses went through. 

'''



    
'''

Model Training Functions:

'''


'Functions for Training with Gradient Descent'

def compute_cost(X, y, w, b): 
    """
    Computes the squared error cost function over all examples with partial vectorization.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w: Weight parameters(numpy.ndarray).  
    - b: Bias parameter(scalar).
    Returns:
    - cost: Value of the squared error cost function over all examples(scalar).
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i],w) + b       
        cost = cost + (f_wb_i - y[i])**2              
    cost = cost/(2*m)                                 
    return cost 

 
def compute_regularized_cost_old(X, y, w, b, lambda_):
    """
    Computes the regularized squared error cost functions over all examples with partial vectorization.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w: Weight parameters(numpy.ndarray).  
    - b: Bias parameter(scalar).
    - lambda_: Controls strength of regularization(scalar).
    Returns:
    - total_cost: Output of the regularized squared error cost function over all examples(scalar).
    """

    m,n = X.shape
    
    #Computing the squared error term.
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    
    #Computing the regularization term.
    reg_cost = 0.
    for j in range(n):
        reg_cost = reg_cost + (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost
    
    #Computing the sum of these terms.
    total_cost = cost + reg_cost                                      
    return total_cost

def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for each weight parameter and the bias parameter with partial vectorization. 
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w: Weight parameters(ndarray).  
    - b: Bias parameter(scalar).
    Returns:
      dj_db: The gradient of the cost with respect to the bias parameter(scalar). 
      dj_dw: The gradient of the cost with respect to each weight parameter(numpy.ndarray).
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i,j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw/m                                
    dj_db = dj_db/m                                
        
    return dj_db,dj_dw

 
def compute_regularized_gradient_old(X, y, w, b, lambda_): 
    """
    Computes the regularized gradient for each weight parameter and the unregularized gradient for the bias parameter with partial vectorization. 
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w: Weight parameters(ndarray).  
    - b: Bias parameter(scalar).
    - lambda_: Controls strength of regularization(scalar).
    Returns:
      dj_db: The gradient of the cost with respect to the bias parameter(scalar). 
      dj_dw: The gradient of the cost with respect to each weight parameter(numpy.ndarray).
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.
    
    #Computed the gradients for each parameter. 
    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]                 
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]               
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m   
    
    #Compute the gradient for the regularization term and add this to each corresponding weight parameter gradient.   
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs gradient descent to learn the weight(w) and bias(b) parameters. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_in: Initial values of weight parameters(numpy.ndarray).
    - b_in: Initial value of parameter(scaler).
    - cost_function: Function to compute cost.
    - gradient_function: Function to compute gradients.
    - alpha: Learning rate(scalar).
    - num_iters: Number of iterations for gradient descent(scalar).
    Returns:
    - w: Updated values of weight parameters after running gradient descent(numpy.ndarray).
    - b: Updated value of bias parameter after running gradient descent(scalar).
    - J_history: List of cost function outputs across training intervals.
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J at each iteration primarily for graphing later
    J_history = []
    
    w = copy.deepcopy(w_in) #avoid modifying global w within function
    b = b_in
    alpha = np.float128(alpha)
    
    save_interval = np.ceil(num_iters/100) # prevent resource exhaustion for long runs

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each 10 iterations
        if i == 0 or i % save_interval == 0:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w , b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            #print(f"Iteration {i:4d}: Cost {cost_function(X, y, w, b):8.2f}   ")
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw}, dj_db: {dj_db}  ",
                  f"w: {w}, b:{b}")
    return w, b, J_history #return w,b and history for graphing



def gradient_descent_reg(X, y, w_in, b_in, cost_function, gradient_function, alpha, lambda_, num_iters): 
    """
    Performs gradient descent to learn the weight(w) and bias(b) parameters. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha and regularization term lambda_.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_in: Initial values of weight parameters(numpy.ndarray).
    - b_in: Initial value of parameter(scaler).
    - cost_function: Function to compute regularized cost.
    - gradient_function: Function to compute regularized gradients.
    - alpha: Learning rate(scalar).
    - lambda_: Controls strength of regularization(scalar).
    - num_iters: Number of iterations for gradient descent(scalar).
    Returns:
    - w: Updated values of weight parameters after running gradient descent(numpy.ndarray).
    - b: Updated value of bias parameter after running gradient descent(scalar).
    - J_history: List of cost function outputs across training intervals.
    """
    
    #Number of training examples.
    m = X.shape[0]
    
    #An array to store cost J at each iteration primarily for graphing later.
    J_history = []
    
    w = copy.deepcopy(w_in) #Avoid modifying global w within function.
    b = b_in
    
    save_interval = np.ceil(num_iters/100) #Prevent resource exhaustion for long runs.

    for i in range(num_iters):

        #Compute the gradient for each parameter.
        dj_db,dj_dw = gradient_function(X, y, w, b, lambda_)   

        #Update Parameters using w, b, alpha and gradient.
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        #Save regularized squared error cost J at intervals.
        if i == 0 or i % save_interval == 0:
            J_history.append(cost_function(X, y, w , b, lambda_))

        #Print regularized squared error cost, mse, gradients, and parameter values 10 times throughout training. 
        if i% math.ceil(num_iters/10) == 0:
            y_pred = np.dot(X, w) + b
            mse = np.mean((y - y_pred) ** 2)
            print(f"Iteration {i:4}: Cost {cost_function(X, y, w , b, lambda_):0.2e} MSE {mse:0.2e} ",
                  f"dj_dw: {dj_dw}, dj_db: {dj_db}  ",
                  f"w: {w}, b:{b}")
    print()
    return w, b, J_history    #Return w,b and history for graphing


def gradient_descent_reg_no_stats(X, y, w_in, b_in, gradient_function, alpha, lambda_, num_iters): 
    """
    Performs gradient descent to learn the weight(w) and bias(b) parameters. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha and regularization term lambda_.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_in: Initial values of weight parameters(numpy.ndarray).
    - b_in: Initial value of parameter(scaler).
    - cost_function: Function to compute regularized cost.
    - gradient_function: Function to compute regularized gradients.
    - alpha: Learning rate(scalar).
    - lambda_: Controls strength of regularization(scalar).
    - num_iters: Number of iterations for gradient descent(scalar).
    Returns:
    - w: Updated values of weight parameters after running gradient descent(numpy.ndarray).
    - b: Updated value of bias parameter after running gradient descent(scalar).
    """
    
    #Number of training examples.
    m = len(X)
    
    w = copy.deepcopy(w_in) #Avoid modifying global w within function.
    b = b_in
    
    for i in range(num_iters):

        #Compute the gradient for each parameter.
        dj_db,dj_dw = gradient_function(X, y, w, b, lambda_)   

        #Update Parameters using w, b, alpha and gradient.
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        
    return w, b 



def k_fold_cross_validation_no_stats(X, y, w_int, b_int, scalar_function, cost_function, gradient_function, gradient_descent_function, alpha, num_iters, lambda_, k):
    """
    Performs gradient descent within K-fold cross validation with one scaling method. 
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_int: Initial values of weight parameters(numpy.ndarray).
    - b_int: Initial value of parameter(scaler).
    - scaler_function: Scaler class to initialize scaler instance for feature scaling.
    - cost_function: Function to compute regularized cost.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    - alpha: Learning rate(scalar).
    - num_iters: Number of iterations for gradient descent(scalar).
    - lambda_: Controls strength of regularization(scalar).
    - k: Number of folds for cross validation.
    Returns:
    - avg_mse: Average mse across all folds from cross validation(scalar).
    """
    
    #K-Folds with shuffling.
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    #Use RobustScaler for scaling within each fold.
    scaler = scalar_function()
    fold_mses = []
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        
        #Split the data into train and test for this fold.
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Apply RobustScaler to the training data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #Perform gradient descent on the scaled training data.
        w, b = gradient_descent_function(X_train_scaled, y_train, w_int, b_int, cost_function, gradient_function, alpha, lambda_, num_iters)
        
        #Make predictions on the test set. 
        y_predicted = np.dot(X_test_scaled, w) + b 

        #Compute the mean squared error on the test set.  
        mse = np.mean((y_test - y_predicted) ** 2)
        fold_mses.append(mse)

    #Return average mse across all folds.
    avg_mse = np.mean(fold_mses)
    return avg_mse


def grid_search(X, y, w_in, b_in, scaler_function, cost_function, gradient_function, gradient_descent_function, best_alpha=None, best_iterations=None, best_lamb=None, lowest_MSE=float('inf'), best_b=None):
    """
    Performs gradient descent using a standard train and test split(not cross validation), with one scaling method, over different gradient descent hyperparameters to print the optimal combination based on the lowest mse.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_in: Initial values of weight parameters(numpy.ndarray).
    - b_in: Initial value of parameter(scaler).
    - scaler_function: Scaler class to initialize scaler instance for feature scaling.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    - best_alpha: Variable to store best learning rate(default is None).
    - best_iterations: Variable to store best number of iterations(default is None).
    - best_lamb: Variable to store best lambda value(default is None).
    - lowest_MSE: Variable to store best lambda value(default is float('inf')).
    - best_b: Variable to store best bias parameter(default is None).
    """
    
    #Splitting the feature and target arrays into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Scaling feature splits. 
    scaler = scaler_function()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    #Hyperparameter grid.
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
    iterations = [500, 1000, 1500, 2000, 5000]
    lambda_regs = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    
    #Loop to perform gradient descent with all combinations of hyperparameters to find the combination which yields the lowest MSE. 
    for lr in learning_rates:
        for num_iter in iterations:
            for lambda_reg in lambda_regs:
                #Peform gradient descent.
                weights, bias,_ = gradient_descent_function(X_train_scaled, y_train, w_in, b_in, cost_function, gradient_function, lr, lambda_reg, num_iter)
            
                #Predicting on the validation set.
                y_predicted = np.dot(X_val_scaled, weights) + bias
            
                #Evaluating the model using mean squared error.
                mse = np.mean((y_val - y_predicted) ** 2)
                print(f"Testing model with lr = {lr}, num_iter = {num_iter}, lambda = {lambda_reg}, MSE = {mse}")
                print()
            
                #Updating the best parameters if current MSE is lower
                if mse < lowest_MSE:
                    lowest_MSE = mse
                    best_alpha = lr
                    best_iterations = num_iter
                    best_lamb = lambda_reg
                    best_b = bias

    print(f"Best parameters: Learning Rate = {best_alpha}, Iterations = {best_iterations}, Lambda = {best_lamb}, Bias = {best_b}, Lowest MSE = {lowest_MSE}")


def grid_search_with_full_data(X, y, optimal_w, optimal_b, scaler_function, gradient_function, gradient_descent_function):
    """
    Performs gradient descent on all examples, with one scaling method, over different gradient descent hyperparameters to obtain the optimal combination based on the lowest mse.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - optimal_w: Optimal values of weight parameters obtained from other training methods(numpy.ndarray).
    - optimal_b: Optimal value of parameter obtained from other training methods(scaler).
    - scaler_function: Scaler class to initialize scaler instance for feature scaling.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    Returns:
    - best_w: Updated values of weight parameters after running gradient descent(numpy.ndarray).
    - best_b: Updated value of bias parameter after running gradient descent(scalar).
    - best_params: Dictionary including the gradient descent hyperparameters that result in the lowest mse.
    """
    
    #Scaling all feature examples(non splits). 
    scaler = scaler_function()
    X_train_scaled = scaler.fit_transform(X)
   
    best_mse = float('inf')
    best_w = None
    best_b = None
    best_params = None

    #Hyperparameter grid.
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
    iterations = [1000, 5000, 10000, 15000]
    lambda_regs = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    
    #Loop to perform gradient descent with all combinations of hyperparameters to find the combination which yields the lowest MSE. 
    for lr in learning_rates:
        for num_iter in iterations:
            for lambda_reg in lambda_regs:
                print(f"Testing alpha={lr}, num_iters={num_iter}, lambda_={lambda_reg}")

                #Train using the entire dataset with current hyperparameters.
                w, b = gradient_descent_function(X_train_scaled, y, optimal_w, optimal_b, gradient_function, lr, lambda_reg, num_iter)

                #Make predictions on the entire dataset.
                y_pred = np.dot(X_train_scaled, w) + b

                #Compute MSE on the entire dataset.
                mse = np.mean((y - y_pred) ** 2)

                #Update the best parameters if current MSE is lower.
                if mse < best_mse:
                    best_mse = mse
                    best_w = w
                    best_b = b
                    best_params = {'alpha': lr, 'num_iters': num_iter, 'lambda_': lambda_reg}

                print(f"MSE: {mse:.4f}, Best MSE so far: {best_mse:.4f}")
    
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best MSE: {best_mse:.4f}")
    return best_w, best_b, best_params



def grid_search_with_full_data_and_combined_scaler(X, y, optimal_w, optimal_b, scaler_function_robust, scaler_function_minmax, gradient_function, gradient_descent_function):
    """
    Performs gradient descent on all examples, with combined scaling methods, over different gradient descent hyperparameters to obtain the optimal combination based on the lowest mse.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - optimal_w: Optimal values of weight parameters obtained from other training methods(numpy.ndarray).
    - optimal_b: Optimal value of parameter obtained from other training methods(scaler).
    - scaler_function_robust: RobustScaler class to initialize scaler instance for feature scaling.
    - scaler_function_minmax: MinMaxScaler class to initialize scaler instance for feature scaling.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    Returns:
    - best_w: Updated values of weight parameters after running gradient descent(numpy.ndarray).
    - best_b: Updated value of bias parameter after running gradient descent(scalar).
    - best_params: Dictionary including the gradient descent hyperparameters that result in the lowest mse.
    """
    
    #Scaling all feature examples(non splits) with robustscaler. 
    scaler_robust = scaler_function_robust()
    X_train_scaled_robust = scaler_robust.fit_transform(X)
    
    #Scaling with MinMaxscaler to obtain combined scaler.
    scaler_minmax = scaler_function_minmax()
    X_train_scaled_combined = scaler_minmax.fit_transform(X_train_scaled_robust)
    
    
   
    best_mse = float('inf')
    best_w = None
    best_b = None
    best_params = None

    #Hyperparameter grid.
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
    iterations = [1000, 5000, 10000, 15000]
    lambda_regs = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    
    #Loop to perform gradient descent with all combinations of hyperparameters to find the combination which yields the lowest MSE. 
    for lr in learning_rates:
        for num_iter in iterations:
            for lambda_reg in lambda_regs:
                print(f"Testing alpha={lr}, num_iters={num_iter}, lambda_={lambda_reg}")

                #Train using the entire dataset with current hyperparameters.
                w, b = gradient_descent_function(X_train_scaled_combined, y, optimal_w, optimal_b, gradient_function, lr, lambda_reg, num_iter)

                #Make predictions on the entire dataset.
                y_pred = np.dot(X_train_scaled_combined, w) + b

                #Compute MSE on the entire dataset.
                mse = np.mean((y - y_pred) ** 2)

                #Update the best parameters if current MSE is lower.
                if mse < best_mse:
                    best_mse = mse
                    best_w = w
                    best_b = b
                    best_params = {'alpha': lr, 'num_iters': num_iter, 'lambda_': lambda_reg}

                print(f"MSE: {mse:.4f}, Best MSE so far: {best_mse:.4f}")
    
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best MSE: {best_mse:.4f}")
    return best_w, best_b, best_params





def cross_validation_grid_search_over_k(X, y, w_in, b_in, scaler_function, gradient_function, gradient_descent_function):
    """
    Performs gradient descent within K-fold cross validation, with one scaling method, over different fold configurations(k) and gradient descent hyperparameters to obtain the optimal combination based on the average mse across all folds. 
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_in: Initial values of weight parameters(numpy.ndarray).
    - b_in: Initial value of parameter(scaler).
    - scaler_function: Scaler class to initialize scaler instance for feature scaling.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    Returns:
    - overall_best: Dictionary including the fold configuration(k) and gradient descent hyperparameters that result in the lowest average mse across all folds.
    """
    
    w = copy.deepcopy(w_in) #Avoid modifying global w within function.
    b = b_in
    k_values = [5, 10]
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
    num_iters = [1000, 5000, 10000, 15000]
    lambda_regs = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    results = []

    for k in k_values:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        best_params = None
        lowest_avg_mse = float('inf')

        for lr in learning_rates:
            for num_iter in num_iters:
                for lambda_reg in lambda_regs:
                    mse_scores = []

                    for train_index, test_index in kf.split(X):
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        #Scale data.
                        scaler = scaler_function()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        #Train using gradient descent.
                        weights, bias = gradient_descent_function(X_train_scaled, y_train, w, b, gradient_function, lr, lambda_reg, num_iter)

                        #Predict on test set.
                        y_pred = np.dot(X_test_scaled, weights) + bias

                        #Calculate MSE.
                        mse = np.mean((y_test - y_pred) ** 2)
                        mse_scores.append(mse)

                    #Average MSE for this parameter combination and k.
                    avg_mse = np.mean(mse_scores)

                    #Check if this is the best (lowest) average MSE found for this k.
                    if avg_mse < lowest_avg_mse:
                        lowest_avg_mse = avg_mse
                        best_params = {'alpha': lr, 'num_iters': num_iter, 'lambda': lambda_reg, 'avg_mse': avg_mse, 'k': k}

        #Store the best result for this k.
        results.append(best_params)

    #Find the overall best result across all k values.
    overall_best = min(results, key=lambda x: x['avg_mse'])
    return overall_best




def cross_validation_grid_search_over_k_with_combined_scaler(X, y, w_in, b_in, robust_scaler_function, minmax_scaler_function, gradient_function, gradient_descent_function):
    """
    Performs gradient descent within K-fold cross validation, with combined scaling methods, over different fold configurations(k) and gradient descent hyperparameters to obtain the optimal combination based on the average mse across all folds. 
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_in: Initial values of weight parameters(numpy.ndarray).
    - b_in: Initial value of parameter(scaler).
    - robust_scaler_function: RobustScaler class to initialize scaler instance for feature scaling.
    - minmax_scaler_function: MinMaxScaler class to initialize scaler instance for feature scaling.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    Returns:
    - overall_best: Dictionary including the fold configuration(k) and gradient descent hyperparameters that result in the lowest average mse across all folds.
    """
    
    w = copy.deepcopy(w_in) #Avoid modifying global w within function.
    b = b_in
    k_values = [5, 10]
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
    num_iters = [1000, 5000, 10000, 15000]
    lambda_regs = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    results = []

    for k in k_values:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        best_params = None
        lowest_avg_mse = float('inf')

        for lr in learning_rates:
            for num_iter in num_iters:
                for lambda_reg in lambda_regs:
                    mse_scores = []

                    for train_index, test_index in kf.split(X):
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        #Scaling all feature examples(non splits) with robustscaler. 
                        scaler_robust = robust_scaler_function()
                        X_train_scaled_robust = scaler_robust.fit_transform(X_train)
                        X_test_scaled_robust = scaler_robust.transform(X_test)
    
                        #Scaling with MinMaxscaler to obtain combined scaler.
                        scaler_minmax = minmax_scaler_function()
                        X_train_scaled_combined = scaler_minmax.fit_transform(X_train_scaled_robust)
                        X_test_scaled_combined = scaler_minmax.transform(X_test_scaled_robust)

                        #Train using gradient descent.
                        weights, bias = gradient_descent_function(X_train_scaled_combined, y_train, w, b, gradient_function, lr, lambda_reg, num_iter)

                        #Predict on test set.
                        y_pred = np.dot(X_test_scaled_combined, weights) + bias

                        #Calculate MSE.
                        mse = np.mean((y_test - y_pred) ** 2)
                        mse_scores.append(mse)

                    #Average MSE for this parameter combination and k.
                    avg_mse = np.mean(mse_scores)

                    #Check if this is the best (lowest) average MSE found for this k.
                    if avg_mse < lowest_avg_mse:
                        lowest_avg_mse = avg_mse
                        best_params = {'alpha': lr, 'num_iters': num_iter, 'lambda': lambda_reg, 'avg_mse': avg_mse, 'k': k}

        #Store the best result for this k.
        results.append(best_params)

    #Find the overall best result across all k values.
    overall_best = min(results, key=lambda x: x['avg_mse'])
    return overall_best




def cross_validation_and_grid_search_with_scaler(X, y, w_in, b_in, scaler_function, gradient_function, gradient_descent_function, k):
    """
    Performs gradient descent within K-fold cross validation, with one scaling method, over different gradient descent hyperparameters to obtain the optimal combination based on the average mse across all folds and weight and bias parameters from the best fold. 
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_in: Initial values of weight parameters(numpy.ndarray).
    - b_in: Initial value of parameter(scaler).
    - scaler_function: Scaler class to initialize scaler instance for feature scaling.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    - k: Number of folds for cross validation.
    Returns:
    - best_params: Dictionary including the gradient descent hyperparameters that result in the lowest average mse across all folds.
    - best_weights: Updated values of the weight parameters after gradient descent from the fold that had the lowest mse(numpy.ndarray).
    - best_bias: Updated value of the bias parameter after gradient descent from the fold that had the lowest mse(scalar).
    - results: List of dictionaries including the gradient descent hyperparameter combinations and corresponding average mse across all folds for each cross validation run.
    """
    w = copy.deepcopy(w_in)# Avoid modifying global w within function.
    b = b_in
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
    num_iters = [1000, 5000, 10000, 15000, 20000]
    lambda_regs = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    results = []

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    lowest_avg_mse = float('inf')
    best_params = None
    best_weights = None
    best_bias = None

    #Loop through hyperparameter grid.
    for lr in learning_rates:
        for num_iter in num_iters:
            for lambda_reg in lambda_regs:
                mse_scores = []
                fold_weights_and_biases = []   #Store weights and bias for each fold.
                print(f"Testing alpha={lr}, num_iters={num_iter}, lambda_={lambda_reg}")
                
                #Perform K-fold cross-validation.
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    #Scale training and test splits. 
                    scaler = scaler_function()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    #Train using gradient descent.
                    weights, bias = gradient_descent_function(X_train_scaled, y_train, w, b, gradient_function, lr, lambda_reg, num_iter)

                    #Store weights and bias for this fold.
                    fold_weights_and_biases.append((weights, bias))

                    #Predict on test set.
                    y_pred = np.dot(X_test_scaled, weights) + bias

                    #Calculate MSE.
                    mse = np.mean((y_test - y_pred) ** 2)
                    mse_scores.append(mse)

                #Average MSE for this hyperparameter combination.
                avg_mse = np.mean(mse_scores)

                #Store the result for this parameter combination.
                results.append({'alpha': lr, 'num_iters': num_iter, 'lambda': lambda_reg, 'avg_mse': avg_mse})

                #Check if this is the best hyperparameter combination.
                if avg_mse < lowest_avg_mse:
                    lowest_avg_mse = avg_mse
                    best_params = {'alpha': lr, 'num_iters': num_iter, 'lambda': lambda_reg, 'avg_mse': avg_mse}
                    
                    #Store the weights and bias from the fold that had the lowest MSE.
                    best_fold_index = np.argmin(mse_scores)
                    best_weights, best_bias = fold_weights_and_biases[best_fold_index]  #From the best fold.

    #Return the best parameters, best weights/bias, and all results.
    return best_params, best_weights, best_bias, results



def cross_validation_and_grid_search_with_combined_scaler(X, y, w_in, b_in, robust_scaler_function, minmax_scaler_function, gradient_function, gradient_descent_function, k):
    """
    Performs gradient descent within K-fold cross validation, with one scaling method, over different gradient descent hyperparameters to obtain the optimal combination based on the average mse across all folds and weight and bias parameters from the best fold. 
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - w_in: Initial values of weight parameters(numpy.ndarray).
    - b_in: Initial value of parameter(scaler).
    - robust_scaler_function: RobustScaler class to initialize scaler instance for feature scaling.
    - minmax_scaler_function: MinMaxScaler class to initialize scaler instance for feature scaling.
    - gradient_function: Function to compute regularized gradients.
    - gradient_descent_function: Function to run gradient descent.
    - k: Number of folds for cross validation.
    Returns:
    - best_params: Dictionary including the gradient descent hyperparameters that result in the lowest average mse across all folds.
    - best_weights: Updated values of the weight parameters after gradient descent from the fold that had the lowest mse(numpy.ndarray).
    - best_bias: Updated value of the bias parameter after gradient descent from the fold that had the lowest mse(scalar).
    - results: List of dictionaries including the gradient descent hyperparameter combinations and corresponding average mse across all folds for each cross validation run.
    """
    w = copy.deepcopy(w_in)  #Avoid modifying global w within function.
    w = np.array(w)
    b = b_in
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
    num_iters = [1000, 5000, 10000, 15000]
    lambda_regs = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    results = []

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    lowest_avg_mse = float('inf')
    best_params = None
    best_weights = None
    best_bias = None

    #Loop through hyperparameter grid.
    for lr in learning_rates:
        for num_iter in num_iters:
            for lambda_reg in lambda_regs:
                mse_scores = []
                fold_weights_and_biases = []  #Store weights and bias for each fold.
                print(f"Testing alpha={lr}, num_iters={num_iter}, lambda_={lambda_reg}")
                
                #Perform K-fold cross-validation.
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    #Step 1: Apply Robust Scaler.
                    scaler_robust = robust_scaler_function()
                    X_train_scaled_robust = scaler_robust.fit_transform(X_train)
                    X_test_scaled_robust = scaler_robust.transform(X_test)

                    #Step 2: Apply Min-Max Scaler.
                    scaler_minmax = minmax_scaler_function()
                    X_train_scaled_combined = scaler_minmax.fit_transform(X_train_scaled_robust)
                    X_test_scaled_combined = scaler_minmax.transform(X_test_scaled_robust)

                    #Train using gradient descent.
                    weights, bias = gradient_descent_function(X_train_scaled_combined, y_train, w, b, gradient_function, lr, lambda_reg, num_iter)

                    #Store weights and bias for this fold.
                    fold_weights_and_biases.append((weights, bias))

                    #Predict on test set.
                    y_pred = np.dot(X_test_scaled_combined, weights) + bias

                    #Calculate MSE.
                    mse = np.mean((y_test - y_pred) ** 2)
                    mse_scores.append(mse)

                #Average MSE for this hyperparameter combination.
                avg_mse = np.mean(mse_scores)

                #Store the result for this parameter combination.
                results.append({'alpha': lr, 'num_iters': num_iter, 'lambda': lambda_reg, 'avg_mse': avg_mse})

                #Check if this is the best combination.
                if avg_mse < lowest_avg_mse:
                    lowest_avg_mse = avg_mse
                    best_params = {'alpha': lr, 'num_iters': num_iter, 'lambda': lambda_reg, 'avg_mse': avg_mse}
                    
                    #Store the weights and bias from the fold that had the lowest MSE.
                    best_fold_index = np.argmin(mse_scores)
                    best_weights, best_bias = fold_weights_and_biases[best_fold_index]  #From the best fold.

    #Return the best parameters, best weights/bias, and all results.
    return best_params, best_weights, best_bias, results



'Functions for Training with Tree-Based Methods'

def random_forest_grid_search_with_cv(X, y, k):
    """
    Performs grid search within cross-validation for a RandomForestRegressor.
    
    Parameters:
    - X: Feature matrix(numpy.ndarray).
    - y: Target array(numpy.ndarray).
    - k: Number of folds for cross-validation.
    
    Returns:
    - best_params: Dictionary including random forest hyperparameters that result in the lowest average mse across all folds.
    - best_mse: The lowest average mse across all folds obtained from cross-validation(scalar).
    """
    #Hyperparameter grid.
    n_estimators_options = [100, 200, 500]
    max_depths = [None, 2, 5, 10, 20, 30, 40, 50]
    min_samples_splits = [2, 5, 10, 20, 30, 40, 50]
    min_samples_leafs = [1, 2, 4, 10, 20]
    max_features_options = [None, 'sqrt', 'log2', 0.5]
    max_leaf_nodes_options = [None, 10, 20, 30, 40, 50, 80, 100]
    bootstrap_options = [True, False]
    max_samples_options = [None, 0.5, 0.8]  #Only used when bootstrap=True.
    
    best_mse = float('inf')
    best_params = None
    
    #Cross-validation setup.
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    #Nested loops for grid search.
    for n_estimators in n_estimators_options:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for min_samples_leaf in min_samples_leafs:
                    for max_features in max_features_options:
                        for max_leaf_nodes in max_leaf_nodes_options:
                            for bootstrap in bootstrap_options:
                                for max_samples in max_samples_options:
                                    if bootstrap is False and max_samples is not None:
                                        #Skip max_samples when bootstrap is False (not applicable).
                                        continue
                                    
                                    print(f'Testing combination: n_estimators={n_estimators}, '
                                          f'max_depth={max_depth}, '
                                          f'min_samples_split={min_samples_split}, '
                                          f'min_samples_leaf={min_samples_leaf}, '
                                          f'max_features={max_features}, '
                                          f'max_leaf_nodes={max_leaf_nodes}, '
                                          f'bootstrap={bootstrap}, max_samples={max_samples}')
                                    
                                    mse_per_fold = []
                                    
                                    #Cross-validation loop.
                                    for train_idx, test_idx in kf.split(X):
                                        X_train, X_test = X[train_idx], X[test_idx]
                                        y_train, y_test = y[train_idx], y[test_idx]
                                        
                                        #Initialize RandomForestRegressor with current hyperparameters.
                                        model = RandomForestRegressor(
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            max_leaf_nodes=max_leaf_nodes,
                                            bootstrap=bootstrap,
                                            max_samples=max_samples,
                                            random_state=42  #Fixed random_state for consistency.
                                        )
                                        
                                        #Train the model.
                                        model.fit(X_train, y_train)
                                        
                                        #Predict and calculate MSE.
                                        y_pred = model.predict(X_test)
                                        mse = np.mean((y_test - y_pred) ** 2)
                                        mse_per_fold.append(mse)
                                    
                                    #Average MSE across all folds.
                                    avg_mse = np.mean(mse_per_fold)
                                    
                                    #Update best hyperparameters if the current combination is better.
                                    if avg_mse < best_mse:
                                        best_mse = avg_mse
                                        best_params = {
                                            'n_estimators': n_estimators,
                                            'max_depth': max_depth,
                                            'min_samples_split': min_samples_split,
                                            'min_samples_leaf': min_samples_leaf,
                                            'max_features': max_features,
                                            'max_leaf_nodes': max_leaf_nodes,
                                            'bootstrap': bootstrap,
                                            'max_samples': max_samples
                                        }
    
    print(f'Optimal hyperparameters: {best_params}')
    print(f'Best average MSE: {best_mse}')
    
    return best_params, best_mse




'''

Visualization Functions:

'''

def plot_isolated_feature_regressions_from_gradient_descent_with_polyfeature_combined(X_poly, y, w, b, feature_labels, title, x3_index, poly_index):
    """
    Plot isolated feature regressions for a linear regression model trained with gradient descent. 
    The effects of other features are held constant at their mean. This function plots the combined 
    effect of the original feature (x3) and its polynomial transformation (x3^2), while plotting other features individually.
    
    Parameters:
    - X_poly: Feature matrix(numpy.ndarray) with polynomial transformations.
    - y: Target array(numpy.ndarray).
    - w: Weight vector(numpy.ndarray) from the gradient descent model.
    - b: Bias term from the gradient descent model.
    - feature_labels: List of strings for feature labels.
    - title: String for title of entire figure.
    - x3_index: Index of the original feature (e.g., x3).
    - poly_index: Index of the polynomial feature (e.g., x3^2).
    """
    num_features = X_poly.shape[1]  #Total number of features including polynomial terms.
    
    #One fewer subplot since combining x3 and x3_poly. 
    num_subplots = num_features - 1

    X_mean = X_poly.mean(axis=0)  #Calculate the mean of the polynomial features.
    
    #Create a plot with the correct number of subplots.
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4), sharey=True)
    
    subplot_idx = 0  #To track subplot index since x3 and x3_poly will not be plotted individually.
    
    for i in range(num_features):
        #Skip the individual plots of x3 and x3_poly, since plotting them together.
        if i == x3_index or i == poly_index:
            continue
        
        #Set up a range for the i-th feature.
        X_plot = np.linspace(X_poly[:, i].min(), X_poly[:, i].max(), 100)
        
        #Prepare to predict while holding all other features constant at their means.
        y_pred_plot = []
        for val in X_plot:
            X_example = X_mean.copy()
            X_example[i] = val
            #Calulate and append target predictions.
            y_pred = np.dot(X_example, w) + b
            y_pred_plot.append(y_pred)
        
        #Plot actual data points.
        axes[subplot_idx].scatter(X_poly[:, i], y, color='blue', label='Actual', s=2)
        
        #Plot gradient descent model predictions for the i-th feature.
        axes[subplot_idx].plot(X_plot, y_pred_plot, color='red', label='Predicted')
        
        #Label the axes.
        axes[subplot_idx].set_xlabel(feature_labels[i])
        axes[subplot_idx].legend()
        
        subplot_idx += 1  #Increment the subplot index.
    
    #Set the y-axis label only for the first subplot.
    axes[0].set_ylabel('Sale Price($dollars)')
    
    #Plot combined regression of both x3 and x3_poly together.
    X_plot_combined = np.linspace(X_poly[:, x3_index].min(), X_poly[:, x3_index].max(), 100)
    y_pred_combined = []
    
    for val in X_plot_combined:
        X_example = X_mean.copy()
        X_example[x3_index] = val
        X_example[poly_index] = val ** 2  #Apply quadratic transformation.
        #Calulate and append target predictions.
        y_pred = np.dot(X_example, w) + b
        y_pred_combined.append(y_pred)
    
    #Plot the combined polynomial feature regression in the last subplot.
    axes[subplot_idx].scatter(X_poly[:, x3_index], y, color='blue', label='Actual', s=2)
    axes[subplot_idx].plot(X_plot_combined, y_pred_combined, color='red', label='Predicted')
    axes[subplot_idx].set_xlabel(f'Linear and Quadratic Contributions of {feature_labels[x3_index]}')
    axes[subplot_idx].legend()
    
    #Add a title to the entire figure.
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()    


    
    


