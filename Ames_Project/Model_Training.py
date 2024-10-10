'''
A Comprehensive Regression Analysis of Housing Prices in Ames, Iowa


Research Questions: 

Q1: Which training method yields the best performing model?

Q2: Which feature scaling/engineering methods increase model performance?

Q3: Which features in the best performing models are the most important in predicting the housing price target?
'''


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from Project_Tools import analyze_data, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, gradient_descent_reg_with_convergence_check_for_tracking, k_fold_cross_validation_with_scaler, k_fold_cross_validation_with_combined_scaler, cross_validation_and_grid_search_with_convergence_tracking_with_scaler, cross_validation_and_grid_search_with_convergence_tracking_with_combined_scaler, retrain_on_full_data_with_scaling, retrain_on_full_data_with_combined_scaling, decision_tree_with_grid_search_within_cross_validation, train_decision_tree_regression_within_CV_and_outside_on_all_examples, train_random_forest_regression_within_CV_and_outside_on_all_examples, plot_feature_vs_target_scatter, plot_isolated_feature_regressions_from_gradient_descent_model_with_polys_and_individuals_with_rowsandcols, plot_isolated_feature_regressions_from_tree_based_model_with_polyfeature_combined, plot_isolated_feature_regressions_from_tree_based_model_with_polyfeature_combined_with_rowsandcols, plot_feature_importance_for_gradient_descent, compute_feature_importance_for_random_forest
from scipy.stats import normaltest
from astropy.stats import biweight_midcorrelation
import dcor
import copy
import math


#Reading the Ames dataset into a dataframe.
data = pd.read_csv('AmesHousing.csv')

#Parsing this dataframe for suitable features to be used in this analysis. 

#***NOTE*** The features below were initially selected from the entire AmesHousing dataset. These initial features were selected based on if they were numerical variables and measured on a continuous scale i.e., measured on a interval or ratio scale.  

arr1 = data['Lot Area'].values
arr3 = data['Gr Liv Area'].values  #This feature is translated to "Gross Living Area".
arr7 = data['1st Flr SF'].values   #This feature is translated to "1st Floor Square Feet".
arr9 = data['Year Built'].values
y = data['SalePrice'].values

#Imported the interpolated feature arrays which were calculated in "Interpolation_of_missing_values.py". 
filename1 = './interpolated_arrays/inter_arr2.csv'
imported_inter_Lot_Frontage_arr2 = np.loadtxt(filename1)  #There were 490 NaN values in the "Lot Frontage" feature array, accounting for 16.7% of the total number of "Lot Frontage" examples.

filename2 = './interpolated_arrays/inter_arr4.csv'
imported_inter_BsmtFin_SF_1_arr4 = np.loadtxt(filename2)  #There was 1 NaN value in the "BsmtFin SF 1" feature array. This feature is translated to "BasementFinished Square Feet 1".

filename3 = './interpolated_arrays/inter_arr5.csv'
imported_inter_Bsmt_Unf_SF_arr5 = np.loadtxt(filename3)   #There was 1 NaN value in the "Bsmt Unf SF" feature array. This feature is translated to "Basement Unfinished Square Feet".

filename4 = './interpolated_arrays/inter_arr6.csv'
imported_inter_Total_Bsmt_SF_arr6 = np.loadtxt(filename4) #There was 1 NaN value in the "Total Bsmt SF" feature array.

filename5 = './interpolated_arrays/inter_arr8.csv'
imported_inter_Garage_Area_arr8 = np.loadtxt(filename5)   #There was 1 NaN value in the "Garage Area" feature array.

#There were 0 NaN values in the remaining feature arrays. There were also 0 NaN values in the "SalePrice" target array.


#Loop to calculate the "House Age" feature array from the "Year Built" feature array. 
for i in np.arange(len(arr9)): 
        arr9[i] = 2024 - arr9[i]
        

#For more efficient computation in the context of this analysis, converted the data type of the housing price target array(y) from int64 to float64. 
y_float = y.astype(np.float64)



#Stacking the selected 1-D feature arrays into a 2-D feature array with shape (2930,3). 
#***NOTE*** These features were selected from the results of the correlation analysis conducted in "Correlation_Analysis_for_Feature_Selection.py".  
X_selected_final = np.stack((arr3, imported_inter_Total_Bsmt_SF_arr6, arr9),axis=1)






#Results from the distribution and outlier analysis from "Correlation_Analysis_for_Feature_Selection.py"
'''

Analyzing Gr Liv Area:

Normality test p-value: 0.0000
Gr Liv Area is not normally distributed (p <= 0.05)
Number of outliers detected in Gr Liv Area: 75

Analyzing Total Bsmt SF:

Normality test p-value: 0.0000
Total Bsmt SF is not normally distributed (p <= 0.05)
Number of outliers detected in Total Bsmt SF: 123

Analyzing House Age:

Normality test p-value: 0.0000
House Age is not normally distributed (p <= 0.05)
Number of outliers detected in House Age: 9

'''




#Examine the peak-to-peak ranges of the "X_selected_final" 2-D feature array to determine if scaling would be necessary. 
##print(np.ptp(X_selected_final, axis=0))

#Peak-to-Peak Ranges of each Feature in "X_selected_final" = [5308. 6110.  138.]

'Based on these differences in feature ranges, re-scaling these features would benefit model training performance.'




'''

Defining Model Training with Gradient Descent using Scaled Features Routine: 

Implemented a squared error cost function combined with a regularization term(ridge) to compute parameter gradients for gradient descent(see functions "compute_regularized_cost", "compute_regularized_gradient", and "gradient_descent_reg_with_convergence_check_for_tracking" in "Project_Tools.py"). 

Used a K-fold cross validation function with grid search functionality to obtain optimal hyperparameters for further training in cross validation and training on all feature examples. 

Applied this analysis using 4 different feature scaling methods including standard scaling(Z-score normalization) robust scaling, min-max scaling, and a combination of robust scaling with min-max scaling. 

***Note*** Even though the distributions of the feature and target variables are not normal, they do not contain many outliers so it is worth examining model performance with standard scaling which is more sensitive to outliers as it centers and scales data with the mean and standard deviation. Also, standard scaling will be used alone without combining with min-max scaling because one of the main benefits of standard scaling is that is preserves much of the distribution and variance in the data. Applying min-max scaling after introduces a risk of compressing the data too much and losing these important variance relationships between features which standard scaling aims to preserve. 

Min-max scaling is more appropriate after robust scaling because robust scaling centers and scales data using the median and IQR which is more robust to outliers but doesn't capture the full spread of data like the mean and standard deviation do. Because of this and that the range of values between features may still vary widely depending on the distribution of values outside the IQR, applying min-max scaling after can help to bring all the feature ranges to the same range without losing the resilience to outliers that robust scaling provides. 

Compared these scaled models using the average mean squared error across all folds from cross validation using the optimal hyparameters for each scaled model. 

'''






#Initial weight and bias parameter values. 
w_initial = np.array([0., 0., 0.])
b_initial = 0.




'The specific features included in this model were "Gr Liv Area", "Total Bsmt Area", and "House Age".'



'Finding the optimal gradient descent hyperparameters using the StandardScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_final, y_float, w_initial, b_initial, StandardScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.5, 'num_iters': 1000, 'lambda_': 25.0, 'avg_mse': 1875242251.305547}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_final, y_float, w_initial, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, num_iters=1000, lambda_=25.0, k=5, tol=1e-6)

#Average MSE across all folds: 1875242251.305547   

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_standard, b_final_standard, mse_final_standard, J_history_standard = retrain_on_full_data_with_scaling(X_selected_final, y_float, w_initial, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, lambda_=25.0, num_iters=1000, tol=1e-6)
##print(w_final_standard, b_final_standard, mse_final_standard)

#Final weight parameters: [40107.10203024,  21996.45472971, -25721.17062764] 
#Final bias parameter: 180796.06006825936
#Final MSE: 1833693020.6791437
#Converged at iteration 77 








'Finding the optimal gradient descent hyperparameters using the RobustScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_final, y_float, w_initial, b_initial, RobustScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 1000, 'lambda_': 10.0, 'avg_mse': 1875406279.6175983}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_final, y_float, w_initial, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=1000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1875406279.6175983   

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_robust, b_final_robust, mse_final_robust, J_history_robust = retrain_on_full_data_with_scaling(X_selected_final, y_float, w_initial, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=10.0, num_iters=1000, tol=1e-6)
##print(w_final_robust, b_final_robust, mse_final_robust)

#Final weight parameters: [49095.61739996,  25446.24555474, -39910.21759557] 
#Final bias parameter: 174523.248242139 
#Final MSE: 1833628286.8688233
#Converged at iteration 87 








'Finding the optimal gradient descent hyperparameters using the MinMaxScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_final, y_float, w_initial, b_initial, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
#print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.5, 'num_iters': 15000, 'lambda_': 0.01, 'avg_mse': 1875453991.4335735}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_scaler(X_selected_final, y_float, w_initial, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, num_iters=15000, lambda_=0.01, k=5, tol=1e-6)

#Average MSE across all folds: 1875453991.4335735

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_minmax, b_final_minmax, mse_final_minmax, J_history_minmax = retrain_on_full_data_with_scaling(X_selected_final, y_float, w_initial, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, lambda_=0.01, num_iters=15000, tol=1e-6)
##print(w_final_minmax, b_final_minmax, mse_final_minmax)

#Final weight parameters: [424359.10578815,  304759.94216821, -118083.00313646] 
#Final bias parameter: 68222.65429328391 
#Final MSE: 1833556290.6312397
#Converged at iteration 12152










'Finding the optimal gradient descent hyperparameters using a combined Robust/MinMax scaler for a fixed cross validation fold configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_combined_scaler(X_selected_final, y_float, w_initial, b_initial, RobustScaler, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
#print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.5, 'num_iters': 15000, 'lambda_': 0.01, 'avg_mse': 1875453991.4335735}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_combined_scaler(X_selected_final, y_float, w_initial, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, num_iters=15000, lambda_=0.01, k=5, tol=1e-6)

#Average MSE across all folds: 1875453991.4335735


'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled, J_history_combined_scaled = retrain_on_full_data_with_combined_scaling(X_selected_final, y_float, w_initial, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, lambda_=0.01, num_iters=15000, tol=1e-6)
##print(w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled)

#Final weight parameters: [424359.10578815,  304759.94216821, -118083.00313646] 
#Final bias parameter: 68222.6542932839 
#Final MSE: 1833556290.6312397  
#Converged at iteration 12152






final_average_mse_across_all_folds_of_each_scaled_model = {'standard scaled model': 1875242251.305547, 'robust scaled model': 1875406279.6175983, 'minmax scaled model': 1875453991.4335735, 'combined scaled model': 1875453991.4335735}

min_key = min(final_average_mse_across_all_folds_of_each_scaled_model, key=final_average_mse_across_all_folds_of_each_scaled_model.get)
##print(f"Model with minimum average MSE: {min_key}, MSE: {final_average_mse_across_all_folds_of_each_scaled_model[min_key]}")

#Model with minimum average MSE: standard scaled model, MSE: 1875242251.305547   
#Final weight parameters of standard scaled model from training on all exammples: [40107.10203024,  21996.45472971, -25721.17062764]  
#Final bias parameter of standard scaled model from training on all examples: 180796.06006825936 
#Final MSE of standard scaled model from training on all examples: 1833693020.6791437


'''
Plot the distribution of coordinates of each selected standard scaled feature vs the housing price target(see "./Visualizations/Scatter_Plots/Fig2_Scatter_of_Target_vs_each_Selected_Standard_Scaled_Feature").
''' 

#Scale all examples with Standard scaler. 
standard_scaler = StandardScaler()
X_selected_final_standard_scaled = standard_scaler.fit_transform(X_selected_final) 

#Scatterplot function.
##plot_feature_vs_target_scatter(X_selected_final_standard_scaled, y_float, feature_names=['Gross Living Area(sqft)','Total Basement Area(sqft)','House Age(years)'], target_name="Sale Price($dollars)", title='Figure 2: Scatterplots of each Selected Standard Scaled Feature vs the Housing Price Target')


'''

Results: 

This analysis indicates that implementing standard scaling is optimal for training a model using gradient descent with the initially selected features. 

However, this model is not highly predictive and may indicate underfitting due to the high error values across all scaling methods. Further analysis is required to obtain better model performance. 

'''














'''

Feature Engineering Analysis: 


Part 1 - Model Training with Composite Feature. 

There appears to be a mild, sparse spread of zeros in the coordinate distribution of the "Total Bsmt Area" feature against the housing price target(see "./Visualizations/Scatter_Plots/Fig1_Scatter_of_Target_vs_each_Selected_Feature"). There are 79 values of zeros in the "Total Bsmt Area" feature array. 

Although this is only 2.7% of the total feature examples, it is worth adding the "Total Bsmt Area" feature with features which also had moderate to strong correlations with the housing price target to create a composite feature with a more continuous coordinate distribution of feature values against the housing price target. The "1st Flr SF" and "Garage Area" features will be added to the "Total Bsmt Area" feature because they had a moderate linear correlation with the housing price target and were measured in the same units as the basement area which is important for maintaining interpretability.

Because "1st Flr SF" had a strong correlation with "Total Bsmt Area"(biweight=0.8204808801718345) the individual features will not be included seperately to reduce multicollinearity between these features. The regularization term, which has been implemented in model training so far, can help mitigate this multicollinearity amoung features. However, this correlation is too strong to include these features individually in a linear model.(see "./Visualizations/Scatter_Plots/Fig3_Scatter_of_each_Selected_Feature_including_Composite_Feature_vs_Target"). 




Part 2 - Model Training with Interaction Term. 

The "House Age" feature had a moderate, negative linear correlation(biweight= -0.5193176556071337) with the "Garage Area" feature. This indicates a moderate trend in Ames, Iowa where more recently built homes(low "House Age") are being built with large garages or atleast had a large garage at the time the data was sampled(garage could have been renovated). It is worth examining the multiplicative interaction between "House Age" and "Garage Area" to gain more insight into whether home buyers potentially prefer recently built homes with large garages and if they are willingly to pay more for this combination. 

Although, "House Age" is moderately correlated to "Garage Area" the individual "House Age" and "Garage Area" features will be included in the model to examine whether these features still have effects on the housing price target regardless of their combined effect. The regularization implementation that will be continued to applied in training these models will help mitigate any multicollinearity between features. 





Part 3 - Polynomial Feature Transformations.

Polynomial feature transformations may also increase model performance. Specifically for the "House Age" feature which appears to exhibit a slightly quadratic relationship against the housing price target(see "./Visualizations/Scatter_Plots/Fig3_Scatter_of_each_Selected_Feature_including_Composite_Feature_vs_Target"). 



'''




'''Combined the 'Total Bsmt Area', '1st Flr SF', and 'Garage Area' feature arrays to create a new composite feature which may exibit a moderate to strong correlations with the housing price target and retain feature independence amoung other features.'''

#Creating composite feature array with shape (2930,).
composite_feat_arr = np.sum([imported_inter_Total_Bsmt_SF_arr6, arr7, imported_inter_Garage_Area_arr8], axis=0)


#Stacked this new composite array with the other selected feature arrays to create a new 2-D feature array with shape (2930,3).
X_selected_with_composite = np.stack((arr3, composite_feat_arr, arr9),axis=1)



'''
Plot the distribution of coordinates of each selected feature including composite feature vs the housing price target(see "./Visualizations/Scatter_Plots/Fig3_Scatter_of_each_Selected_Feature_including_Composite_Feature_vs_Target").
'''
#Scatterplot function.
##plot_feature_vs_target_scatter(X_selected_with_composite, y_float, feature_names=['Gross Living Area(sqft)','Composite: "Basement + 1st floor + Garage" Area(sqft)','House Age(years)'], target_name='Sale Price($dollars)', title='Figure 3: Scatterplots of each Selected Feature including Composite Feature vs the Housing Price Target')






'First analyzed the distribution of values of each feature in the new "X_selected_with_composite" feature array.'

#Converting composite feature array into dataframe with column labels. 
dframe = pd.DataFrame(X_selected_with_composite, columns=['Gross Living Area(sqft)','Total Basement, 1st floor, and Garage Area(sqft)','House Age(years)'])

#Function to examine the normality of distributions and count outliers of a dataframe.
##analyze_data(dframe)


'''

Analyzing Gross Living Area(sqft):

Normality test p-value: 0.0000
Gross Living Area(sqft) is not normally distributed (p <= 0.05)
Number of outliers detected in Gross Living Area(sqft): 75

Analyzing Total Basement, 1st floor, and Garage Area(sqft):

Normality test p-value: 0.0000
Total Basement, 1st floor, and Garage Area(sqft) is not normally distributed (p <= 0.05)
Number of outliers detected in Total Basement, 1st floor, and Garage Area(sqft): 44

Analyzing House Age(years):

Normality test p-value: 0.0000
House Age(years) is not normally distributed (p <= 0.05)
Number of outliers detected in House Age(years): 9






This new composite feature and the previously selected features do not exibit a normal distribution. The new composite feature also has a low number of outliers(1.5% of all examples). The previously implemented scaling methods and correlation metrics will continue to be utilized.


'''





'Quantified any linear or non-linear correlations between the housing price target and the new composite feature.'


bicor_val = biweight_midcorrelation(y_float, X_selected_with_composite[:,1])
##print(bicor_val)

#Biweight midcorrelation coefficient = 0.6918983054888151

distcor_val = dcor.distance_correlation(y_float, X_selected_with_composite[:,1])
##print(distcor_val)

#Distance correlation coefficient = 0.6923201273595997





'Calculated the strength and direction of linear correlations among features using the biweight correlation coefficient. Used this same loop to compute all combinations of features i.e., changed the indexing in the first argument of the biweight midcorrelation function.'

n_selected_with_composite = X_selected_with_composite.shape[1]
among_features_lincorr_list = []
for i in range(n_selected_with_composite):
    bicor_value = biweight_midcorrelation(X_selected_with_composite[:,1], X_selected_with_composite[:,i])
    among_features_lincorr_list.append(bicor_value)

##print(among_features_lincorr_list)



'''
Biweight Correlation Coefficients: 



Gr Liv Area vs Composite Feature = 0.509159623383186

Gr Liv Area vs House Age = -0.2862394916273845



Composite Feature vs Gr Liv Area = 0.509159623383186

Composite Feature vs House Age = -0.44722435982988723



House Age vs Gr Liv Area = -0.2862394916273845

House Age vs Composite Feature = -0.44722435982988723


'''





'Calculated the strength of linear or non-linear correlations among features using the distance correlation coefficient. Used this same loop to compute all combinations of features i.e., changed the indexing in the first argument of the distance correlation function.'

among_features_nonlincorr_list = []
for i in range(n_selected_with_composite):
    distcor_value = dcor.distance_correlation(X_selected_with_composite[:,2], X_selected_with_composite[:,i])
    among_features_nonlincorr_list.append(distcor_value)
    
##print(among_features_nonlincorr_list)



'''
Distance Correlation Coefficients: 



Gr Liv Area vs Composite Feature = 0.49150843799482796

Gr Liv Area vs House Age = 0.33224213314454815



Composite Feature vs Gr Liv Area = 0.49150843799482796

Composite Feature vs House Age = 0.4399974173808199



House Age vs Gr Liv Area = 0.33224213314454815

House Age vs Composite Feature = 0.4399974173808199







Results: 


The new composite feature had a fairly strong positive linear correlation with the housing price target(biweight = 0.6918983054888151). 

The new composite feature also had a moderate positive linear relationship with the "Gr Liv Area" feature(biweight = 0.509159623383186). 

However, it is worth examining how models perform using this composite feature because appropriate features for regression analysis are limited in the Ames Housing dataset and the composite feature does exhibit a fairly strong linear correlation with the housing price target. 

It is also worth noting that this correlation between the composite feature and the "Gr Liv Area" feature had a distance correlation coefficient just below the coefficient threshold of 0.5 with a value of 0.49150843799482796. 



'''













'''

Feature Engineering Analysis Part 1 - Model Training with Composite Features:


The specific features included in these models were "Gr Liv Area", composite("Total Bsmt Area"+"1st Flr SF"+"Garage Area"), and "House Age".

'''

#Examine the peak-to-peak ranges of the "X_selected_with_composite" 2-D feature array to determine if scaling would be necessary. 
##print(np.ptp(X_selected_with_composite, axis=0))

#Peak-to-Peak Ranges of each Feature in "X_selected_with_composite" = [ 5308. 11886.   138.]

'Based on these differences in feature ranges, re-scaling these features would benefit model training performance.'







'Implemented the same model training with feature scaling routine that was initially conducted on the "X_selected_final" features:'


'Finding the optimal gradient descent hyperparameters using the StandardScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite, y_float, w_initial, b_initial, StandardScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 1000, 'lambda_': 25.0, 'avg_mse': 1801895027.3376033}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite, y_float, w_initial, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=1000, lambda_=25.0, k=5, tol=1e-6)

#Average MSE across all folds: 1801895027.3376033  

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_standard, b_final_standard, mse_final_standard, J_history_standard = retrain_on_full_data_with_scaling(X_selected_with_composite, y_float, w_initial, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=25.0, num_iters=1000, tol=1e-6)
##print(w_final_standard, b_final_standard, mse_final_standard)

#Final weight parameters: [35319.90380356,  26457.4731165,  -24215.02042908] 
#Final bias parameter: 180796.06006825936 
#Final MSE: 1767166878.351205
#Converged at iteration 58 








'Finding the optimal gradient descent hyperparameters using the RobustScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite, y_float, w_initial, b_initial, RobustScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 1000, 'lambda_': 10.0, 'avg_mse': 1801950365.3398137}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite, y_float, w_initial, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=1000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1801950365.3398137  

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_robust, b_final_robust, mse_final_robust, J_history_robust = retrain_on_full_data_with_scaling(X_selected_with_composite, y_float, w_initial, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=10.0, num_iters=1000, tol=1e-6)
##print(w_final_robust, b_final_robust, mse_final_robust)

#Final weight parameters: [43231.92486388,  33202.82425458, -37580.05555051] 
#Final bias parameter: 172568.63524036735 
#Final MSE: 1767115184.8725142
#Converged at iteration 113








'Finding the optimal gradient descent hyperparameters using the MinMaxScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite, y_float, w_initial, b_initial, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.5, 'num_iters': 15000, 'lambda_': 0.1, 'avg_mse': 1802004581.9571674}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_scaler(X_selected_with_composite, y_float, w_initial, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, num_iters=15000, lambda_=0.1, k=5, tol=1e-6)

#Average MSE across all folds: 1802004581.9571674

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_minmax, b_final_minmax, mse_final_minmax, J_history_minmax = retrain_on_full_data_with_scaling(X_selected_with_composite, y_float, w_initial, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, lambda_=0.1, num_iters=15000, tol=1e-6)
##print(w_final_minmax, b_final_minmax, mse_final_minmax)

#Final weight parameters: [372887.4747242,  340586.9659812,  -111395.37091687] 
#Final bias parameter: 62769.04347066796 
#Final MSE: 1767085754.212672
#Converged at iteration 13784










'Finding the optimal gradient descent hyperparameters using a combined Robust/MinMax scaler for a fixed cross validation fold configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_combined_scaler(X_selected_with_composite, y_float, w_initial, b_initial, RobustScaler, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.5, 'num_iters': 15000, 'lambda_': 0.1, 'avg_mse': 1802004581.9571674}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_combined_scaler(X_selected_with_composite, y_float, w_initial, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, num_iters=15000, lambda_=0.1, k=5, tol=1e-6)

#Average MSE across all folds: 1802004581.9571674


'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled, J_history_combined_scaled = retrain_on_full_data_with_combined_scaling(X_selected_with_composite, y_float, w_initial, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, lambda_=0.1, num_iters=15000, tol=1e-6)
##print(w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled)

#Final weight parameters: [372887.4747242,  340586.9659812,  -111395.37091687] 
#Final bias parameter: 62769.04347066799 
#Final MSE: 1767085754.212672  
#Converged at iteration 13784






final_average_mse_across_all_folds_of_each_scaled_model_with_compositefeature = {'standard scaled model': 1801895027.3376033, 'robust scaled model': 1801950365.3398137 , 'minmax scaled model': 1802004581.9571674, 'combined scaled model': 1802004581.9571674}

min_key_2 = min(final_average_mse_across_all_folds_of_each_scaled_model_with_compositefeature, key=final_average_mse_across_all_folds_of_each_scaled_model_with_compositefeature.get)
##print(f"Model with minimum average MSE: {min_key_2}, MSE: {final_average_mse_across_all_folds_of_each_scaled_model_with_compositefeature[min_key_2]}")

#Model with minimum average MSE: standard scaled model, MSE: 1801895027.3376033
#Final weight parameters of standard scaled model from training on all exammples: [35319.90380356,  26457.4731165,  -24215.02042908] 
#Final bias parameter of standard scaled model from training on all examples: 180796.06006825936 
#Final MSE of standard scaled model from training on all examples: 1767166878.351205











''' 

Removed the "1st Flr SF" feature from the composite array because it was used to calculate the "Gr Liv Area" feature in the Ames Housing dataset and is highly correlated with the "Total Bsmt Area" feature. The second composite array included the "Total Bsmt SF" and "Garage Area" features. Stacked them into a new 2-D feature array with shape (2930,3). 

The specific features included in this model were "Gr Liv Area", composite_2("Total Bsmt Area"+"Garage Area"), and "House Age".
 

'''
#Constructed composite_2 feature array with shape (2930,).
composite_2_feat_arr = np.sum([imported_inter_Total_Bsmt_SF_arr6, imported_inter_Garage_Area_arr8], axis=0)

#Stacked features into a new 2-D feature array with shape (2930,3).
X_selected_with_composite_2 = np.stack((arr3, composite_2_feat_arr, arr9),axis=1)


#Examine the peak-to-peak ranges of the "X_selected_with_composite_2" 2-D feature array to determine if scaling would be necessary. 
##print(np.ptp(X_selected_with_composite_2, axis=0))

#Peak-to-Peak Ranges of each Feature in "X_selected_with_composite_2" = [5308. 7528.  138.]

'Based on these differences in feature ranges, re-scaling these features would benefit model training performance.'









'Implemented the same model training with feature scaling routine that was initially conducted on the "X_selected_final" features:'



'Finding the optimal gradient descent hyperparameters using the StandardScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_2, y_float, w_initial, b_initial, StandardScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 1000, 'lambda_': 25.0, 'avg_mse': 1759272140.677011}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite_2, y_float, w_initial, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=1000, lambda_=25.0, k=5, tol=1e-6)

#Average MSE across all folds: 1759272140.677011  


'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_standard, b_final_standard, mse_final_standard, J_history_standard = retrain_on_full_data_with_scaling(X_selected_with_composite_2, y_float, w_initial, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=25.0, num_iters=1000, tol=1e-6)
##print(w_final_standard, b_final_standard, mse_final_standard)

#Final weight parameters: [36406.84802937,  27675.13488502, -22032.23989218] 
#Final bias parameter: 180796.0600682594 
#Final MSE: 1723444000.2756617
#Converged at iteration 56 








'Finding the optimal gradient descent hyperparameters using the RobustScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_2, y_float, w_initial, b_initial, RobustScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 1000, 'lambda_': 10.0, 'avg_mse': 1759421283.0663903}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite_2, y_float, w_initial, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=1000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1759421283.0663903   

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_robust, b_final_robust, mse_final_robust, J_history_robust = retrain_on_full_data_with_scaling(X_selected_with_composite_2, y_float, w_initial, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=10.0, num_iters=1000, tol=1e-6)
##print(w_final_robust, b_final_robust, mse_final_robust)

#Final weight parameters: [44541.6656339,  32516.52900921, -34152.99797885] 
#Final bias parameter: 173769.77729605764 
#Final MSE: 1723387160.3412282
#Converged at iteration 109








'Finding the optimal gradient descent hyperparameters using the MinMaxScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_2, y_float, w_initial, b_initial, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.5, 'num_iters': 15000, 'lambda_': 0.01, 'avg_mse': 1759574803.6932578}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_scaler(X_selected_with_composite_2, y_float, w_initial, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, num_iters=15000, lambda_=0.01, k=5, tol=1e-6)

#Average MSE across all folds: 1759574803.6932578

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_minmax, b_final_minmax, mse_final_minmax, J_history_minmax = retrain_on_full_data_with_scaling(X_selected_with_composite_2, y_float, w_initial, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, lambda_=0.01, num_iters=15000, tol=1e-6)
##print(w_final_minmax, b_final_minmax, mse_final_minmax)

#Final weight parameters: [384996.31881586,  361863.50813988, -101035.74361108] 
#Final bias parameter: 51273.25201249696 
#Final MSE: 1723332719.9882076
#Converged at iteration 13837










'Finding the optimal gradient descent hyperparameters using a combined Robust/MinMax scaler for a fixed cross validation fold configuration(k). Initialized weight and bias parameters at 0.'
##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_combined_scaler(X_selected_with_composite_2, y_float, w_initial, b_initial, RobustScaler, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.5, 'num_iters': 15000, 'lambda_': 0.01, 'avg_mse': 1759574803.6932578}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_combined_scaler(X_selected_with_composite, y_float, w_initial, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, num_iters=15000, lambda_=0.01, k=5, tol=1e-6)

#Average MSE across all folds: 1802097348.6843953


'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled, J_history_combined_scaled = retrain_on_full_data_with_combined_scaling(X_selected_with_composite_2, y_float, w_initial, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, lambda_=0.01, num_iters=15000, tol=1e-6)
##print(w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled)

#Final weight parameters: [384996.31881586,  361863.50813988, -101035.74361108] 
#Final bias parameter: 51273.25201249696 
#Final MSE: 1723332719.9882076  
#Converged at iteration 13837






final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_feature = {'standard scaled model': 1759272140.677011, 'robust scaled model': 1759421283.0663903, 'minmax scaled model': 1759574803.6932578, 'combined scaled model': 1802097348.6843953}

min_key_3 = min(final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_feature, key=final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_feature.get)
##print(f"Model with minimum average MSE: {min_key_3}, MSE: {final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_feature[min_key_3]}")

#Model with minimum average MSE: standard scaled model, MSE: 1759272140.677011
#Final weight parameters of standard scaled model from training on all exammples: [36406.84802937,  27675.13488502, -22032.23989218] 
#Final bias parameter of standard scaled model from training on all examples: 180796.0600682594  
#Final MSE of standard scaled model from training on all examples: 1723444000.2756617







'''
Feature Engineering Analysis Part 2 - Model Training with Interaction Term:

The multiplicative interaction term("House Age" * "Garage Area") will be implemented in the models which include the composite and composite_2 feature.

The specific features included in this model were "Gr Liv Area", "composite("Total Bsmt Area"+"1st Flr SF"+"Garage Area")", "Garage Area", "House Age", and "interaction("House Age" * "Garage Area")".

'''

#Created a new multiplicative interaction term with the "House Age" and "Garage Area" features. 
Garage_Area_and_House_Age_interaction = imported_inter_Garage_Area_arr8 * arr9 

#Stacked all these features into new 2-D feature array with shape (2930,5)
X_selected_with_composite_and_interaction = np.stack((arr3, composite_feat_arr, imported_inter_Garage_Area_arr8, arr9, Garage_Area_and_House_Age_interaction),axis=1)

w_initial_2 = np.array([0., 0., 0., 0., 0.])

#Examine the peak-to-peak ranges of the "X_selected_with_composite_and_interaction" 2-D feature array to determine if scaling would be necessary. 
##print(np.ptp(X_selected_with_composite_and_interaction, axis=0))

#Peak-to-Peak Ranges of each Feature in "X_selected_with_composite_and_interaction" = [5.30800e+03 1.18860e+04 1.48800e+03 1.38000e+02 1.51776e+05]

'Based on these differences in feature ranges, re-scaling these features would benefit model training performance.'





'Implemented the same model training with feature scaling routine that was initially conducted on the "X_selected_final" features:'


'Finding the optimal gradient descent hyperparameters using the StandardScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, StandardScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.5, 'num_iters': 1000, 'lambda_': 10.0, 'avg_mse': 1651977046.5778615}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, num_iters=1000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1651977046.5778615  

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_standard, b_final_standard, mse_final_standard, J_history_standard = retrain_on_full_data_with_scaling(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, lambda_=10.0, num_iters=1000, tol=1e-6)
##print(w_final_standard, b_final_standard, mse_final_standard)

#Final weight parameters: [32750.90531065,  19482.92605986,  29334.10662894,  -1348.069121, -23077.9764574 ] 
#Final bias parameter: 180796.06006825936 
#Final MSE: 1613586787.406604
#Converged at iteration 580 







'Finding the optimal gradient descent hyperparameters using the RobustScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 1000, 'lambda_': 10.0, 'avg_mse': 1651838884.277936}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=1000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1651838884.277936  

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_robust, b_final_robust, mse_final_robust, J_history_robust = retrain_on_full_data_with_scaling(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=10.0, num_iters=1000, tol=1e-6)
##print(w_final_robust, b_final_robust, mse_final_robust)

#Final weight parameters: [39917.50545036,  24425.34294024,  34790.22439584,  -2255.59751654, -22567.56697379] 
#Final bias parameter: 178349.22744621927 
#Final MSE: 1613635657.932696
#Converged at iteration 574








'Finding the optimal gradient descent hyperparameters using the MinMaxScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 20000, 'lambda_': 1.0, 'avg_mse': 1656829334.0645185}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_scaler(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=20000, lambda_=1.0, k=5, tol=1e-6)

#Average MSE across all folds: 1656829334.0645185

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_minmax, b_final_minmax, mse_final_minmax, J_history_minmax = retrain_on_full_data_with_scaling(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=1.0, num_iters=20000, tol=1e-6)
##print(w_final_minmax, b_final_minmax, mse_final_minmax)

#Final weight parameters: [336985.28217013,  242696.29482202,  186084.94102846,  -22282.17592459, -199974.07164262] 
#Final bias parameter: 34605.986043257726 
#Final MSE: 1619161859.0127215
#Converged at iteration 16141







'Finding the optimal gradient descent hyperparameters using a combined Robust/MinMax scaler for a fixed cross validation fold configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_combined_scaler(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 20000, 'lambda_': 1.0, 'avg_mse': 1656829334.0645185}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_combined_scaler(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=20000, lambda_=1.0, k=5, tol=1e-6)

#Average MSE across all folds: 1656829334.0645185


'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled, J_history_combined_scaled = retrain_on_full_data_with_combined_scaling(X_selected_with_composite_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=1.0, num_iters=20000, tol=1e-6)
##print(w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled)

#Final weight parameters: [336985.28217013,  242696.29482202,  186084.94102846,  -22282.17592459, -199974.07164262] 
#Final bias parameter: 34605.98604325772 
#Final MSE: 1619161859.0127215  
#Converged at iteration 16141






final_average_mse_across_all_folds_of_each_scaled_model_with_composite_and_interaction = {'standard scaled model': 1651977046.5778615, 'robust scaled model': 1651838884.277936 , 'minmax scaled model': 1656829334.0645185, 'combined scaled model': 1656829334.0645185}

min_key_4 = min(final_average_mse_across_all_folds_of_each_scaled_model_with_composite_and_interaction, key=final_average_mse_across_all_folds_of_each_scaled_model_with_composite_and_interaction.get)
##print(f"Model with minimum average MSE: {min_key_4}, MSE: {final_average_mse_across_all_folds_of_each_scaled_model_with_composite_and_interaction[min_key_4]}")

#Model with minimum average MSE: robust scaled model, MSE: 1651838884.277936
#Final weight parameters of standard scaled model from training on all exammples: [39917.50545036,  24425.34294024,  34790.22439584,  -2255.59751654, -22567.56697379]
#Final bias parameter of standard scaled model from training on all examples: 178349.22744621927 
#Final MSE of standard scaled model from training on all examples: 1613635657.932696







'''

The specific features included in this model were "Gr Liv Area", "composite_2("Total Bsmt Area" + "Garage Area")", "Garage Area", "House Age", and "interaction("House Age" * "Garage Area")".

'''


#Stacked all these features into new 2-D feature array with shape (2930,5)
X_selected_with_composite_2_and_interaction = np.stack((arr3, composite_2_feat_arr, imported_inter_Garage_Area_arr8, arr9, Garage_Area_and_House_Age_interaction),axis=1)




'Implemented the same model training with feature scaling routine that was initially conducted on the "X_selected_final" features:'


'Finding the optimal gradient descent hyperparameters using the StandardScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, StandardScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.5, 'num_iters': 1000, 'lambda_': 10.0, 'avg_mse': 1648582628.0601}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, num_iters=1000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1648582628.0601  

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_standard, b_final_standard, mse_final_standard, J_history_standard = retrain_on_full_data_with_scaling(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, lambda_=10.0, num_iters=1000, tol=1e-6)
##print(w_final_standard, b_final_standard, mse_final_standard)

#Final weight parameters: [34410.94948131,  21452.70954953,  25472.48412589,  -1315.09909116, -22227.10830729] 
#Final bias parameter: 180796.06006825936 
#Final MSE: 1607199802.9947753
#Converged at iteration 596 







'Finding the optimal gradient descent hyperparameters using the RobustScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 1000, 'lambda_': 10.0, 'avg_mse': 1648435620.017269}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=1000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1648435620.017269  

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_robust, b_final_robust, mse_final_robust, J_history_robust = retrain_on_full_data_with_scaling(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=10.0, num_iters=1000, tol=1e-6)
##print(w_final_robust, b_final_robust, mse_final_robust)

#Final weight parameters: [41929.35432065,  25158.99757497,  30207.81024399,  -2172.0518841, -21747.55958265] 
#Final bias parameter: 178780.3842269164 
#Final MSE: 1607240786.4494777
#Converged at iteration 584








'Finding the optimal gradient descent hyperparameters using the MinMaxScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 20000, 'lambda_': 1.0, 'avg_mse': 1654737439.4530072}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_scaler(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=20000, lambda_=1.0, k=5, tol=1e-6)

#Average MSE across all folds: 1654737439.4530072

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_minmax, b_final_minmax, mse_final_minmax, J_history_minmax = retrain_on_full_data_with_scaling(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=1.0, num_iters=20000, tol=1e-6)
##print(w_final_minmax, b_final_minmax, mse_final_minmax)

#Final weight parameters: [353568.00840718,  263063.58483797,  165139.48999665,  -20899.56100594, -195358.85294136] 
#Final bias parameter: 31285.8409403431 
#Final MSE: 1612847790.1416779
#Converged at iteration 16573







'Finding the optimal gradient descent hyperparameters using a combined Robust/MinMax scaler for a fixed cross validation fold configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_combined_scaler(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 20000, 'lambda_': 1.0, 'avg_mse': 1654737439.4530072}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_combined_scaler(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=20000, lambda_=1.0, k=5, tol=1e-6)

#Average MSE across all folds: 1654737439.4530072


'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled, J_history_combined_scaled = retrain_on_full_data_with_combined_scaling(X_selected_with_composite_2_and_interaction, y_float, w_initial_2, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=1.0, num_iters=20000, tol=1e-6)
##print(w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled)

#Final weight parameters: [353568.00840718,  263063.58483797,  165139.48999665,  -20899.56100594, -195358.85294136] 
#Final bias parameter: 31285.84094034309 
#Final MSE: 1612847790.1416779  
#Converged at iteration 16573






final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_and_interaction = {'standard scaled model': 1648582628.0601, 'robust scaled model': 1648435620.017269 , 'minmax scaled model': 1654737439.4530072, 'combined scaled model': 1654737439.4530072}

min_key_5 = min(final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_and_interaction, key=final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_and_interaction.get)
##print(f"Model with minimum average MSE: {min_key_5}, MSE: {final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_and_interaction[min_key_5]}")

#Model with minimum average MSE: robust scaled model, MSE: 1648435620.017269
#Final weight parameters of standard scaled model from training on all exammples: [41929.35432065,  25158.99757497,  30207.81024399,  -2172.0518841, -21747.55958265]
#Final bias parameter of standard scaled model from training on all examples: 178780.3842269164 
#Final MSE of standard scaled model from training on all examples: 1607240786.4494777












'''

Feature Engineering Analysis Part 3 - Polynomial Feature Transformations:

This quadratic transformation on the "House Age" feature will be implemented in the models which include the composite with interaction and composite_2 with interaction.

The specific features included in this model were "Gr Liv Area", "composite("Total Bsmt Area"+"1st Flr SF"+"Garage Area")", "Garage Area", "House Age", "House Age Squared", and the interaction("House Age" * "Garage Area").

'''



#Transformed the "House Age" feature using a quadratic transformation. 
arr9_squared = arr9**2


#Stacked all these features into new 2-D feature array with shape (2930,6)
X_selected_with_composite_interaction_and_polyfeature = np.stack((arr3, composite_feat_arr, imported_inter_Garage_Area_arr8, arr9, arr9_squared, Garage_Area_and_House_Age_interaction),axis=1)

#Initial weight and bias parameter values. 
w_initial_3 = np.array([0., 0., 0., 0., 0., 0.])



#Examine the peak-to-peak ranges of the "X_selected_with_composite_interaction_and_polyfeature" 2-D feature array to determine if scaling would be necessary. 
##print(np.ptp(X_selected_with_composite_interaction_and_polyfeature, axis=0))

#Peak-to-Peak Ranges of each Feature in "X_selected_with_composite_interaction_and_polyfeature" = [5.30800e+03 1.18860e+04 1.48800e+03 1.38000e+02 2.29080e+04 1.51776e+05]

'Based on these differences in feature ranges, re-scaling these features would benefit model training performance.'






'Implemented the same model training with feature scaling routine that was initially conducted on the "X_selected_final" features:'


'Finding the optimal gradient descent hyperparameters using the StandardScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, StandardScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.5, 'num_iters': 5000, 'lambda_': 10.0, 'avg_mse': 1628469596.294723}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, num_iters=5000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1628469596.294723


'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_standard, b_final_standard, mse_final_standard, J_history_standard = retrain_on_full_data_with_scaling(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, lambda_=10.0, num_iters=5000, tol=1e-6)
##print(w_final_standard, b_final_standard, mse_final_standard)

#Final weight parameters: [31407.76439351,  20410.43071129,  25835.96670591, -24335.53562665, 20396.29515505, -19924.40444393] 
#Final bias parameter: 180796.06006825936 
#Final MSE: 1586878493.8827815
#Converged at iteration 1782








'Finding the optimal gradient descent hyperparameters using the RobustScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.1, 'num_iters': 15000, 'lambda_': 10.0, 'avg_mse': 1628897706.8240771}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.1, num_iters=15000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1628897706.8240771  

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_robust, b_final_robust, mse_final_robust, J_history_robust = retrain_on_full_data_with_scaling(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.1, lambda_=10.0, num_iters=15000, tol=1e-6)
##print(w_final_robust, b_final_robust, mse_final_robust)

#Final weight parameters: [38414.34489334,  25432.49173688,  31578.30329166, -33413.87481787, 20658.19937655, -20217.48085975] 
#Final bias parameter: 173750.339873723
#Final MSE: 1587699474.346245
#Converged at iteration 14291 








'Finding the optimal gradient descent hyperparameters using the MinMaxScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 5000, 'lambda_': 10.0, 'avg_mse': 1836422494.970696}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_scaler(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=5000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1836422494.970696

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero. ***Note*** increased iterations to 6000 because the gradients did not converge under 5000 iterations when training on all examples'

##w_final_minmax, b_final_minmax, mse_final_minmax, J_history_minmax = retrain_on_full_data_with_scaling(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=10.0, num_iters=6000, tol=1e-6)
##print(w_final_minmax, b_final_minmax, mse_final_minmax)

#Final weight parameters: [265902.6834158,  181530.12161333, 143256.31008042, -96515.9674721, 29036.90525281, -82605.67131528] 
#Final bias parameter: 75450.68704614438 
#Final MSE: 1764980832.5350685
#Converged at iteration 5400










'Finding the optimal gradient descent hyperparameters using a combined Robust/MinMax scaler for a fixed cross validation fold configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_combined_scaler(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 5000, 'lambda_': 10.0, 'avg_mse': 1836422494.970696}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_combined_scaler(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=5000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1836422494.970696  


'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero. ***Note*** increased iterations to 6000 because the gradients did not converge under 5000 iterations when training on all examples'

##w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled, J_history_combined_scaled = retrain_on_full_data_with_combined_scaling(X_selected_with_composite_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=10.0, num_iters=6000, tol=1e-6)
##print(w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled)

#Final weight parameters: [265902.6834158,  181530.12161333, 143256.31008042, -96515.9674721, 29036.90525281, -82605.67131528] 
#Final bias parameter: 75450.68704614438 
#Final MSE: 1764980832.535069  
#Converged at iteration 5400





final_average_mse_across_all_folds_of_each_scaled_model_with_composite_interaction_and_polyfeature = {'standard scaled model': 1628469596.294723, 'robust scaled model': 1628897706.8240771, 'minmax scaled model': 1836422494.970696, 'combined scaled model': 1836422494.970696}

min_key_6 = min(final_average_mse_across_all_folds_of_each_scaled_model_with_composite_interaction_and_polyfeature, key=final_average_mse_across_all_folds_of_each_scaled_model_with_composite_interaction_and_polyfeature.get)
##print(f"Model with minimum average MSE: {min_key_6}, MSE: {final_average_mse_across_all_folds_of_each_scaled_model_with_composite_interaction_and_polyfeature[min_key_6]}")

#Model with minimum average MSE: standard scaled model, MSE: 1628469596.294723
#Final weight parameters of standard scaled model from training on all exammples: [31407.76439351,  20410.43071129,  25835.96670591, -24335.53562665, 20396.29515505, -19924.40444393]
#Final bias parameter of standard scaled model from training on all examples: 180796.06006825936 
#Final MSE of standard scaled model from training on all examples: 1586878493.8827815







'''

Added the quadratic term for the "House Age" feature to the "X_selected_with_composite_2_and_interaction" model. 

The specific features included in this model were "Gr Liv Area", composite_2("Total Bsmt Area" + "Garage Area"), "Garage Age" "House Age", "House Age Squared", and the interaction("House Age" * "Garage Area"). 


'''


#Stacked all these features into new 2-D feature array with shape (2930,6)
X_selected_with_composite_2_interaction_and_polyfeature = np.stack((arr3, composite_2_feat_arr, imported_inter_Garage_Area_arr8, arr9, arr9_squared, Garage_Area_and_House_Age_interaction),axis=1)







'Implemented the same model training with feature scaling routine that was initially conducted on the "X_selected_final" features:'




'Finding the optimal gradient descent hyperparameters using the StandardScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, StandardScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.5, 'num_iters': 5000, 'lambda_': 25.0, 'avg_mse': 1633448073.935783}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, num_iters=5000, lambda_=25.0, k=5, tol=1e-6)

#Average MSE across all folds: 1633448073.935783


'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_standard, b_final_standard, mse_final_standard, J_history_standard = retrain_on_full_data_with_scaling(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, StandardScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.5, lambda_=25.0, num_iters=5000, tol=1e-6)
##print(w_final_standard, b_final_standard, mse_final_standard)

#Final weight parameters: [33440.24048273,  21906.79097603,  22649.34366006, -18272.96889341, 14752.38725995, -19557.9774121] 
#Final bias parameter: 180796.06006825936 
#Final MSE: 1589273829.2782097
#Converged at iteration 1416








'Finding the optimal gradient descent hyperparameters using the RobustScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)


#Optimal Hyperparameters: {'alpha': 0.1, 'num_iters': 15000, 'lambda_': 10.0, 'avg_mse': 1633580705.04759}


'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0.'

##k_fold_cross_validation_with_scaler(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.1, num_iters=15000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1633580705.04759   

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_robust, b_final_robust, mse_final_robust, J_history_robust = retrain_on_full_data_with_scaling(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.1, lambda_=10.0, num_iters=15000, tol=1e-6)
##print(w_final_robust, b_final_robust, mse_final_robust)

#Final weight parameters: [40820.54747538,  25595.56885325,  27690.8217458,  -28292.13008991, 17342.09764063, -19837.40785654] 
#Final bias parameter: 175006.62922615727 
#Final MSE: 1588859051.5539494
#Converged at iteration 13426 








'Finding the optimal gradient descent hyperparameters using the MinMaxScaler for a fixed cross validation configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_scaler(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 5000, 'lambda_': 10.0, 'avg_mse': 1852787393.6371064}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_scaler(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=5000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1852787393.6371064

'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.***Note*** increased iterations to 6000 because the gradients did not converge under 5000 iterations when training on all examples.'

##w_final_minmax, b_final_minmax, mse_final_minmax, J_history_minmax = retrain_on_full_data_with_scaling(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=10.0, num_iters=6000, tol=1e-6)
##print(w_final_minmax, b_final_minmax, mse_final_minmax)

#Final weight parameters: [276150.72510091, 178622.44214051, 137848.02508316, -92720.72470812, 26381.9465485,  -83779.53604989] 
#Final bias parameter: 74149.40367663387 
#Final MSE: 1777064469.1029158
#Converged at iteration 5225










'Finding the optimal gradient descent hyperparameters using a combined Robust/MinMax scaler for a fixed cross validation fold configuration(k). Initialized weight and bias parameters at 0.'

##best_parameters,_,_ = cross_validation_and_grid_search_with_convergence_tracking_with_combined_scaler(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, MinMaxScaler, compute_regularized_gradient, gradient_descent_reg_with_convergence_check_for_tracking, k=5, tol=1e-6)
##print(best_parameters)

#Optimal Hyperparameters: {'alpha': 0.8, 'num_iters': 5000, 'lambda_': 10.0, 'avg_mse': 1852787393.6371064}

'Perform the final cross validation using the optimal hyperparameters to obtain the average mse across all folds. Initialized weight and bias parameters at 0. '

##k_fold_cross_validation_with_combined_scaler(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, num_iters=5000, lambda_=10.0, k=5, tol=1e-6)

#Average MSE across all folds: 1852787393.6371064  


'Retraining(outside cross validation) across all examples using the optimal hyperparameters and again initializing weight and bias parameters at zero.'

##w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled, J_history_combined_scaled = retrain_on_full_data_with_combined_scaling(X_selected_with_composite_2_interaction_and_polyfeature, y_float, w_initial_3, b_initial, RobustScaler, MinMaxScaler, compute_regularized_cost, compute_regularized_gradient, gradient_descent_reg_with_convergence_check, alpha=0.8, lambda_=10.0, num_iters=6000, tol=1e-6)
##print(w_final_combined_scaled, b_final_combined_scaled, mse_final_combined_scaled)

#Final weight parameters: [276150.72510091, 178622.44214051, 137848.02508316, -92720.72470812, 26381.9465485,  -83779.53604989] 
#Final bias parameter: 74149.40367663385 
#Final MSE: 1777064469.1029153 
#Converged at iteration 5225





final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_interaction_and_polyfeature = {'standard scaled model': 1633448073.935783, 'robust scaled model': 1633580705.04759, 'minmax scaled model': 1852787393.6371064, 'combined scaled model': 1852787393.6371064}

min_key_7 = min(final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_interaction_and_polyfeature, key=final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_interaction_and_polyfeature.get)
##print(f"Model with minimum average MSE: {min_key_7}, MSE: {final_average_mse_across_all_folds_of_each_scaled_model_with_composite_2_interaction_and_polyfeature[min_key_7]}")

#Model with minimum average MSE: standard scaled model, MSE: 1633448073.935783   
#Final weight parameters of standard scaled model from training on all exammples: [33440.24048273,  21906.79097603,  22649.34366006, -18272.96889341, 14752.38725995, -19557.9774121]
#Final bias parameter of standard scaled model from training on all examples: 180796.06006825936 
#Final MSE of standard scaled model from training on all examples: 1589273829.2782097








'''

Comparing the 7 optimal models using gradient descent from each "model training with feature scaling routine" implementation by the average mse across all folds from cross validation. 

Specifically the models compared are: 


the standard scaled model including the "X_selected_final" features. 

the standard scaled model including the "X_selected_with_composite" features.

the standard scaled model including the "X_selected_with_composite_2" features. 

the robust scaled model including the "X_selected_with_composite_and_interaction" features.

the robust scaled model including the "X_selected_with_composite_2_and_interaction" features.

the standard scaled model including the "X_selected_with_composite_interaction_and_polyfeature" features.

the standard scaled model including the "X_selected_with_composite_2_interaction_and_polyfeature" features.

'''





final_average_mse_across_all_folds_of_each_optimal_scaled_gradient_descent_model = {'the standard scaled model including the "X_selected_final" features': 1875242251.305547, 'the standard scaled model including the "X_selected_with_composite" features': 1801895027.3376033, 'the standard scaled model including the "X_selected_with_composite_2" features': 1759272140.677011, 'the robust scaled model including the "X_selected_with_composite_and_interaction" features': 1651838884.277936, 'the robust scaled model including the "X_selected_with_composite_2_and_interaction" features': 1648435620.017269, 'the standard scaled model including the "X_selected_with_composite_interaction_and_polyfeature" features': 1628469596.294723, 'the standard scaled model including the "X_selected_with_composite_2_interaction_and_polyfeature" features': 1633448073.935783}

min_key_8 = min(final_average_mse_across_all_folds_of_each_optimal_scaled_gradient_descent_model, key=final_average_mse_across_all_folds_of_each_optimal_scaled_gradient_descent_model.get)
##print(f"Model with minimum average MSE: {min_key_8}, MSE: {final_average_mse_across_all_folds_of_each_optimal_scaled_gradient_descent_model[min_key_8]}")

#Model with minimum average MSE: the standard scaled model including the "X_selected_with_composite_interaction_and_polyfeature" features, MSE: 1628469596.294723  
#Final weight parameters of the standard scaled model including the "X_selected_with_composite_interaction_and_polyfeature" features from training on all exammples: [31407.76439351,  20410.43071129,  25835.96670591, -24335.53562665, 20396.29515505, -19924.40444393]
#Final bias parameter of the standard scaled model including the "X_selected_with_composite_interaction_and_polyfeature" features from training on all examples: 180796.06006825936 
#Final MSE of standard scaled model from training on all examples: 1586878493.8827815



'''
Regression plots for the optimal gradient descent model for each feature against the housing price target(see "./Visualizations/Regression_Plots/Gradient_Descent_Models/Fig_4_Regression_Plots_of_each_Feature_vs_Housing_Price_Target _from_Optimal_Gradient_Descent_Model").

***Note*** Each specific feature regression is plotted where the other feature values are held constant at their means. For the "House Age" feature, the linear("House Age") and quadratic("House Age Squared") regression contributions are combined and plotted together in one subplot. 

All examples of the "X_selected_with_composite_interaction_and_polyfeature" features were scaled using standard scaling before plotting regression plots to align with the scaling used during training.

'''

#Scale all examples with Standard scaler. 
X_selected_with_composite_interaction_and_polyfeature_standard_scaled = standard_scaler.fit_transform(X_selected_with_composite_interaction_and_polyfeature) 


#Optimal weight and bias parameters from optimal gradient descent model trained on all examples. 
optimal_gradient_descent_model_weights = np.array([31407.76439351,  20410.43071129,  25835.96670591, -24335.53562665, 20396.29515505, -19924.40444393])
optimal_gradient_descent_model_bias = 180796.06006825936


#List of feature labels.
optimal_gradient_descent_model_feature_names_list = ['Gross Living Area(sqft)', 'Composite: "Basement + 1st Floor + Garage" Area(sqft)', 'Garage Area(sqft)', 'House Age(years)', 'House Age Squared(years)', 'Interaction: "House Age * Garage Area" (years * sqft)']

#Title of entire figure.
optimal_gradient_descent_model_title = 'Figure 4: Regression Plots of each Standard Scaled Feature vs the Housing Price Target from the Optimal Gradient Descent Model'


#Function to plot isolated feature regressions with subplots oriented in rows and columns. 
##plot_isolated_feature_regressions_from_gradient_descent_model_with_polys_and_individuals_with_rowsandcols(X_selected_with_composite_interaction_and_polyfeature_standard_scaled, y_float, optimal_gradient_descent_model_weights, optimal_gradient_descent_model_bias, optimal_gradient_descent_model_feature_names_list, optimal_gradient_descent_model_title, original_indices=[0,1,2,3,5], poly_index=3, n_cols=3)




'''
Examining feature importance in predicting the housing price target by comparing the magnitudes of weight coefficients across all features from the optimal "X_selected_with_composite_interaction_and_polyfeature" gradient descent model(see ./Visualizations/Bar_Plots/Fig_9_Bar_Plot_of_Feature_Importance_from_Optimal_Standard_Scaled_Gradient_Descent_Model). 

***NOTE*** The weight coefficients compared below are from the model trained on all examples.

'''

#Function to plot feature importances based on the magnitudes of weight coefficients using a bar plot.
##plot_feature_importance_for_gradient_descent(X_selected_with_composite_interaction_and_polyfeature, feature_names=optimal_gradient_descent_model_feature_names_list, weight_coefficients=optimal_gradient_descent_model_weights)




'''

Features top to bottom of highest importance in predicting the housing price target to lowest importance(listed non-absolute weights below):


Gross Living Area                                  Weight Coefficient = 31407.76439351

Garage Area                                        Weight Coefficient = 25835.96670591

House Age                                          Weight Coefficient = -24335.53562665

Composite: "Basement + 1st Floor + Garage" Area    Weight Coefficient = 20410.43071129

House Age Squared                                  Weight Coefficient = 20396.29515505

Interaction: "House Age * Garage Area"             Weight Coefficient = -19924.40444393


'''
















'''

More robust model training methods: 



Decision Trees:

Trained different groups of features using decision trees. Implemented to decision tree function with grid search within cross validation to obtain optimal hyperparameters for decision tree training.

***NOTE*** Trained decision tree models within cross validation using consistent cross validation configurations as used for the gradient descent models i.e., k=5, shuffle=True, and random_state=42. This decision tree function also trained on all examples outside of cross validation.

***NOTE*** Because decision trees split the data based on feature thresholds and are only concerned with the relative ordering of feature values and not their absolute magnitudes, decision trees are not sensitive to feature scaling. The tree's splitting process will work the same regardless of feature scaling and modeling scaled features will generally neither increase or decrease model performance. As a result, features were not scaled in this analysis to improve interpretability for model visualizations. 

***NOTE*** Also, each split in the tree introduces a non-linear boundary that divides the data which inherently captures non-linear interactions. Including polynomial terms introduces the risk of overfitting due to the increased, potentially unnecessary, dimensionality by the included polynomial terms. Also, decision trees already model interactions between features implicitly, so including polynomial terms can lead to redundant splits. However, because these features appear to result in underfitted models due to their large error terms(mse), the feature arrays which included polynomial terms were used in this decision tree analysis to examine any potential increase in model performance. 

Also, the interaction term("House Age" * "Garage Area") was the least important feature in predicting the housing price target in the models trained with gradient descent. Therefore, included models where the interaction term was removed as well as the individual "Garage Area" feature to reduce weak predictors and feature redundancy since "Garage Area" is included in the composite and composite_2 features.

As a result, the feature arrays used in the decision tree models included: 

"X_selected_final"
"X_selected_with_composite"
"X_selected_with_composite_2"
"X_selected_with_composite_and_polyfeature"
"X_selected_with_composite_2_and_polyfeature"
"X_selected_with_composite_and_interaction"
"X_selected_with_composite_2_and_interaction"
"X_selected_with_composite_interaction_and_polyfeature"
"X_selected_with_composite_2_interaction_and_polyfeature"



Random Forests:

Trained different groups of features using random forests.

***NOTE*** Trained random forest models within cross validation using consistent cross validation configurations as used for the gradient descent models i.e., k=5, shuffle=True, and random_state=42. This random forest function also trained on all examples outside of cross validation.

Implemented the same feature metrics as noted above for the decision tree models. However, due to the high computation cost for grid searching within cross validation for random forest training, the optimal hyperparameters found for the decision tree models will be used for the random forest models.

As a result, the feature arrays used in the random forest models included: 

"X_selected_final"
"X_selected_with_composite"
"X_selected_with_composite_2"
"X_selected_with_composite_and_polyfeature"
"X_selected_with_composite_2_and_polyfeature"
"X_selected_with_composite_and_interaction"
"X_selected_with_composite_2_and_interaction"
"X_selected_with_composite_interaction_and_polyfeature"
"X_selected_with_composite_2_interaction_and_polyfeature"

'''




'Decision Tree Analysis'




'Trained a model using a decision tree with the "X_selected_final" feature array. As a result, the specific features included in this model were "Gr Liv Area", "Total Bsmt Area", and "House Age".'


#Function to grid search through hyperparameters for decision tree regressor within K-fold cross validation to find optimal hyperparameters which result in the lowest average mse across all folds.
##decision_tree_with_grid_search_within_cross_validation(X_selected_final, y_float, k=5)

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': None, 'max_leaf_nodes': 50, 'average mse': 1236664223.1556942}


#Function to train a model using a decision tree within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_decision_tree_model_outside_CV_from_all_examples_using_X_selected_final = train_decision_tree_regression_within_CV_and_outside_on_all_examples(X_selected_final, y_float, k=5, max_depth=None, min_samples_split=10, min_samples_leaf=4, max_features=None, max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1236664223.1556942
#Final MSE from training on all examples: 385970103.570371




'Trained a model using a decision tree with the "X_selected_with_composite" feature array. As a result, the specific features included in this model were "Gr Liv Area", composite("Total Bsmt Area"+"1st Flr SF"+"Garage Area"), and "House Age".'


#Function to grid search through hyperparameters for decision tree regressor within K-fold cross validation to find optimal hyperparameters which result in the lowest average mse across all folds.
##decision_tree_with_grid_search_within_cross_validation(X_selected_with_composite, y_float, k=5)

#Optimal Hyperparameters: {'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': None, 'max_leaf_nodes': 50, 'average mse': 1243765695.0447621}


#Function to train a model using a decision tree within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_decision_tree_model_outside_CV_from_all_examples_using_X_selected_with_composite = train_decision_tree_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite, y_float, k=5, max_depth=10, min_samples_split=10, min_samples_leaf=4, max_features=None, max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1243765695.0447621
#Final MSE from training on all examples: 433916471.4002403




'Trained a model using a decision tree with the "X_selected_with_composite_2" feature array. As a result, the specific features included in this model were "Gr Liv Area", composite_2("Total Bsmt Area"+"Garage Area"), and "House Age".'


#Function to grid search through hyperparameters for decision tree regressor within K-fold cross validation to find optimal hyperparameters which result in the lowest average mse across all folds.
##decision_tree_with_grid_search_within_cross_validation(X_selected_with_composite_2, y_float, k=5)

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 2, 'max_features': None, 'max_leaf_nodes': 50, 'average mse': 1306357187.3014874}



#Function to train a model using a decision tree within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_decision_tree_model_outside_CV_from_all_examples_using_X_selected_with_composite_2 = train_decision_tree_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_2, y_float, k=5, max_depth=None, min_samples_split=20, min_samples_leaf=2, max_features=None, max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1306357187.3014874
#Final MSE from training on all examples: 510929424.02765405



'Trained a model using a decision tree with the "X_selected_with_composite_and_polyfeature" feature array. As a result, the specific features included in this model were "Gr Liv Area", composite("Total Bsmt Area"+"1st Flr SF"+"Garage Area"), "House Age", and "House Age Squared".'

#Stacked all these features into new 2-D feature array with shape (2930,4)
X_selected_with_composite_and_polyfeature = np.stack((arr3, composite_feat_arr, arr9, arr9_squared),axis=1)



#Function to grid search through hyperparameters for decision tree regressor within K-fold cross validation to find optimal hyperparameters which result in the lowest average mse across all folds.
##decision_tree_with_grid_search_within_cross_validation(X_selected_with_composite_and_polyfeature, y_float, k=5)

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_leaf_nodes': 50, 'average mse': 1163459883.8630824}



#Function to train a model using a decision tree within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_decision_tree_model_outside_CV_from_all_examples_using_X_selected_with_composite_and_polyfeature = train_decision_tree_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_and_polyfeature, y_float, k=5, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features='sqrt', max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1163459883.8630824
#Final MSE from training on all examples: 3394859.60458313


'Trained a model using a decision tree with the "X_selected_with_composite_2_and_polyfeature" feature array. As a result, the specific features included in this model were "Gr Liv Area", composite("Total Bsmt Area" + "Garage Area"), "House Age", and "House Age Squared".'

#Stacked all these features into new 2-D feature array with shape (2930,4)
X_selected_with_composite_2_and_polyfeature = np.stack((arr3, composite_2_feat_arr, arr9, arr9_squared),axis=1)



#Function to grid search through hyperparameters for decision tree regressor within K-fold cross validation to find optimal hyperparameters which result in the lowest average mse across all folds.
##decision_tree_with_grid_search_within_cross_validation(X_selected_with_composite_2_and_polyfeature, y_float, k=5)

#Optimal Hyperparameters: {'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'average mse': 1283834773.249491}



#Function to train a model using a decision tree within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_decision_tree_model_outside_CV_from_all_examples_using_X_selected_with_composite_2_and_polyfeature = train_decision_tree_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_2_and_polyfeature, y_float, k=5, max_depth=10, min_samples_split=20, min_samples_leaf=2, max_features='sqrt', max_leaf_nodes=None, random_state=42)

#Average MSE across 5 folds: 1283834773.249491
#Final MSE from training on all examples: 574422320.1939601




'Trained a model using a decision tree with the "X_selected_with_composite_and_interaction" feature array. As a result, the specific features included in this model were "Gr Liv Area", "composite("Total Bsmt Area"+"1st Flr SF"+"Garage Area")", "Garage Area", "House Age", and "interaction(House Age * Garage Area)".'

#Function to grid search through hyperparameters for decision tree regressor within K-fold cross validation to find optimal hyperparameters which result in the lowest average mse across all folds.
##decision_tree_with_grid_search_within_cross_validation(X_selected_with_composite_and_interaction, y_float, k=5)

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': None, 'max_leaf_nodes': 50, 'average mse': 1267941769.014067}



#Function to train a model using a decision tree within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_decision_tree_model_outside_CV_from_all_examples_using_X_selected_with_composite_and_interaction = train_decision_tree_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_and_interaction, y_float, k=5, max_depth=None, min_samples_split=10, min_samples_leaf=1, max_features=None, max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1267941769.014067
#Final MSE from training on all examples: 248695793.1378382



'Trained a model using a decision tree with the "X_selected_with_composite_2_and_interaction" feature array. As a result, the specific features included in this model were "Gr Liv Area", "composite_2("Total Bsmt Area" + "Garage Area")", "Garage Area", "House Age", and "interaction(House Age * Garage Area)".'


#Function to grid search through hyperparameters for decision tree regressor within K-fold cross validation to find optimal hyperparameters which result in the lowest average mse across all folds.
##decision_tree_with_grid_search_within_cross_validation(X_selected_with_composite_2_and_interaction, y_float, k=5)

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 1, 'max_features': None, 'max_leaf_nodes': 100, 'average mse': 1316359199.985784}



#Function to train a model using a decision tree within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_decision_tree_model_outside_CV_from_all_examples_using_X_selected_with_composite_2_and_interaction = train_decision_tree_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_2_and_interaction, y_float, k=5, max_depth=None, min_samples_split=20, min_samples_leaf=1, max_features=None, max_leaf_nodes=100, random_state=42)


#Average MSE across 5 folds: 1316359199.985784
#Final MSE from training on all examples: 484539331.10634416



'Trained a model using a decision tree with the "X_selected_with_composite_interaction_and_polyfeature" feature array. As a result, the specific features included in this model were "Gr Liv Area", "composite("Total Bsmt Area"+"1st Flr SF"+"Garage Area")", "Garage Area", "House Age", "House Age Squared", and "interaction(House Age * Garage Area)".'

#Function to grid search through hyperparameters for decision tree regressor within K-fold cross validation to find optimal hyperparameters which result in the lowest average mse across all folds.
##decision_tree_with_grid_search_within_cross_validation(X_selected_with_composite_interaction_and_polyfeature, y_float, k=5)

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': None, 'max_leaf_nodes': 50, 'average mse': 1265740653.8291984}



#Function to train a model using a decision tree within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_decision_tree_model_outside_CV_from_all_examples_using_X_selected_with_composite_interaction_and_polyfeature = train_decision_tree_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_interaction_and_polyfeature, y_float, k=5, max_depth=None, min_samples_split=10, min_samples_leaf=1, max_features=None, max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1265740653.8291984
#Final MSE from training on all examples: 248695793.1378382






'Trained a model using a decision tree with the "X_selected_with_composite_2_interaction_and_polyfeature" feature array. As a result, the specific features included in this model were "Gr Liv Area", "composite_2("Total Bsmt Area" + "Garage Area")", "Garage Area", "House Age", "House Age Squared", and "interaction(House Age * Garage Area)".'


#Function to grid search through hyperparameters for decision tree regressor within K-fold cross validation to find optimal hyperparameters which result in the lowest average mse across all folds.
##decision_tree_with_grid_search_within_cross_validation(X_selected_with_composite_2_interaction_and_polyfeature, y_float, k=5)

#Optimal Hyperparameters: {'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 1, 'max_features': 0.5, 'max_leaf_nodes': None, 'average mse': 1287310544.5515544}



#Function to train a model using a decision tree within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_decision_tree_model_outside_CV_from_all_examples_using_X_selected_with_composite_2_interaction_and_polyfeature = train_decision_tree_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_2_interaction_and_polyfeature, y_float, k=5, max_depth=10, min_samples_split=20, min_samples_leaf=1, max_features=0.5, max_leaf_nodes=None, random_state=42)


#Average MSE across 5 folds: 1287310544.5515544
#Final MSE from training on all examples: 557493619.253105




'''

Comparing the 9 models using decision tree modeling by the average mse across all folds from cross validation. 

Specifically the models compared are: 


the unscaled decision tree model including the "X_selected_final" features. 

the unscaled decision tree model including the "X_selected_with_composite" features.

the unscaled decision tree model including the "X_selected_with_composite_2" features. 

the unscaled decision tree model including the "X_selected_with_composite_and_polyfeature" features.

the unscaled decision tree model including the "X_selected_with_composite_2_and_polyfeature" features.

the unscaled decision tree model including the "X_selected_with_composite_and_interaction" features.

the unscaled decision tree model including the "X_selected_with_composite_2_and_interaction" features.

the unscaled decision tree model including the "X_selected_with_composite_interaction_and_polyfeatures" features.

the unscaled decision tree model including the "X_selected_with_composite_2_interaction_and_polyfeature" features.

'''


final_average_mse_across_all_folds_of_each_unscaled_decision_tree_model = {'the unscaled decision tree model including the "X_selected_final" features': 1236664223.1556942, 'the unscaled decision tree model including the "X_selected_with_composite" features': 1243765695.0447621, 'the unscaled decision tree model including the "X_selected_with_composite_2" features': 1306357187.3014874, 'the unscaled decision tree model including the "X_selected_with_composite_and_polyfeature" features': 1163459883.8630824, 'the unscaled decision tree model including the "X_selected_with_composite_2_and_polyfeature" features': 1283834773.249491, 'the unscaled decision tree model including the "X_selected_with_composite_and_interaction" features': 1267941769.014067, 'the unscaled decision tree model including the "X_selected_with_composite_2_and_interaction" features': 1316359199.985784, 'the unscaled decision tree model including the "X_selected_with_composite_interaction_and_polyfeatures" features': 1265740653.8291984, 'the unscaled decision tree model including the "X_selected_with_composite_2_interaction_and_polyfeature" features.': 1287310544.5515544}

min_key_9 = min(final_average_mse_across_all_folds_of_each_unscaled_decision_tree_model, key=final_average_mse_across_all_folds_of_each_unscaled_decision_tree_model.get)
##print(f"Model with minimum average MSE: {min_key_9}, MSE: {final_average_mse_across_all_folds_of_each_unscaled_decision_tree_model[min_key_9]}")

#Model with minimum average MSE: the unscaled decision tree model including the "X_selected_with_composite_and_polyfeature" features, MSE: 1163459883.8630824
#Final MSE from training on all examples: 3394859.60458313




'''
Plotting the regression plots of the optimal decision tree model(see "./Visualizations/Regression_Plots/Decision_Tree_Models/Fig_5_Regression_Plots_of_each_Unscaled_Feature_vs_Housing_Price_Target_from_Optimal_Decision_Tree_Model").

***Note*** Each specific feature regression is plotted where the other feature values are held constant at their means. For the "House Age" feature, the "House Age" and "House Age Squared" features are combined and plotted together in one subplot. However, because decision trees model non-linear relationships implictly, this combined visualization is simply visualizing the original and squared "House Age" terms together and not visualizing the enforcement of the combined linear and quadratic behaviors like in the optimal gradient descent model. These features were not scaled before plotting.
'''

#List of feature labels.
optimal_decision_tree_model_feature_list = ['Gross Living Area(sqft)', 'Composite: "Basement + 1st Floor + Garage" Area(sqft)', 'House Age(years)', 'House Age Squared(years)']

#Figure title.
optimal_decision_tree_model_title = 'Figure 5: Regression Plots of each Unscaled Feature vs the Housing Price Target from the Optimal Decision Tree Model'

#Function to plot the regressions from tree-based models.
##plot_isolated_feature_regressions_from_tree_based_model_with_polyfeature_combined(X_selected_with_composite_and_polyfeature, y_float, model=trained_decision_tree_model_outside_CV_from_all_examples_using_X_selected_with_composite_and_polyfeature, feature_labels=optimal_decision_tree_model_feature_list, title=optimal_decision_tree_model_title, x3_index=2, poly_index=3)








'Random Forest Analysis'


'Trained a model using a random forest with the "X_selected_final" feature array. As a result, the specific features included in this model were "Gr Liv Area", "Total Bsmt Area", and "House Age".'

#Optimal hyperparameters regarding decision trees to be used in this corresponding random forest model.  

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': None, 'max_leaf_nodes': 50, 'average mse': 1236664223.1556942}


#Function to train a model using a random forest within K-fold cross validation and outside CV on all examples. Used a fixed "n_estimators" of 100. 
##_,_,_,trained_random_forest_model_outside_CV_from_all_examples_using_X_selected_final = train_random_forest_regression_within_CV_and_outside_on_all_examples(X_selected_final, y_float, k=5, n_estimators=100, max_depth=None, min_samples_split=10, min_samples_leaf=4, max_features=None, max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1145088027.3604884
#Final MSE from training on all examples: 822815840.1343879


'Trained a model using a random forest with the "X_selected_with_composite" feature array. As a result, the specific features included in this model were "Gr Liv Area", composite("Total Bsmt Area"+"1st Flr SF"+"Garage Area"), and "House Age".'

#Optimal hyperparameters regarding decision trees to be used in this corresponding random forest model.  

#Optimal Hyperparameters: {'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': None, 'max_leaf_nodes': 50, 'average mse': 1243765695.0447621}


#Function to train a model using a random forest within K-fold cross validation and outside CV on all examples. Used a fixed "n_estimators" of 100.
##_,_,_,trained_random_forest_model_outside_CV_from_all_examples_using_X_selected_with_composite = train_random_forest_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite, y_float, k=5, n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=4, max_features=None, max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1104263630.9283166
#Final MSE from training on all examples: 780337833.2150604


'Trained a model using a random forest with the "X_selected_with_composite_2" feature array. As a result, the specific features included in this model were "Gr Liv Area", composite_2("Total Bsmt Area"+"Garage Area"), and "House Age".'

#Optimal hyperparameters regarding decision trees to be used in this corresponding random forest model.  

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 2, 'max_features': None, 'max_leaf_nodes': 50, 'average mse': 1306357187.3014874}



#Function to train a model using a random forest within K-fold cross validation and outside CV on all examples. Used a fixed "n_estimators" of 100.
##_,_,_,trained_random_forest_model_outside_CV_from_all_examples_using_X_selected_with_composite_2 = train_random_forest_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_2, y_float, k=5, n_estimators=100, max_depth=None, min_samples_split=20, min_samples_leaf=2, max_features=None, max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1102766445.3350215
#Final MSE from training on all examples: 748164792.7730349


'Trained a model using a random forest with the "X_selected_with_composite_and_polyfeature" feature array. As a result, the specific features included in this model were "Gr Liv Area", "composite("Total Bsmt Area"+"1st Flr SF"+"Garage Area")", "House Age", and "House Age Squared".'

#Optimal hyperparameters regarding decision trees to be used in this corresponding random forest model.  

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_leaf_nodes': 50, 'average mse': 1163459883.8630824}



#Function to train a model using a random forest within K-fold cross validation and outside CV on all examples. Used a fixed "n_estimators" of 100.
##_,_,_,trained_random_forest_model_outside_CV_from_all_examples_using_X_selected_with_composite_and_polyfeature = train_random_forest_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_and_polyfeature, y_float, k=5, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features='sqrt', max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1082338691.8576145
#Final MSE from training on all examples: 732012776.6399677


'Trained a model using a random forest with the "X_selected_with_composite_2_and_polyfeature" feature array. As a result, the specific features included in this model were "Gr Liv Area", composite_2("Total Bsmt Area"+"Garage Area"), "House Age", and "House Age Squared".'

#Optimal hyperparameters regarding decision trees to be used in this corresponding random forest model.  

#Optimal Hyperparameters: {'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'average mse': 1283834773.249491}



#Function to train a model using a random forest within K-fold cross validation and outside CV on all examples. Used a fixed "n_estimators" of 100.
##_,_,_,trained_random_forest_model_outside_CV_from_all_examples_using_X_selected_with_composite_2_and_polyfeature = train_random_forest_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_2_and_polyfeature, y_float, k=5, n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=2, max_features='sqrt', max_leaf_nodes=None, random_state=42)

#Average MSE across 5 folds: 1081342010.037117
#Final MSE from training on all examples: 664837284.6968312


'Trained a model using a random forest with the "X_selected_with_composite_and_interaction" feature array. As a result, the specific features included in this model were "Gr Liv Area", "composite("Total Bsmt Area"+"1st Flr SF"+"Garage Area")", "Garage Area", "House Age", and "interaction(House Age * Garage Area)".'

#Optimal hyperparameters regarding decision trees to be used in this corresponding random forest model.

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': None, 'max_leaf_nodes': 50, 'average mse': 1267941769.014067}



#Function to train a model using a random forest within K-fold cross validation and outside CV on all examples. Used a fixed "n_estimators" of 100.
##_,_,_,trained_random_forest_model_outside_CV_from_all_examples_using_X_selected_with_composite_and_interaction = train_random_forest_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_and_interaction, y_float, k=5, n_estimators=100, max_depth=None, min_samples_split=10, min_samples_leaf=1, max_features=None, max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1055223381.1078861
#Final MSE from training on all examples: 674316870.2560065


'Trained a model using a random forest with the "X_selected_with_composite_2_and_interaction" feature array. As a result, the specific features included in this model were "Gr Liv Area", "composite_2("Total Bsmt Area" + "Garage Area")", "Garage Area", "House Age", and "interaction(House Age * Garage Area)".'


#Optimal hyperparameters regarding decision trees to be used in this corresponding random forest model.

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 1, 'max_features': None, 'max_leaf_nodes': 100, 'average mse': 1316359199.985784}



#Function to train a model using a random forest within K-fold cross validation and outside CV on all examples. Used a fixed "n_estimators" of 100.
##_,_,_,trained_random_forest_model_outside_CV_from_all_examples_using_X_selected_with_composite_2_and_interaction = train_random_forest_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_2_and_interaction, y_float, k=5, n_estimators=100, max_depth=None, min_samples_split=20, min_samples_leaf=1, max_features=None, max_leaf_nodes=100, random_state=42)

#Average MSE across 5 folds: 1061955955.9055735
#Final MSE from training on all examples: 588996443.1694311


'Trained a model using a random forest with the "X_selected_with_composite_interaction_and_polyfeature" feature array. As a result, the specific features included in this model were "Gr Liv Area", "composite("Total Bsmt Area"+"1st Flr SF"+"Garage Area")", "Garage Area", "House Age", "House Age Squared", and "interaction(House Age * Garage Area)".'

#Optimal hyperparameters regarding decision trees to be used in this corresponding random forest model.

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': None, 'max_leaf_nodes': 50, 'average mse': 1265740653.8291984}



#Function to train a model using a random forest within K-fold cross validation and outside CV on all examples. Used a fixed "n_estimators" of 100.
##_,_,_,trained_random_forest_model_outside_CV_from_all_examples_using_X_selected_with_composite_interaction_and_polyfeature = train_random_forest_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_interaction_and_polyfeature, y_float, k=5, n_estimators=100, max_depth=None, min_samples_split=10, min_samples_leaf=1, max_features=None, max_leaf_nodes=50, random_state=42)

#Average MSE across 5 folds: 1055534641.5612459
#Final MSE from training on all examples: 675537249.1818153


'Trained a model using a random forest with the "X_selected_with_composite_2_interaction_and_polyfeature" feature array. As a result, the specific features included in this model were "Gr Liv Area", "composite_2("Total Bsmt Area" + "Garage Area")", "Garage Area", "House Age", "House Age Squared", and "interaction(House Age * Garage Area)".'


#Optimal hyperparameters regarding decision trees to be used in this corresponding random forest model.

#Optimal Hyperparameters: {'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 1, 'max_features': 0.5, 'max_leaf_nodes': None, 'average mse': 1287310544.5515544}



#Function to train a model using a random forest within K-fold cross validation and outside CV on all examples. Used a fixed "n_estimators" of 100.
##_,_,_,trained_random_forest_model_outside_CV_from_all_examples_using_X_selected_with_composite_2_interaction_and_polyfeature = train_random_forest_regression_within_CV_and_outside_on_all_examples(X_selected_with_composite_2_interaction_and_polyfeature, y_float, k=5, n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=1, max_features=0.5, max_leaf_nodes=None, random_state=42)

#Average MSE across 5 folds: 1054769392.7928307
#Final MSE from training on all examples: 602778922.7282625



'''

Comparing the 9 models using random forest modeling by the average mse across all folds from cross validation. 

Specifically the models compared are: 


the unscaled random_forest model including the "X_selected_final" features. 

the unscaled random_forestmodel including the "X_selected_with_composite" features.

the unscaled random_forest model including the "X_selected_with_composite_2" features. 

the unscaled random_forest model including the "X_selected_with_composite_and_polyfeature" features.

the unscaled random_forest model including the "X_selected_with_composite_2_and_polyfeature" features.

the unscaled random_forest model including the "X_selected_with_composite_and_interaction" features.

the unscaled random_forest model including the "X_selected_with_composite_2_and_interaction" features.

the unscaled random_forestmodel including the "X_selected_with_composite_interaction_and_polyfeatures" features.

the unscaled random_forest model including the "X_selected_with_composite_2_interaction_and_polyfeature" features.


'''


final_average_mse_across_all_folds_of_each_unscaled_random_forest_model = {'the unscaled random forest model including the "X_selected_final" features': 1145088027.3604884, 'the unscaled random forest model including the "X_selected_with_composite" features': 1104263630.9283166, 'the unscaled random forest model including the "X_selected_with_composite_2" features': 1102766445.3350215, 'the unscaled random forest model including the "X_selected_with_composite_and_polyfeature" features': 1082338691.8576145, 'the unscaled random forest model including the "X_selected_with_composite_2_and_polyfeature" features': 1081342010.037117, 'the unscaled random forest model including the "X_selected_with_composite_and_interaction" features': 1055223381.1078861, 'the unscaled random forest model including the "X_selected_with_composite_2_and_interaction" features': 1061955955.9055735, 'the unscaled random forest model including the "X_selected_with_composite_interaction_and_polyfeatures" features': 1055534641.5612459, 'the unscaled random forest model including the "X_selected_with_composite_2_interaction_and_polyfeature" features.': 1054769392.7928307}

min_key_10 = min(final_average_mse_across_all_folds_of_each_unscaled_random_forest_model, key=final_average_mse_across_all_folds_of_each_unscaled_random_forest_model.get)
##print(f"Model with minimum average MSE: {min_key_10}, MSE: {final_average_mse_across_all_folds_of_each_unscaled_random_forest_model[min_key_10]}")

#Model with minimum average MSE: the unscaled random forest model including the "X_selected_with_composite_2_interaction_and_polyfeature" features., MSE: 1054769392.7928307
#Final MSE from training on all examples: 602778922.7282625


'''
Plotting the regression plots of the optimal random forest model(see "./Visualizations/Regression_Plots/Random_Forest_Models/Fig_6_Regression_Plots_of_each_Unscaled_Feature_vs_Housing_Price_Target_from_Optimal_Random_Forest_Model").

***Note*** Each specific feature regression is plotted where the other feature values are held constant at their means. For the "House Age" feature, the "House Age" and "House Age Squared" features are combined and plotted together in one subplot. However, because random forests model non-linear relationships implictly, this combined visualization is simply visualizing the original and squared "House Age" terms together and not visualizing the enforcement of the combined linear and quadratic behaviors like in the optimal gradient descent model. These features were not scaled before plotting.
'''

#List of feature labels.
optimal_random_forest_model_feature_list = ['Gross Living Area(sqft)', 'Composite_2: "Basement + Garage" Area(sqft)', 'Garage Area(sqft)', 'House Age(years)', 'House Age Squared(years)', 'Interaction: "House Age * Garage Area" (years * sqft)']

#Figure title.
optimal_random_forest_model_title = 'Figure 6: Regression Plots of each Unscaled Feature vs the Housing Price Target from the Optimal Random Forest Model'

#Function to plot the regressions from tree-based models.
##plot_isolated_feature_regressions_from_tree_based_model_with_polyfeature_combined_with_rowsandcols(X_selected_with_composite_2_interaction_and_polyfeature, y_float, model=trained_random_forest_model_outside_CV_from_all_examples_using_X_selected_with_composite_2_interaction_and_polyfeature, feature_labels=optimal_random_forest_model_feature_list, title=optimal_random_forest_model_title, original_indices=[0,1,2,3,5], poly_index=3, n_cols=3)












'''

Comparision of all optimal models from each model training method:

Comparing the optimal models from each model training method by the average mse across all folds from cross validation. 



Specifically the models compared are: 


the standard scaled gradient descent model including the "X_selected_with_composite_interaction_and_polyfeature" features.

the unscaled decision tree model including the "X_selected_with_composite_and_polyfeature" features.

the unscaled random forest model including the "X_selected_with_composite_2_interaction_and_polyfeature" features.

'''


final_average_mse_across_all_folds_of_each_optimal_model_from_each_model_training_method = {'the standard scaled gradient descent model including the "X_selected_with_composite_interaction_and_polyfeature" features': 1628469596.294723, 'the unscaled decision tree model including the "X_selected_with_composite_and_polyfeature" features': 1163459883.8630824, 'the unscaled random forest model including the "X_selected_with_composite_2_interaction_and_polyfeature" features': 1054769392.7928307}

min_key_11 = min(final_average_mse_across_all_folds_of_each_optimal_model_from_each_model_training_method, key=final_average_mse_across_all_folds_of_each_optimal_model_from_each_model_training_method.get)
##print(f"Model with minimum average MSE: {min_key_11}, MSE: {final_average_mse_across_all_folds_of_each_optimal_model_from_each_model_training_method[min_key_11]}")

#Model with minimum average MSE: the unscaled random forest model including the "X_selected_with_composite_2_interaction_and_polyfeature" features, MSE: 1054769392.7928307
#Final MSE from training on all examples: 602778922.7282625
#See "./Visualizations/Regression_Plots/Random_Forest_Models/Fig_6_Regression_Plots_of_each_Unscaled_Feature_vs_Housing_Price_Target_from_Optimal_Random_Forest_Model" for visualization of this model.




'''

Further analysis with increased dimensionality:

The models throughout each training method appear to be underfitting and exhibt underwhelming predictive performance based of their large average MSE values(see Figues 4, 5, and 6 in "Regression_Plots" folder). This may indicate that the models lack enough complexity to accurately predict target values. It is worth examining different feature configurations and how other, higher dimensional, models may impact model performance. 

The first new feature array to be implemented will be a feature array including the features initially selected from the Ames Housing dataset which had suitable distributions for regression modeling. These features will be included as individual features without any scaling, polynomial terms or composite/interaction features. These features include: 

"Lot Area"
"Lot Frontage"
"Gr Liv Area"
"BsmtFin SF 1"
"Bsmt Unf SF"
"Total Bsmt SF"
"1st Flr SF"
"Garage Area"
"House Age"


The second new feature array to be implemented will be a feature array including the features initially selected from the Ames Housing dataset like above but with the interaction(House Age * Garage Area) and the polynomial term for "House Age". These unscaled features include: 

"Lot Area"
"Lot Frontage"
"Gr Liv Area"
"BsmtFin SF 1"
"Bsmt Unf SF"
"Total Bsmt SF"
"1st Flr SF"
"Garage Area"
"House Age"
"House Age Squared"
"Interaction: House Age * Garage Area"



The more robust decision tree and random forest training methods will be used in this analysis. 

'''




'''
Training a model using a decision tree and random forest which includes features initially selected from the Ames Housing dataset which had suitable distributions for regression modeling. .
'''

#Constructing new 2-D feature array with shape (2930, 9).
X_all_initial_features = np.stack((arr1, imported_inter_Lot_Frontage_arr2, arr3, imported_inter_BsmtFin_SF_1_arr4, imported_inter_Bsmt_Unf_SF_arr5, imported_inter_Total_Bsmt_SF_arr6, arr7, imported_inter_Garage_Area_arr8, arr9),axis=1)


'Decision Tree:'

'Trained a model using a decision tree with the "X_all_initial_features" feature array. As a result, the specific features included in this model were "Lot Area", "Lot Frontage", "Gr Liv Area", "BsmtFin SF 1", "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", "Garage Area", and "House Age".'

#Function to grid search through hyperparameters for decision tree regressor within K-fold cross validation to find optimal hyperparameters which result in the lowest average mse across all folds.
##decision_tree_with_grid_search_within_cross_validation(X_all_initial_features, y_float, k=5)

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 4, 'max_features': None, 'max_leaf_nodes': 80, 'average mse': 1232858930.2681909}



#Function to train a model using a decision tree within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_decision_tree_model_outside_CV_from_all_examples_using_X_all_initial_features = train_decision_tree_regression_within_CV_and_outside_on_all_examples(X_all_initial_features, y_float, k=5, max_depth=None, min_samples_split=20, min_samples_leaf=4, max_features=None, max_leaf_nodes=80, random_state=42)


#Average MSE across 5 folds: 1232858930.2681909
#Final MSE from training on all examples: 413191594.70057696


'Random Forest:'

'Trained a model using a random forest with the "X_all_initial_features" feature array. As a result, the specific features included in this model were "Lot Area", "Lot Frontage", "Gr Liv Area", "BsmtFin SF 1", "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", "Garage Area", and "House Age".'

#Optimal hyperparameters regarding decision trees to be used in this corresponding random forest model.

#Optimal Hyperparameters: {'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 4, 'max_features': None, 'max_leaf_nodes': 80, 'average mse': 1232858930.2681909}



#Function to train a model using a random forest within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_random_forest_model_outside_CV_from_all_examples_using_X_all_initial_features = train_random_forest_regression_within_CV_and_outside_on_all_examples(X_all_initial_features, y_float, k=5, n_estimators=100, max_depth=None, min_samples_split=20, min_samples_leaf=4, max_features=None, max_leaf_nodes=80, random_state=42)

#Average MSE across 5 folds: 981238858.2061069
#Final MSE from training on all examples: 590968372.5834516

'Comparing both models by the average mse across all folds from cross validation.'

final_average_mse_across_all_folds_of_each_tree_based_model_using_X_all_initial_features = {'the unscaled decision tree model including the "X_all_initial_features" features': 1232858930.2681909, 'the unscaled random forest model including the "X_all_initial_features" features': 981238858.2061069}

min_key_12 = min(final_average_mse_across_all_folds_of_each_tree_based_model_using_X_all_initial_features, key=final_average_mse_across_all_folds_of_each_tree_based_model_using_X_all_initial_features.get)
##print(f"Model with minimum average MSE: {min_key_12}, MSE: {final_average_mse_across_all_folds_of_each_tree_based_model_using_X_all_initial_features[min_key_12]}")

#Model with minimum average MSE: the unscaled random forest model including the "X_all_initial_features" features, MSE: 981238858.2061069
#Final MSE from training on all examples: 590968372.5834516



'''
Plotting the regression plots of the optimal random forest model including "X_all_initial_features"(see "./Visualizations/Regression_Plots/Random_Forest_Models/Fig_7_Regression_Plots_of_each_Initially_Selected_Unscaled_Feature_vs_Housing_Price_Target_from_Optimal_Random_Forest_Model").

***Note*** Each specific feature regression is plotted where the other feature values are held constant at their means. These features were not scaled before plotting.
'''

#List of feature labels.
optimal_random_forest_model_feature_list_2 = ['Lot Area(sqft)', 'Lot Frontage(sqft)', 'Gross Living Area(sqft)', 'Basement Finished SF 1(sqft)', 'Basement Unfinished SF(sqft)', 'Total Basement SF(sqft)', '1st Floor SF(sqft)',  'Garage Area(sqft)', 'House Age(years)']

#Figure title.
optimal_random_forest_model_title_2 = 'Figure 7: Regression Plots of each Initially Selected Unscaled Feature vs the Housing Price Target from the Optimal Random Forest Model'

#Function to plot the regressions from tree-based models.
##plot_isolated_feature_regressions_from_tree_based_model_with_polyfeature_combined_with_rowsandcols(X_all_initial_features, y_float, model=trained_random_forest_model_outside_CV_from_all_examples_using_X_all_initial_features, feature_labels=optimal_random_forest_model_feature_list_2, title=optimal_random_forest_model_title_2, original_indices=[0,1,2,3,4,5,6,7,8], poly_index=None, n_cols=3)







'''
Training a model using a decision tree and random forest which includes features initially selected from the Ames Housing dataset but with the interaction(House Age * Garage Area) and the polynomial term for "House Age".
'''

#Constructing new 2-D feature array with shape (2930, 11).
X_all_initial_features_with_interaction_and_polyfeature = np.stack((arr1, imported_inter_Lot_Frontage_arr2, arr3, imported_inter_BsmtFin_SF_1_arr4, imported_inter_Bsmt_Unf_SF_arr5, imported_inter_Total_Bsmt_SF_arr6, arr7, imported_inter_Garage_Area_arr8, arr9, arr9_squared, Garage_Area_and_House_Age_interaction),axis=1)


'Decision Tree:'

'Trained a model using a decision tree with the "X_all_initial_features_with_interaction_and_polyfeature" feature array. As a result, the specific features included in this model were "Lot Area", "Lot Frontage", "Gr Liv Area", "BsmtFin SF 1", "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", "Garage Area", "House Age", "House Age Squared", and interaction(House Age * Garage Area).'

#Function to grid search through hyperparameters for decision tree regressor within K-fold cross validation to find optimal hyperparameters which result in the lowest average mse across all folds.
##decision_tree_with_grid_search_within_cross_validation(X_all_initial_features_with_interaction_and_polyfeature, y_float, k=5)

#Optimal Hyperparameters: {'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 2, 'max_features': None, 'max_leaf_nodes': 100, 'average mse': 1212117372.184584}



#Function to train a model using a decision tree within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_decision_tree_model_outside_CV_from_all_examples_using_X_all_initial_features_with_interaction_and_polyfeature = train_decision_tree_regression_within_CV_and_outside_on_all_examples(X_all_initial_features_with_interaction_and_polyfeature, y_float, k=5, max_depth=10, min_samples_split=20, min_samples_leaf=2, max_features=None, max_leaf_nodes=100, random_state=42)


#Average MSE across 5 folds: 1212117372.184584
#Final MSE from training on all examples: 452358922.6149247

'Random Forest:'

'Trained a model using a random forest with the "X_all_initial_features_with_interaction_and_polyfeature" feature array. As a result, the specific features included in this model were "Lot Area", "Lot Frontage", "Gr Liv Area", "BsmtFin SF 1", "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", "Garage Area", "House Age", "House Age Squared", and interaction(House Age * Garage Area).'

#Optimal hyperparameters regarding decision trees to be used in this corresponding random forest model.

#Optimal Hyperparameters: {'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 2, 'max_features': None, 'max_leaf_nodes': 100, 'average mse': 1212117372.184584}




#Function to train a model using a random forest within K-fold cross validation and outside CV on all examples.
##_,_,_,trained_random_forest_model_outside_CV_from_all_examples_using_X_all_initial_features_with_interaction_and_polyfeature = train_random_forest_regression_within_CV_and_outside_on_all_examples(X_all_initial_features_with_interaction_and_polyfeature, y_float, k=5, n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=2, max_features=None, max_leaf_nodes=100, random_state=42)

#Average MSE across 5 folds: 937027398.520839
#Final MSE from training on all examples: 498440939.626371

'Comparing both models by the average mse across all folds from cross validation.'

final_average_mse_across_all_folds_of_each_tree_based_model_using_X_all_initial_features_with_interaction_and_polyfeature = {'the unscaled decision tree model including the "X_all_initial_features_with_interaction_and_polyfeature" features': 1212117372.184584, 'the unscaled random forest model including the "X_all_initial_features_with_interaction_and_polyfeature" features': 937027398.520839}

min_key_13 = min(final_average_mse_across_all_folds_of_each_tree_based_model_using_X_all_initial_features_with_interaction_and_polyfeature, key=final_average_mse_across_all_folds_of_each_tree_based_model_using_X_all_initial_features_with_interaction_and_polyfeature.get)
##print(f"Model with minimum average MSE: {min_key_13}, MSE: {final_average_mse_across_all_folds_of_each_tree_based_model_using_X_all_initial_features_with_interaction_and_polyfeature[min_key_13]}")

#Model with minimum average MSE: the unscaled random forest model including the "X_all_initial_features_with_interaction_and_polyfeature" features, MSE: 937027398.520839
#Final MSE from training on all examples: 498440939.626371


'''
Plotting the regression plots of the optimal random forest model including "X_all_initial_features"(see "./Visualizations/Regression_Plots/Random_Forest_Models/Fig_8_Regression_Plots_of_each_Initially_Selected_Unscaled_Feature_with_Interaction_and_Polynomial_vs_Housing_Price_Target_from_Optimal_Random_Forest_Model").

***Note*** Each specific feature regression is plotted where the other feature values are held constant at their means. For the "House Age" feature, the "House Age" and "House Age Squared" features are combined and plotted together in one subplot. However, because random forests model non-linear relationships implictly, this combined visualization is simply visualizing the original and squared "House Age" terms together and not visualizing the enforcement of the combined linear and quadratic behaviors like in the optimal gradient descent model. These features were not scaled before plotting.
'''

#List of feature labels.
optimal_random_forest_model_feature_list_3 = ['Lot Area(sqft)', 'Lot Frontage(sqft)', 'Gross Living Area(sqft)', 'Basement Finished SF 1(sqft)', 'Basement Unfinished SF(sqft)', 'Total Basement SF(sqft)', '1st Floor SF(sqft)',  'Garage Area(sqft)', 'House Age(years)', 'House Age Squared(years)', 'Interaction: "House Age * Garage Area" (years * sqft)']

#Figure title.
optimal_random_forest_model_title_3 = 'Figure 8: Regression Plots of each Initially Selected Unscaled Feature with Interaction/Polynomial Terms vs Housing Price from the Optimal Random Forest Model'

#Function to plot the regressions from tree-based models..
##plot_isolated_feature_regressions_from_tree_based_model_with_polyfeature_combined_with_rowsandcols(X_all_initial_features_with_interaction_and_polyfeature, y_float, model=trained_random_forest_model_outside_CV_from_all_examples_using_X_all_initial_features_with_interaction_and_polyfeature, feature_labels=optimal_random_forest_model_feature_list_3, title=optimal_random_forest_model_title_3, original_indices=[0,1,2,3,4,5,6,7,8,10], poly_index=8, n_cols=3)







'''

Final comparision of all optimal models throughout the entire analysis:

Comparing the optimal models throughout the entire analysis by the average mse across all folds from cross validation. 



Specifically the models compared are: 


the standard scaled gradient descent model including the "X_selected_with_composite_interaction_and_polyfeature" features.

the unscaled decision tree model including the "X_selected_with_composite_and_polyfeature" features.

the unscaled random forest model including the "X_selected_with_composite_2_interaction_and_polyfeature" features.

the unscaled random forest model including the "X_all_initial_features" features.

the unscaled random forest model including the "X_all_initial_features_with_interaction_and_polyfeature" features.

'''


final_average_mse_across_all_folds_of_each_optimal_model_throughout_entire_analysis = {'the standard scaled gradient descent model including the "X_selected_with_composite_interaction_and_polyfeature" features': 1628469596.294723, 'the unscaled decision tree model including the "X_selected_with_composite_and_polyfeature" features': 1163459883.8630824, 'the unscaled random forest model including the "X_selected_with_composite_2_interaction_and_polyfeature" features': 1054769392.7928307, 'the unscaled random forest model including the "X_all_initial_features" features': 981238858.2061069, 'the unscaled random forest model including the "X_all_initial_features_with_interaction_and_polyfeature" features': 937027398.520839}

min_key_14 = min(final_average_mse_across_all_folds_of_each_optimal_model_throughout_entire_analysis, key=final_average_mse_across_all_folds_of_each_optimal_model_throughout_entire_analysis.get)
##print(f"Model with minimum average MSE: {min_key_14}, MSE: {final_average_mse_across_all_folds_of_each_optimal_model_throughout_entire_analysis[min_key_14]}")

#Model with minimum average MSE: the unscaled random forest model including the "X_all_initial_features_with_interaction_and_polyfeature" features, MSE: 937027398.520839
#Final MSE from training on all examples: 498440939.626371
#See "./Visualizations/Regression_Plots/Random_Forest_Models/Fig_8_Regression_Plots_of_each_Initially_Selected_Unscaled_Feature_with_Interaction_and_Polynomial_vs_Housing_Price_Target_from_Optimal_Random_Forest_Model" for visualization of this model.


'''

Examining feature importance for the optimal unscaled "X_all_initial_features_with_interaction_and_polyfeature" random forest model by comparing feature importance scores across all features included in the model(see ./Visualizations/Bar_Plots/Fig_10_Bar_Plot_of_Feature_Importance_Scores_from_Optimal_Unscaled_Random_Forest_Model). 

***NOTE*** Feature importance scores were calculated from the final models trained on all examples. As a result, feature importance scores(i.e., Mean Decrease in Prediction Error) are based on how much each feature contributes to reducing the final mse in the model trained on all examples.

'''


#Function to compute and plot the feature importance scores for a random forest model using a bar plot. 
##compute_feature_importance_for_random_forest(X_all_initial_features_with_interaction_and_polyfeature, feature_names=optimal_random_forest_model_feature_list_3, model=trained_random_forest_model_outside_CV_from_all_examples_using_X_all_initial_features_with_interaction_and_polyfeature)


'''

Features top to bottom of highest importance in reducing the final mse to lowest importance: 


Gross Living Area                                        Importance Score: 0.2159126527631287

House Age                                                Importance Score: 0.19833643166027468

House Age Squared                                        Importance Score: 0.19073207743915713

Garage Area                                              Importance Score: 0.14205173482482966

Total Basement SF                                        Importance Score: 0.09096124976677596

1st Floor SF                                             Importance Score: 0.08201253841054579

Basement Finished SF 1                                   Importance Score: 0.03169897586138262

Lot Area                                                 Importance Score: 0.028024696347928115

Interaction: "House Age * Garage Area"                   Importance Score: 0.008504458575943232

Lot Frontage                                             Importance Score: 0.0062098507605374805

Basement Unfinished SF                                   Importance Score: 0.005555333589496613


'''






'''

Discussion:


Models Trained with Gradient Descent:


The standard scaled model using the "X_selected_with_composite_interaction_and_polyfeature" feature array was the best performing model(average mse across all folds = 1628469596.294723) when training with gradient descent. Standard scaling was the ideal scaling method as 5 out of the 7 models performed best with standard scaling. This may indicate that the effect of outliers throughout these features was not strong enough to require a robust scaling method which helps manage the effects of outliers on model performance(see "./Visualizations/Regression_Plots/Gradient_Descent_Models/Fig_4_Regression_Plots_of_each_Feature_vs_Housing_Price_Target _from_Optimal_Gradient_Descent_Model"). 


Training with gradient descent benefitted from a richer feature space as model performance increased sequentially with each addition of composite, interaction, and polynomial features. The "Gr Liv Area" feature was the most important feature for predicting the housing price target due to its weight coefficient being the largest(weight = |31407.76439351|). The interaction term(House Age * Garage Area) was the least important feature for predicting the housing price target due to its weight coefficient being the smallest(weight = |-19924.40444393|). This may indicate that home buyers in Ames, Iowa do not specifically prefer the combination of a recently built home with a large garage enough to pay a substantial amount more for the home(see ./Visualizations/Bar_Plots/Fig_9_Bar_Plot_of_Feature_Importance_from_Optimal_Standard_Scaled_Gradient_Descent_Model). 


However, home buyers still value the age of the house and the size of the garage individually as the "House Age"(weight = |-24335.53562665|) and "Garage Area"(weight = |25835.96670591|) features had the 3rd and 2nd largest weight coefficients. Also, the "House Age Squared" feature had the 2nd smallest weight coefficient(weight = |20396.29515505|) and was 16.2% smaller than the "House Age" weight coefficient. This is likely because "House Age" exhibited only a mild quadratic relationship with the housing price target.


Improvements: 

The optimal standard scaled multiple linear regression model trained with gradient descent appears to still be underfitting target predictions due to the high average mse across all folds. Some potential drivers of this underwhelming model performance may be the limitations of linear regression models. Linear regression models are often constrained when estimating coefficients for non-linear feature/target relationships. Adding more polynomial terms, especially for the interaction(House Age * Garage Area), across different features may help estimate regression coefficients for more complex feature relationships with the target. 


Also, the composite feature(Total Bsmt SF + 1st Flr SF + Garage Area) appears to have a regression coefficient which consistently underestimates the target prediction. Because "1st Flr SF" is highly correlated with "Total Bsmt Area" combining these features in an additive way may have introduced redundancy in the model, complicating parameter optimization. Also, these 3 features included in the composite feature do not have very comparable ranges(peak-to-peak ranges = [6.11000e+03, 4.76100e+03, 1.48800e+03]). This may have resulted in gradient descent disproportionately estimating the weight parameter in order to compensate for the difference in magnitudes. Scaling these features before combining them may help ensure that each feature contributes more evenly to the resulting composite feature. 


Similarly, additive composite features assume that the effects of the included features on the target are independent and linear. Interaction terms often improve model performance especially when there are strong dependencies between the features. This is because interaction terms help model more complex, non-linear relationships between features and whether the effect of one feature on the target is influenced by another feature. Including multiplicative interaction terms, specifically(1st Flr SF * Total Bsmt SF), could help model any potential interaction effects between these highly correlated features while also not introducing the same degree of redundancy since this term is providing new information and capturing a different relationship. Also, exploring other methods to engineer a more continuous distribution in the "Total Bsmt SF" feature may improve model performance as this was the initial motivation to include composite features. 


Additionally, heteroscedasticity seems to be present in the model, most notably in the distribution of the "Garage Area" feature against the target. This occurs when the variance of errors are not constant throughtout each feature in a regression model and can lead to less precise coefficient estimates. Log transformations to the features or even the target could help stablize this error variance since the variance appears to increase with larger feature values. Also, implementing other regression techniques like Huber regression which modify the regression model to be more robust against variations in the variance of errors could help mitigate the effects of heteroscedasticity on model performance. Plotting these errors or residuals could also help visualize any regions where the model may be consistently overestimating or underestimating the target. 


Finally, it is worth exploring other optimization algorithms outside of the relatively simple gradient descent algorithm implemented in this analysis. Notably, the Adam optimizer or Adaptive Moment Estimation algorithm is known for its powerful balance of training cost and performance. This algorithm combines the gradient descent with momentum and Root Mean Square Propagation(RMSP) learning algorithms for an optimized gradient descent algorithm which adapts descent after each iteration for more controlled and efficient training. Also, examining the Variance Inflation Factor(VIF) for each feature could help to better understand potential multicollinearity present among features in more detail. This could provide important insight on approaches to help manage these relationships.







Models Trained with Decision Trees and Random Forests: 


The unscaled model including the "X_all_initial_features_with_interaction_and_polyfeature" feature array was the best performing model(average mse across all folds = 937027398.520839) when training with tree-based methods(this model was specifically trained using a random forest). This model was also the best performing model across all the models trained in this analysis(see "./Visualizations/Regression_Plots/Random_Forest_Models/Fig_8_Regression_Plots_of_each_Initially_Selected_Unscaled_Feature_with_Interaction_and_Polynomial_vs_Housing_Price_Target_from_Optimal_Random_Forest_Model"). 


Training with tree-based methods also benefitted from a richer feature space. The two best performing models in this analysis were trained with a random forest and included all the initially selected features(9 features) and all the initially selected features with interaction and polynomial terms(11 features). Also consistent with the optimal standard scaled gradient descent model, the "Gr Liv Area" was the most important feature(Importance Score = 0.2159126527631287) in predicting the housing price target or more specifically, in reducing the prediction error(final mse). Interestingly, despite random forests ability to capture non-linear relationships implicitly, including the polynomial term(House Age Squared) provided important information in reducing the final mse as it had the 3rd largest importance score(0.19073207743915713). This may indicate that although the quadratic contribution in the "House Age" features relationship with the target is not particularly strong(2nd smallest weight coefficient in the optimal gradient descent model), this "House Age Squared" term provided relevant information for the optimal random forest model to learn from.


In contrast, the "Bsmt Unf SF" feature had the smallest importance score(0.005555333589496613). Similarly, there was a substantial drop in importance score after the "1st Flr SF" feature. This could be a consequence of the weak correlations these features had with the housing price target found in the correlation analysis. These weak correlations may have not provided the random forest algorithm with sufficient trends to efficiently model target predictions. Regardless, the model performance did improve when including these features but this may be a result of the increased information available for training and not that these are very relevant features(see ./Visualizations/Bar_Plots/Fig_10_Bar_Plot_of_Feature_Importance_Scores_from_Optimal_Unscaled_Random_Forest_Model).



Improvements: 

Expanding the values available to search through in the current hyperparameter grid for decision tree training could refine hyperparameter tuning and obtain more optimal hyperparameters through this expanded grid. Specifically, there were some models which obtained values for the "max_leaf_nodes" hyperparamter which were the largest available. Adding some larger values for this hyperparameter could obtain a more optimized hyperparameter configuration. Also, including more hyperparameters in the grid in general could also refine model training. Adding hyperparamters such as "splitter" which defines the strategy used to split nodes and/or "ccp_alpha" which defines the magnitude by which the tree is pruned could help improve model performance. 


Due to the high computation cost for searching through hyperparameters for random forest training, grid search functionality was not implemented during random forest training. Instead, the optimal hyperparameters and corresponding values found during decision tree training(which did include grid search functionality) were included as hyperparameters in random forest training for a given feature configuration. It can not be assumed these hyperparameters would be the same regardless that both models were trained on identical feature arrays. Implementing grid search functionality during random forest training would ensure optimal hyperparameters are found specifically for random forest training. 


Additionally, expanding this proposed random forest grid to include hyperparameters specific to random forest training could help refine this grid search. Including random forest specific hyperparameters in the grid such as "n_estimators" which defines the number of trees in the forest, "bootstrap" which defines whether bootstrap sampling(i.e., random sampling with replacement) is implemented and "max_samples" which defines the fraction of samples used when bootstrapping could improve model performance when training with random forests. 


Finally, although tree-based training methods like decision trees and random forests do not require feature scaling to the same degree as gradient descent-based training methods, substantial differences in feature magnitudes can still, in some cases, impact model performance when training with tree-based methods. It is worth exploring if implementing different scaling methods could improve model performance for tree-based training methods. As mentioned in the gradient descent improvements, including different configurations of interaction and polynomial terms could also help improve performance for these tree-based training methods. 







Final Thoughts:

The models trained across all gradient descent and tree-based methods appeared to underfit target predictions based on trends in the performance metrics and model visualizations. For example, the average mse across all folds throughout all the models are relatively high but there are not substantial differences between these values and the final mse from training on all examples i.e., the average mse is high but the final mse is not substantially lower. This trend indicates more of a underfitting pattern throughout the models than a overfitting pattern. 


However, it is worth noting that from the first model trained(the standard scaled gradient descent model using "X_selected_final") to the best performing model(unscaled random forest model using "X_all_initial_features_with_interaction_and_polyfeature") model performance improved by 50.031% when comparing the average mse between these two models. Also, each step in this analysis generally improved model performance. 


Model performance in this analysis may have been constrained by the available data. There were 79 variables in the Ames Housing dataset. The majority of these variables were non-numerical categorical variables not suitable for a multiple linear regression analysis. From the numerical variables there were also many with very narrow ranges and/or sparse distributions skewed by many zero values. This left the 9 features initially selected for analysis. Due to these underfitting trends, the analysis likely would have benefitted from an increased model complexity by including more features throughout the analysis. Similarly, the 9 features ideal for this linear regression analysis, besides the "House Age" feature, measured some area component of a given house. Including features that measured different aspects of a given house could also have improved model performance as it is unlikely that area measurements is the only house quantity that home buyers value in Ames, Iowa. 


Additonally, introducing more model performance metrics across all training methods such as mean absolute error, R-squared, and Friedman_MSE(for tree-based training) could provide important insight into how these models may be performing better in different contexts. Also, the housing price target distribution wasn't examined in-depth in this analysis. It is worth examining this distribution in more detail and potentially implementing transformations as target distributions that are highly skewed and/or contain extreme outliers can affect model performance. 


The exclusion of the improvements and the data availability constraints mentioned above should provide insight into the results obtained in this analysis.



'''







