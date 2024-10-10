'''
Correlation Analysis for Feature Selection


Research Questions: 

Q1: Which features have the strongest correlations with the housing price target?

Q2: Which features have the strongest correlations with other features?
'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from Project_Tools import analyze_data, plot_feature_vs_target_scatter
from scipy.stats import normaltest
from astropy.stats import biweight_midcorrelation
import dcor


#Reading the Ames dataset into a dataframe.
data = pd.read_csv('AmesHousing.csv')

#Parsing this dataframe for suitable features to be used in this analysis. 

'''
***NOTE*** The features below were initially selected from the entire AmesHousing dataset. In order to obtain ideal features for a multiple linear regression analysis, these initial features were selected based on if they were numerical variables with densely distributed values and measured on a continuous scale i.e., measured on a interval or ratio scale. I avoided selecting nurmerical variables which appeared to be measured on a continuous scale but exhibited a sparse distribution of values or had a very narrow range of values.
'''

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


#Stacked each initial 1-D feature arrays into a 2-D feature array with shape (2930,9).
X = np.stack((arr1, imported_inter_Lot_Frontage_arr2, arr3, imported_inter_BsmtFin_SF_1_arr4, imported_inter_Bsmt_Unf_SF_arr5, imported_inter_Total_Bsmt_SF_arr6, arr7, imported_inter_Garage_Area_arr8, arr9), axis=1)




'''

This correlation analysis will be conducted in 3 steps.

Step 1: Perfom normality tests and examine outliers to determine the optimal correlation metrics to implement in this analysis.


Step 2: Quantify the linear or non-linear correlations between the housing price target and each feature. These correlations with the housing price target can provide useful information as to which features would be more relevant to include in further analysis. Features with strong correlations with the target likely provide more relevant information for learning algorithms to utilize in parameter optimization. 

***NOTE*** Features with moderate to strong(coefficient > |0.5|) linear or non-linear correlation coefficients will be selected for further analysis based on their respective linear or non-linear correlations with the housing price target. In other words models will be optimized, potentially using features with linear correlations and features with non-linear correlations, in relation to the housing price target.


Step 3: Examine the selected groups of features(linear or non-linear features) to quantify any linear or non-linear correlations among the features themselves. This could provide useful information about any multicollinearity present among features to help better approach how to managing these effects.

***NOTE*** Features with moderate to strong(coefficient > |0.5|) linear or non-linear correlations with other features will be omitted or managed for further analysis.

'''





'''

Step 1: Perfom normality tests and count outliers to determine the optimal correlation metrics to implement in this analysis. 

'''

#Stacked all of the initial feature arrays including the housing price target array into a 2-D variable array for normality analysis.
X_with_target = np.stack((arr1, imported_inter_Lot_Frontage_arr2, arr3, imported_inter_BsmtFin_SF_1_arr4, imported_inter_Bsmt_Unf_SF_arr5, imported_inter_Total_Bsmt_SF_arr6, arr7, imported_inter_Garage_Area_arr8, arr9, y_float), axis=1)


#Converted feature array with target into dataframe with column labels.
df = pd.DataFrame(X_with_target, columns=['Lot Area','Lot Frontage','Gr Liv Area','BsmtFin SF 1','Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', 'Garage Area', 'House Age', 'Housing Price Target'])

#Function to examine the normality of distributions and count outliers of a dataframe. 
##analyze_data(df)

'''

Results: 

Lot Area:

Normality test p-value: 0.0000
Lot Area is not normally distributed (p <= 0.05)
Number of outliers detected in Lot Area: 127


Lot Frontage:

Normality test p-value: 0.0000
Lot Frontage is not normally distributed (p <= 0.05)
Number of outliers detected in Lot Frontage: 220


Gr Liv Area:

Normality test p-value: 0.0000
Gr Liv Area is not normally distributed (p <= 0.05)
Number of outliers detected in Gr Liv Area: 75


BsmtFin SF 1:

Normality test p-value: 0.0000
BsmtFin SF 1 is not normally distributed (p <= 0.05)
Number of outliers detected in BsmtFin SF 1: 15


Bsmt Unf SF:

Normality test p-value: 0.0000
Bsmt Unf SF is not normally distributed (p <= 0.05)
Number of outliers detected in Bsmt Unf SF: 56


Total Bsmt SF:

Normality test p-value: 0.0000
Total Bsmt SF is not normally distributed (p <= 0.05)
Number of outliers detected in Total Bsmt SF: 123


1st Flr SF:

Normality test p-value: 0.0000
1st Flr SF is not normally distributed (p <= 0.05)
Number of outliers detected in 1st Flr SF: 43


Garage Area:

Normality test p-value: 0.0000
Garage Area is not normally distributed (p <= 0.05)
Number of outliers detected in Garage Area: 42


House Age:

Normality test p-value: 0.0000
House Age is not normally distributed (p <= 0.05)
Number of outliers detected in House Age: 9


Housing Price Target:

Normality test p-value: 0.0000
Housing Price Target is not normally distributed (p <= 0.05)
Number of outliers detected in Housing Price Target: 137





All of the initial features and the housing price target do not follow a normal distribution. Most of these variables have a small number of outliers proportional to the total number of corresponding examples. 

The features with the largest proportion of outliers include "Lot Frontage"(7.5% of examples), "Lot Area"(4.3% of examples), and "Total Bsmt SF"(4.2% of examples). Also, 4.6% of the total number of housing price target examples were outliers. 



Given the results of this first step of the correlation analysis, the biweight midcorrelation and distance correlation tests will be used in the following steps of this correlation analysis. 

Both the biweight midcorrelation and distance correlation tests are robust, non-parametric tests which do not assume a normal distribution and are not sensitive to outliers. 

'''






'''

Step 2: Quantify the linear or non-linear correlations between the housing price target and each feature.

'''

#Used the "astropy" package to calculate the biweight midcorrelation coefficient for each feature in relation to the housing price target(y). This analysis was used to quantify the strength and direction of linear correlations between the housing price target and each feature. 
n = X.shape[1]
bicor_results = []

for i in range(n):
    bicor_value = biweight_midcorrelation(y_float, X[:,i])
    bicor_results.append(bicor_value)
    
##print(bicor_results)

#Biweight midcorrelation coefficients for each feature in relation to the housing price target(y).
''' 
    Lot Area = 0.3740438654480769
    
    Lot Frontage = 0.3180776857273721
    
    Gr Liv Area = 0.69111458797615
    
    BsmtFin SF 1 = 0.3422067115419598
    
    Bsmt Unf SF = 0.19329110579975603
    
    Total Bsmt SF = 0.6006324579933181
    
    1st Flr SF = 0.5726294231901552
    
    Garage Area = 0.6212540270993714
    
    House Age = -0.6255429588732524 
    
'''


#Used the "dcor" package to calculate the distance correlation coefficient for each feature in relation to the housing price target(y_float). This analysis was used to quantify the strength of linear or non-linear correlations between the housing price target and each feature. 
#***NOTE*** the "dcor.distance_correlation()" method requires all arguments to be floating point numbers. 
distcor_results = []
for i in range(n):
    distcor_value = dcor.distance_correlation(y_float, X[:,i])
    distcor_results.append(distcor_value)
    
##print(distcor_results)

#Distance correlation coefficients for each feature in relation to the housing price target(y_float).
''' 
    Lot Area = 0.39447236070644814
    
    Lot Frontage = 0.35680703982339335
    
    Gr Liv Area = 0.696944380820689
    
    BsmtFin SF 1 = 0.37969478526284833
    
    Bsmt Unf SF = 0.18722415939502923
    
    Total Bsmt SF = 0.6144704658087545
    
    1st Flr SF = 0.5902850577193922
    
    Garage Area = 0.6434881317294235
    
    House Age = 0.6308964201727857 
    
'''


#Given the results of this second step in the correlation analysis, there were 5 features that indicated moderate linear correlations with the housing price target. 
#These features included: 
'''

Gr Liv Area: (biweight=0.69111458797615, distcor=0.696944380820689)

Total Bsmt SF: (biweight=0.6006324579933181, distcor=0.6144704658087545)

1st Flr SF: (biweight=0.5726294231901552, distcor=0.5902850577193922)

Garage Area: (biweight=0.6212540270993714, distcor=0.6434881317294235)

House Age: (biweight=-0.6255429588732524, distcor=0.6308964201727857)

'''
#These features will be selected for further analysis. 
#***NOTE*** A feature with a low biweight correlation coefficient(biweight < |0.5|) and a high distance correlation coefficient(distcor > 0.5) would have inferred a moderate to strong non-linear correlation with the housing price target. However, there were no features which inferred this non-linear correlation. 

#***NOTE*** It is worth mentioning that these correlation metrics work best to reveal non-linear correlations when there are stark contrasts between these coefficients like mentioned above i.e., a low biweight and a high distance correlation. It is possible for a feature to exhibit both linear and non-linear correlations where a curvilinear regression model could be more optimal in estimating target predictions. However, in that case, these correlation metrics will not clearly reveal those correlations and further examination of the exact nature of those relationships would be required. 




'''

Step 3: Examine the selected group of features(linear features) to quantify any linear or non-linear correlations among the features themselves.

'''
 
       
    
#Stacking the selected 1-D feature arrays into a 2-D feature array. 
X_selected = np.stack((arr3, imported_inter_Total_Bsmt_SF_arr6, arr7, imported_inter_Garage_Area_arr8, arr9),axis=1)
  
    
    
    
'Linear and non-linear correlations between the "Gr Liv Area" feature and each other feature.'

#Calculating linear correlations between the "Gr Liv Area" feature and each other feature. 
n_selected = X_selected.shape[1]
Gr_Liv_Area_lin_feat_cor = []
for i in range(n_selected):
    bicor_value = biweight_midcorrelation(arr3, X_selected[:,i])
    Gr_Liv_Area_lin_feat_cor.append(bicor_value)

##print(Gr_Liv_Area_lin_feat_cor)

#Biweight midcorrelation coefficients between the "Gr Liv Area" feature and each other feature. These coefficients indicate the strength and direction of linear correlations.
'''

Gr Liv Area vs Total Bsmt SF = 0.3827372372704082

Gr Liv Area vs 1st Flr SF = 0.4947738451782456

Gr Liv Area vs Garage Area = 0.48721490534075385

Gr Liv Area vs House Age = -0.2862394916273845

'''

#Calculating linear or non-linear correlations between the "Gr Liv Area" feature and each other feature.
Gr_Liv_Area_nonlin_feat_cor = []
arr3_float = arr3.astype(np.float64)
for i in range(n_selected):
    distcor_value = dcor.distance_correlation(arr3_float, X_selected[:,i])
    Gr_Liv_Area_nonlin_feat_cor.append(distcor_value)
    
##print(Gr_Liv_Area_nonlin_feat_cor)

#Distance correlation coefficients between the "Gr Liv Area" feature and each other feature. These coefficients indicate the strength of linear or non-linear correlations.
#***NOTE*** A feature with a low biweight correlation coefficient(biweight < |0.5|) and a high distance correlation coefficient(distcor > 0.5) would infer a moderate to strong non-linear correlation with that corresponding feature.
'''

Gr Liv Area vs Total Bsmt SF = 0.3952864496902644

Gr Liv Area vs 1st Flr SF = 0.4914382643079424

Gr Liv Area vs Garage Area = 0.46998449479578613

Gr Liv Area vs House Age = 0.33224213314454815

'''




'Linear and non-linear correlations between the "Total Bsmt SF" feature and each other feature.'

#Calculating linear correlations between the "Total Bsmt SF" feature and each other feature. 
Total_Bsmt_SF_lin_feat_cor = []
for i in range(n_selected):
    bicor_value = biweight_midcorrelation(imported_inter_Total_Bsmt_SF_arr6, X_selected[:,i])
    Total_Bsmt_SF_lin_feat_cor.append(bicor_value)

##print(Total_Bsmt_SF_lin_feat_cor)

#Biweight midcorrelation coefficients between the "Total Bsmt SF" feature and each other feature. 
'''

Total Bsmt SF vs Gr Liv Area = 0.3827372372704082

Total Bsmt SF vs 1st Flr SF = 0.8204808801718345

Total Bsmt SF vs Garage Area = 0.4730727350045116

Total Bsmt SF vs House Age = -0.4264255779974964

'''

#Calculating linear or non-linear correlations between the " Total Bsmt SF" feature and each other feature.
Total_Bsmt_SF_nonlin_feat_cor = []
imported_inter_Total_Bsmt_SF_arr6_float = imported_inter_Total_Bsmt_SF_arr6.astype(np.float64)
for i in range(n_selected):
    distcor_value = dcor.distance_correlation(imported_inter_Total_Bsmt_SF_arr6_float, X_selected[:,i])
    Total_Bsmt_SF_nonlin_feat_cor.append(distcor_value)
    
##print(Total_Bsmt_SF_nonlin_feat_cor)

#Distance correlation coefficients between the "Total Bsmt SF" feature and each other feature.
'''

Total Bsmt SF vs Gr Liv Area = 0.3952864496902644

Total Bsmt SF vs 1st Flr SF = 0.8453764317822788

Total Bsmt SF vs Garage Area = 0.4736532616343763

Total Bsmt SF vs House Age = 0.42387272014938426

'''





'Linear and non-linear correlations between the "1st Flr SF" feature and each other feature.'

#Calculating linear correlations between the "1st Flr SF" feature and each other feature.
First_Flr_SF_lin_feat_cor = []
for i in range(n_selected):
    bicor_value = biweight_midcorrelation(arr7, X_selected[:,i])
    First_Flr_SF_lin_feat_cor.append(bicor_value)
    
##print(First_Flr_SF_lin_feat_cor)

#Biweight midcorrelation coefficients between the "1st Flr SF" feature and each other feature.
'''

1st Flr SF vs Gr Liv Area = 0.4947738451782456

1st Flr SF vs Total Bsmt Area = 0.8204808801718345

1st Flr SF vs Garage Area = 0.4803512609263313

1st Flr SF vs House Age = -0.3170117437175093

'''

#Calculating linear or non-linear correlations between the " 1st Flr SF" feature and each other feature.
First_Flr_SF_nonlin_feat_cor = []
arr7_float = arr7.astype(np.float64)
for i in range(n_selected):
    distcor_value = dcor.distance_correlation(arr7_float, X_selected[:,i])
    First_Flr_SF_nonlin_feat_cor.append(distcor_value)
    
##print(First_Flr_SF_nonlin_feat_cor)  

#Distance correlation coefficients between the "1st Flr SF" feature and each other feature.
'''

1st Flr SF vs Gr Liv Area = 0.4914382643079424

1st Flr SF vs Total Bsmt Area = 0.8453764317822788

1st Flr SF vs Garage Area = 0.47079835328685704

1st Flr SF vs House Age = 0.3218296245281336

'''





'Linear and non-linear correlations between the "Garage Area" feature and each other feature.'

#Calculating linear correlations between the "Garage Area" feature and each other feature.
Garage_Area_lin_feat_cor = []
for i in range(n_selected):
    bicor_value = biweight_midcorrelation(imported_inter_Garage_Area_arr8, X_selected[:,i])
    Garage_Area_lin_feat_cor.append(bicor_value)
    
##print(Garage_Area_lin_feat_cor)

#Biweight midcorrelation coefficients between the "Garage Area" feature and each other feature.
'''

Garage Area vs Gr Liv Area = 0.48721490534075385

Garage Area vs Total Bsmt Area = 0.4730727350045116

Garage Area vs 1st Flr SF = 0.4803512609263313

Garage Area vs House Age = -0.5193176556071337

'''

#Calculating linear or non-linear correlations between the " Garage Area" feature and each other feature.
Garage_Area_nonlin_feat_cor = []
imported_inter_Garage_Area_arr8_float = imported_inter_Garage_Area_arr8.astype(np.float64)
for i in range(n_selected):
    distcor_value = dcor.distance_correlation(imported_inter_Garage_Area_arr8_float, X_selected[:,i])
    Garage_Area_nonlin_feat_cor.append(distcor_value)
    
##print(Garage_Area_nonlin_feat_cor)

#Distance correlation coefficients between the "Garage Area" feature and each other feature.
'''

Garage Area vs Gr Liv Area = 0.46998449479578613

Garage Area vs Total Bsmt Area = 0.4736532616343763

Garage Area vs 1st Flr SF = 0.47079835328685704

Garage Area vs House Age = 0.5126052142403964

'''






'Linear and non-linear correlations between the "House Age" feature and each other feature.'

#Calculating linear correlations between the "House Age" feature and each other feature.
House_Age_lin_feat_cor = []
for i in range(n_selected):
    bicor_value = biweight_midcorrelation(arr9, X_selected[:,i])
    House_Age_lin_feat_cor.append(bicor_value)
    
##print(House_Age_lin_feat_cor)

#Biweight midcorrelation coefficients between the "House Age" feature and each other feature.
'''

House Age vs Gr Liv Area = -0.2862394916273845

House Age vs Total Bsmt Area = -0.4264255779974964

House Age vs 1st Flr SF = -0.3170117437175093

House Age vs Garage Area = -0.5193176556071337

'''

#Calculating linear or non-linear correlations between the "House Age" feature and each other feature.
House_Age_nonlin_feat_cor = []
arr9_float = arr9.astype(np.float64)
for i in range(n_selected):
    distcor_value = dcor.distance_correlation(arr9_float, X_selected[:,i])
    House_Age_nonlin_feat_cor.append(distcor_value)
    
##print(House_Age_nonlin_feat_cor)
 
#Distance correlation coefficients between the "House Age" feature and each other feature.
'''

House Age vs Gr Liv Area = 0.33224213314454815

House Age vs Total Bsmt Area = 0.42387272014938426

House Age vs 1st Flr SF = 0.3218296245281336

House Age vs Garage Area = 0.5126052142403964

'''




'''

Summary:

The "1st Flr SF" feature had a strong linear correlation(biweight=0.8204808801718345) with the "Total Bsmt Area" feature. 

The "1st Flr SF" feature also had linear correlation coefficients ~ |0.5| with both the "Gr Liv Area" and "Garage Area" features. Therefore, to reduce feature redundancy and multicollinearity among features, the "1st Flr SF" feature will be omitted for now. 



The "Garage Area" feature had a moderate, negative linear correlation(biweight= -0.5193176556071337) with the "House Age" feature. 

This correlation was not as strong as the correlation mentioned above with the coefficient just above the |0.5| threshold. However, the "Garage Area" feature also had linear correlation coefficients ~ |0.5| with the "Gr Liv Area", "Total Bsmt Area", and "1st Flr SF" features. Therefore, to stay consistent to the defined correlation coefficient threshold and to reduce multicollinearity among features, the "Garage Area" feature will be omitted for now.  




***As a result, the features to be used to train the initial model will be the "Gr Liv Area", "Total Bsmt Area", and "House Age" features.***


'''


'Visualized the coordinate distribution for each selected feature vs the housing price target using a scatterplot(see "./Visualizations/Scatter_Plots/Fig_1_Scatterplots_of_each_Selected_Feature_vs_Housing_Price_Target").'

#Constucted a new 2-D feature array with the features selected from the correlation analysis. 
X_selected_final = np.stack((arr3, imported_inter_Total_Bsmt_SF_arr6, arr9),axis=1)

#List of feature labels.
X_selected_final_feature_list = ['Gross Living Area(sqft)','Total Basement Area(sqft)','House Age(years)']

#Title for entire figure.
X_selected_final_plot_title = 'Figure 1: Scatterplots of each Selected Feature vs the Housing Price Target'

#Scatterplot function.
##plot_feature_vs_target_scatter(X_selected_final, y_float, feature_names=X_selected_final_feature_list, target_name='Sale Price($dollars)', title=X_selected_final_plot_title)








