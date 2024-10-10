# A_Comprehensive_Regression_Analysis_of_Housing_Prices_in_Ames_Iowa
This repository contains files for a regression analysis of housing prices in Ames, Iowa using various machine learning algorithms. There are 9 project files/folders in the "Ames_Project" folder for this analysis. 

## Objectives
- Practice training regression models with different machine learning algorithms
- Practice implementing different feature selection, scaling, and engineering methods to increase model performance
- Optimize model predictions and obtain a optimal regression model within the scope of this analysis
- Practice different data visualization methods

## Motivations
I am taking a machine learning specialization and wanted to apply some of the methods taught in this specialization by conducting my own regression analysis using some standard and more advanced machine learning algorithms. I chose the Ames Housing Dataset because this is the dataset used in some of the initial courses in the specialization. 

## Reading Through Files
Below are descriptions of the project files in the order you should go about reading them. Note, the Python scripts in this analysis function more like a notebook where running the script as is will not yield a visual output. 

In other words, each function call or print statement is commented out with 2 pound signs(##) and regular comments with 1 pound sign(#). The relevant outputs for each function call are listed and there are detailed explanations throughout each step of the analysis including a comprehensive discussion of the final results. I recommend you read through each script and un-comment any functions you wish to run if you would like understand the functionality more. Make sure to re-comment the functions after running them. 

For the visualization functions, refer to the finished plots provided in the "Visualizations" folder. Running these functions may result in messy plots because the same functions were used for several plots so the configurations may not be tuned for each specific plot.

### "Ames_Project_env_requirements.txt"
This is a text file containing the requirements for the virtual environment used in this analysis.

### "AmesHousing.csv"
This is a CSV file containing the Ames Housing Dataset which was used to select suitable features for this analysis.

### "Project_Tools.py"
This is a Python script file containing all the functions used in this analysis. The functions are categorize by if they were explicitly used in the analyses or not and other sub-categories related to their functionalities. A detailed description of the functionality, parameters, and outputs is provided for each function. Please refer to this script while reading through the analyses.

### "Interpolation_of_missing_values.py" and "Interpolated_arrays"
This is a Python script file containing code to clean up/fill missing values of the initial features using linear interpolation. This script stored these interpolated feature arrays in the "Interpolated_arrays" folder.

### "Correlation_Analysis_for_Feature_Selection.py"
This is a Python script file containing code for a correlation analysis for feature selection. This script quantifies linear and non-linear correlations between features and the housing price target and between features and other features. 

### "Model_Training.py"
This is a Python script file containing the code for training different regression models. This is the main analysis. 

The general outline for this analysis is as follows:
- Trained models with gradient descent using the selected features from "Correlation_Analysis_for_Feature_Selection.py"
- Trained models with gradient descent using different feature configurations from various feature engineering implementations
- Trained models with tree-based methods including decision trees and random forests
- Discussion

### "Visualizations" 
This is a folder containing various scatter, bar, and isolated regression plots constructed throughout the analyses. The analysis scripts will indicate when to ideally view these plots.

### "__pycache__"
This is a folder containing different Python bytecode documents (i.e., cache files for the different scripts used in this analysis). *Not relevant*





