'''Script to fill any missing values of each feature array using linear interpolation'''

import numpy as np 
import pandas as pd
from Project_Tools import fill_missing_values_with_interpolation


#Reading the Ames dataset in a Dataframe.
data = pd.read_csv('AmesHousing.csv')

#Parsing the Ames dataset for relevant features to be used in this analysis.
arr1 = data['Lot Area'].values
arr2 = data['Lot Frontage'].values
arr3 = data['Gr Liv Area'].values
arr4 = data['BsmtFin SF 1'].values
arr5 = data['Bsmt Unf SF'].values
arr6 = data['Total Bsmt SF'].values
arr7 = data['1st Flr SF'].values
arr8 = data['Garage Area'].values
arr9 = data['Year Built'].values
y = data['SalePrice'].values

#Loop to calculate the "House Age" feature array from the "Year Built" feature array. 
for i in np.arange(len(arr9)): 
        arr9[i] = 2024 - arr9[i]
        
        
        
#Filled missing values using linear interpolation. 
filled_arr2 = fill_missing_values_with_interpolation(arr2)
filled_arr4 = fill_missing_values_with_interpolation(arr4)
filled_arr5 = fill_missing_values_with_interpolation(arr5)
filled_arr6 = fill_missing_values_with_interpolation(arr6)
filled_arr8 = fill_missing_values_with_interpolation(arr8)


'''

There were 490 NaN values in the "Lot Frontage" feature array, accounting for 16.7% of the total number of "Lot Frontage" examples.
There was 1 NaN value in the "BsmtFin SF 1" feature array.
There was 1 NaN value in the "Bsmt Unf SF" feature array. 
There was 1 NaN value in the "Total Bsmt SF" feature array.
There was 1 NaN value in the "Garage Area" feature array.
There were 0 NaN values in the remaining feature arrays. There were also 0 NaN values in the "SalePrice" target array. 

'''


#Created CSV files for each interpolated feature array.
file_name1 = 'inter_arr2.csv'
np.savetxt(file_name1, filled_arr2)
print(f"Array saved to {file_name1} in the current working directory.")

file_name2 = 'inter_arr4.csv'
np.savetxt(file_name2, filled_arr4)
print(f"Array saved to {file_name2} in the current working directory.")

file_name3 = 'inter_arr5.csv'
np.savetxt(file_name3, filled_arr5)
print(f'Array saved to {file_name3} in the current working directory')

file_name4 = 'inter_arr6.csv'
np.savetxt(file_name4, filled_arr6)
print(f'Array saved to {file_name4} in the current working directory')

file_name5 = 'inter_arr8.csv'
np.savetxt(file_name5, filled_arr8)
print(f'Array saved to {file_name5} in the current working directory')



