#loading the libraries
import pandas  as pd
df = pd.read_csv("/home/waswa/Downloads/MD_agric_exam-4313.csv")
df.head()

print(df.shape)

df.describe()


# Extract and print the column names
column_names = df.columns
print(column_names)

#determine the number of unique crop types
unique_crop_types = df["Crop_type"].unique()
num_unique_crop_type = len(unique_crop_types)

print("Unique crop:", num_unique_crop_type)

#filter the dataset to include only rows where crop type is wheat
wheat = df[df["Crop_type"]=="wheat"]

#identify maximum annual yield for wheat
max_wheat = wheat["Annual_yield"].max()

#print the max annual yield for wheat rounded to 2 decimal places
print("Maximun annual yield for wheat:", round(max_wheat, 2))

#Find the total rainfall for crop types where the average pollution level is above 0.2. 
# Calculate average pollution level for each crop type
average_pollution = df.groupby('Crop_type')['Pollution_level'].mean()

# Filter crop types where the average pollution level is above 0.2
high_pollution_crop_types = average_pollution[average_pollution > 0.2].index

# Filter the DataFrame to include only rows where the 'Crop_type' is in high_pollution_crop_types
filtered_data = df[df['Crop_type'].isin(high_pollution_crop_types)]

# Calculate total rainfall for crop types where the average pollution level is above 0.2
total_rainfall = filtered_data['Rainfall'].sum()

# Print the total rainfall for crop types with average pollution level above 0.2
print("Total rainfall for crop types with average pollution level above 0.2:", total_rainfall)

#Write a function to calculate the temperature range (Max_temperature_C - Min_temperature_C) for each farmer's field. Then, call the function with the following `Field_ID`: `1458`, `1895`, and `5443`. What are the results of these 3 calls?

def calculate_temperature_range(field_id):
    # Filter the DataFrame for the specified Field_ID
    field_data = df[df['Field_ID'] == field_id]
    
    # Calculate the temperature range for the field
    temperature_range = field_data['Max_temperature_C'].iloc[0] - field_data['Min_temperature_C'].iloc[0]
    
    return temperature_range

# Call the function with the specified Field_IDs
field_ids = [1458, 1895, 5443]
for field_id in field_ids:
    temperature_range = calculate_temperature_range(field_id)
    print(f"Temperature range for Field ID {field_id}: {temperature_range}")

#What does the following code achieve?

a = df['Crop_type'].unique()

b = float('inf')

c = ''

for crop in a:

   
    d = df[df['Crop_type'] == crop]['Min_temperature_C'].mean()
    if d < b:

        b = d

        c = crop

print(c)

#Write code to calculate the total plot size for plots where the pH is less than 5.5.

#filter the rows to include those with pH of less than 5.5
filtered_data = df[df["pH"] < 5.5]

#Calculate the total plot size 
total_plot_size = filtered_data["Plot_size"].sum()

#print the total plot size
print("Total plot size where pH is less than 5.5", total_plot_size)

#Using Pandas, create a dataframe that includes entries with a 'Min_temperature_C’< -5 and a 'Max_temperature_C' > 30. How many rows are in the filtered dataset?


# Filter the DataFrame based on conditions
filtered_data = df[(df['Min_temperature_C'] < -5) & (df['Max_temperature_C'] > 30)]

# Calculate the number of rows in the filtered dataset
num_rows_filtered = len(filtered_data)

# Print the number of rows in the filtered dataset
print("Number of rows in the filtered dataset:", num_rows_filtered)

#Using Numpy, calculate the standard deviation of the 'Rainfall' for plots where the 'Plot_size' is larger than the median plot size of the dataset (round to 2 decimal places).
import numpy as np

# Calculate the median plot size using NumPy
median_plot_size = np.median(df['Plot_size'])

# Filter the dataset to include only plots where the 'Plot_size' is larger than the median plot size
filtered_data = df[df['Plot_size'] > median_plot_size]

# Calculate the standard deviation of the 'Rainfall' for the filtered dataset
std_rainfall = np.std(filtered_data['Rainfall'])

# Round the standard deviation to 2 decimal places
std_rainfall_rounded = round(std_rainfall, 2)

# Print the result
print("Standard deviation of Rainfall for plots with Plot_size larger than the median:", std_rainfall_rounded)

#If you concatenate the first three digits of the most common ‘Max_temperature_C’ 
#with the last three letters of the least common 'Crop_type', what string do you get?
#Note: Use the first mode if there are multiple modes

# Find the most common Max_temperature_C (using first mode if there are multiple modes)
most_common_temp = df['Max_temperature_C'].mode()[0]

# Find the least common Crop_type
least_common_crop = df['Crop_type'].value_counts().index[-1]

# Concatenate the first three digits of the most common Max_temperature_C with the last three letters of the least common Crop_type
result_string = str(most_common_temp)[:3] + least_common_crop[-3:]

# Print the resulting string
print("Resulting string:", result_string)


#Write Python code to create a violin plot visualising the distribution of 'Annual_yield' across different 'Elevation' ranges. 
#se the provided elevation range categories (Low: < 300m, Medium: 300m - 600m, High: > 600m) to categorise the data before plotting. 
#Examine the violin plot displaying the distribution of 'Annual Yield' across three elevation categories (Low, Medium, High). 
#What insight does the violin plot provide regarding the relationship between elevation ranges and annual yield distribution?


import seaborn as sns
import matplotlib.pyplot as plt

# Categorize data based on elevation ranges
def categorize_elevation(elevation):
    if elevation < 300:
        return 'Low'
    elif elevation <= 600:
        return 'Medium'
    else:
        return 'High'

df['Elevation_Category'] = df['Elevation'].apply(categorize_elevation)

# Create violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='Elevation_Category', y='Annual_yield', data=df)
plt.title('Distribution of Annual Yield Across Elevation Categories')
plt.xlabel('Elevation Category')
plt.ylabel('Annual Yield')
plt.show()


#Assuming each 'Crop_type' contributes an integer value equal to its length (e.g., 'wheat' contributes 5), write a recursive function to sum the integer values for each unique crop type in the dataset. 
#What is the sum?

#Define the recursive function to sum the integer values for each unique crop type
def sum_crop_type_lengths(df, crop_types=None):
    # Initialize the dictionary to store crop types and their corresponding lengths
    if crop_types is None:
        crop_types = {}

    # Base case: if the DataFrame is empty, return the sum of lengths of all unique crop types
    if df.empty:
        return sum(len(crop) for crop in crop_types.keys())

    # Recursive case: process the first row of the DataFrame
    # Update the crop_types dictionary with the length of the current Crop_type
    crop_type = df.iloc[0]['Crop_type']
    if crop_type in crop_types:
        crop_types[crop_type] += len(crop_type)
    else:
        crop_types[crop_type] = len(crop_type)

    # Recursively call the function with the remaining rows of the DataFrame
    return sum_crop_type_lengths(df.iloc[1:], crop_types)

# Call the recursive function
total_sum = sum_crop_type_lengths(df)

# Print the sum of integer values for each unique crop type
print("Total sum of integer values for each unique crop type:", total_sum)

#Write Python code to perform a t-test comparing the average 'Annual_yield' between 'coffee' and 'banana' crop types using scipy.stats. 
#What is the p-value, rounded to three decimal places?
import pandas as pd
from scipy.stats import ttest_ind

# Read CSV file into a DataFrame
data = pd.read_csv('Md_Agric.csv')

# Extract data for 'coffee' and 'banana' crop types
coffee_yield = data[data['Crop_type'] == 'coffee']['Annual_yield']
banana_yield = data[data['Crop_type'] == 'banana']['Annual_yield']

# Perform t-test
t_statistic, p_value = ttest_ind(coffee_yield, banana_yield)

# Print the p-value rounded to three decimal places
print("P-value:", round(p_value, 3))
