import pandas as pd

# Assuming you have two CSV files: dataset1.csv and dataset2.csv
file_path1 = 'Karnataka_Watershed_Labeled.csv'
file_path2 = 'Karnataka_Watershed_unLabeled.csv'

# Load the datasets into DataFrames
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Concatenate the DataFrames along rows (axis=0)
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_dataset.csv', index=False)
