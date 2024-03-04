import pandas as pd
import os
import json

def combine_dataset(data_folder, all_csv_files, out_csv_name):
    data_frames = [] 
    # dtype_info = [] 

    for file in all_csv_files:
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path, low_memory=False)
        
        # Collect the file path and data types information
        # dtype_info.append({
        #     "file_path": file_path,
        #     "file_types": str(df.dtypes.to_dict())  # Convert the dtype Series to a dictionary and then to a string
        # })
        filtered_df = df[(df['RESOURCE_TYPE'] == 'GENERATOR') & (df['MARKETPRODUCTTYPE'] == 'EN')]
        result_df = filtered_df.drop(columns=['STARTTIME_GMT', 'STOPTIME_GMT', 'STARTDATE', 'MARKET_RUN_ID', 'TIMEINTERVALSTART_GMT', 'TIMEINTERVALEND_GMT'])
        data_frames.append(result_df)

    # dtype_df = pd.DataFrame(dtype_info)
    # dtype_df.to_csv('dtype_info.csv', index=False)  

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df
    sorted_df = combined_df.sort_values(by=['STARTTIME', 'STOPTIME'], ascending=True)
    sorted_df.to_csv(f'{out_csv_name}.csv', index=False)

# load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
train_percentage = config['train_percentage']

# load the csv files
data_folder = '/Users/wei/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/24Sp/ENG573-SIEMENS/Data/DAM'
all_files = os.listdir(data_folder)
csv_files = sorted([file for file in all_files if file.endswith('.csv')])

# split dataset
split_index = int(len(csv_files) * (train_percentage / 100))
train_files = csv_files[:split_index]
valid_files = csv_files[split_index:]
print(train_files[0], train_files[-1])
print(valid_files[0], valid_files[-1])

combine_dataset(data_folder, train_files, out_csv_name="train_combined")
combine_dataset(data_folder, valid_files, out_csv_name="valid_combined")