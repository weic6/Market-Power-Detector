'''
This script divides all csv files into training and validating files. Combine all training and validating csv files into train_combined.csv and valid_combined.csv, respectively
'''
import pandas as pd
import os
import json
from glob import glob

def combine_PUB_BID_dataset(all_csv_files, out_csv_name):
    """
    Filters rows, removes specific columns and then combines CSV files into a new CSV file.

    Parameters:
    - all_csv_files (list): A list of filenames (str) of the CSV files to be processed.
    - out_csv_name (str): The name of the output CSV file where the combined dataset will be saved.

    Returns:
    A new CSV file which contains the combined dataset.
    """
    
    data_frames = [] 

    for file in all_csv_files:
        df = pd.read_csv(file, low_memory=False)
        filtered_df = df[(df['RESOURCE_TYPE'] == 'GENERATOR') & (df['MARKETPRODUCTTYPE'] == 'EN')] #consider generator and EN for now
        result_df = filtered_df.drop(columns=['STARTTIME_GMT', 'STOPTIME_GMT', 'STARTDATE', 'MARKET_RUN_ID', 'TIMEINTERVALSTART_GMT', 'TIMEINTERVALEND_GMT'])
        data_frames.append(result_df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    sorted_df = combined_df.sort_values(by=['STARTTIME', 'STOPTIME'], ascending=True)
    
    return sorted_df.to_csv(f'{out_csv_name}.csv', index=False)
    # dtype_df = [] 
    # for file in all_csv_files:
    #     df = pd.read_csv(file, low_memory=False)
    #     Collect the file path and data types information
    #     dtype_info.append({
    #         "file": file,
    #         "file_types": str(df.dtypes.to_dict())  # Convert the dtype Series to a dictionary and then to a string
    #     })
    # dtype_df = pd.DataFrame(dtype_info)
    # dtype_df.to_csv('dtype_info.csv', index=False)  

def combine_LMP_dataset(all_csv_files, out_csv_name):
    """
    Filters rows, removes specific columns and then combines CSV files into a new CSV file.

    Parameters:
    - all_csv_files (list): A list of filenames (str) of the CSV files to be processed.
    - out_csv_name (str): The name of the output CSV file where the combined dataset will be saved.

    Returns:
    A new CSV file which contains the combined dataset.
    """
    
    data_frames = [] 

    for file in all_csv_files:
        df = pd.read_csv(file, low_memory=False)
        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    sorted_df = combined_df.sort_values(by=['INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT'], ascending=True) ## TODO: match the date between LMP dataset and 
    
    return sorted_df.to_csv(f'{out_csv_name}.csv', index=False)

def main():
    # load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    train_percentage = config['train_percentage']

    # load the csv files
    pub_bid_data_folder = os.path.join(os.path.dirname(__file__), '../../Data/DAM/PUB_BID')
    pub_bid_csv_files = sorted(glob(os.path.join(pub_bid_data_folder, '*_PUB_BID_DAM_*.csv')))
    print(len(pub_bid_csv_files)) #87???
    
    # split dataset
    split_ratio = int(len(pub_bid_csv_files) * (train_percentage / 100))
    pub_bid_train_files = pub_bid_csv_files[:split_ratio]
    pub_bid_valid_files = pub_bid_csv_files[split_ratio:]
    # print(pub_bid_train_files[0], pub_bid_train_files[-1])
    # print(pub_bid_valid_files[0], pub_bid_valid_files[-1])
    
    output_pub_bid_train_csv_name = 'test1_pub_bid_train_combined'
    output_pub_bid_valid_csv_name = 'test1_pub_bid_valid_combined'

    # combine_PUB_BID_dataset(pub_bid_train_files, out_csv_name=output_pub_bid_train_csv_name)
    # combine_PUB_BID_dataset(pub_bid_valid_files, out_csv_name=output_pub_bid_valid_csv_name)
    
    
    
    # load LMP csv files
    lmp_data_folder = os.path.join(os.path.dirname(__file__), '../../Data/DAM/LMP')
    lmp_csv_files = sorted(glob(os.path.join(lmp_data_folder, '*_LMP_DAM_LMP_*.csv')))
    # print(lmp_csv_files)
    
    # split dataset
    lmp_train_files = lmp_csv_files[:split_ratio]
    lmp_valid_files = lmp_csv_files[split_ratio:]
    # print(lmp_train_files[0], lmp_train_files[-1])
    # print(lmp_valid_files[0], lmp_valid_files[-1])
    # print(len(lmp_csv_files)) #93???
    
    #process LMP
    output_lmp_train_csv_name = 'lmp_train_combined'
    output_lmp_valid_csv_name = 'lmp_valid_combined'
    combine_LMP_dataset(lmp_train_files, out_csv_name=output_lmp_train_csv_name)
    combine_LMP_dataset(lmp_valid_files, out_csv_name=output_lmp_valid_csv_name)
    
if __name__ == '__main__':
    main()
