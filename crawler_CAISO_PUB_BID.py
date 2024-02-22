import datetime
import pytz # set time zone
import requests
import time
import os
from tqdm import tqdm  # add progress bar

def is_in_dst(date_str): #return true if date_str is in DST timeframe
    """
    Determine if a given date in California (Pacific Time Zone) is within Daylight Saving Time (DST).

    Parameters:
    date_str (str): A string representing a date in the format "YYYYMMDD".

    Returns:
    True if the date is within DST, False otherwise.
    """
    date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
    california_tz = pytz.timezone('America/Los_Angeles') #set time zone as California 
    date_in_california = california_tz.localize(date_obj) #convert the datetime object to California timezone
    return date_in_california.dst() != datetime.timedelta(0)

def generate_dates(start_date_str, end_date_str):
    """
    Generates a list of date strings for each day between a specified and valid start and end date, inclusive.

    Parameters:
    start_date_str (str): The start date in 'YYYYMMDD' format.
    end_date_str (str): The end date in 'YYYYMMDD' format.

    Returns:
    date_strings: A list of date strings, each in 'YYYYMMDD' format, for each day from the start date to the end date, inclusive.
    """
    
    # Convert strings to datetime objects
    start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d")
    
    # Generate dates
    current_date = start_date
    date_strings = []
    while current_date <= end_date:
        date_strings.append(current_date.strftime("%Y%m%d"))
        current_date += datetime.timedelta(days=1)
    
    return date_strings


def download_public_bid_within_date_range(market, start_date, end_date, save_to_path):
    """
    Downloads public bid data from CAISO OASIS in market for a specified date range.
    
    The function generates dates between the start and end dates, checks if date is in Daylight Saving Time, then constructs the appropriate API URL for each date, and downloads the data in ZIP format.

    Parameters:
    market (str): DAM OR RTM
    start_date (str): The start date in 'YYYYMMDD' format.
    end_date (str): The end date in 'YYYYMMDD' format.
    save_to_path (str): The file path where the downloaded ZIP files will be saved.

    Returns:
    None: No return value.
    """
    dates = generate_dates(start_date, end_date)
    if (len(dates)==0):
        print("Error: start date is later than the end date.")
        return
    
    fail_downloads = []  # record failed downloads
    if not os.path.exists(save_to_path):
        os.makedirs(save_to_path)
    with tqdm(total=len(dates)) as p_bar: #progress bar
        for date in dates:
            dst = is_in_dst(date)
            opentime = "T07" if dst else "T08" #During the DST timeframe, use T07 instead of T08.
            api_url = f"http://oasis.caiso.com/oasisapi/GroupZip?resultformat=6&version=3&groupid=PUB_{market}_GRP&startdatetime={date}{opentime}:00-0000"
            
            try:
                p_bar.set_description(f"Downloading: {date}_{date}_PUB_{market}_GRP_N_N_v3_csv.zip")
                response = requests.get(api_url)
                
                if response.status_code == 200:
                    # Save file
                    file_path = os.path.join(save_to_path, f"{date}_{date}_PUB_{market}_GRP_N_N_v3_csv.zip") #data will be stored in a ZIP file
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
            
                else:
                    print(f"Warning: Unable to download file for {date}. Status code: {response.status_code}")
                    fail_downloads.append(date)
                    
                p_bar.update(1)    
            except requests.exceptions.ConnectionError:# handle connection errors
                print(f"ConnectionError for downloading data for date: {date}. Retrying in 10 seconds...")
                time.sleep(10)  # Wait a bit longer before retrying
                continue     
            
            # Wait for 5 seconds before next request
            time.sleep(5)
        
        
    if (len(fail_downloads)==0):
        print(f"Successfully downloaded data from {start_date} to {end_date}")
    else:
        print(f"Fail to download data in following dates:")
        for fail_download in fail_downloads:
            print(fail_download)
            

def main():
    start_date = "20230802" # "YYYYMMDD"
    end_date = "20231123" # "YYYYMMDD"
    save_to_path = "Data"
    market = "RTM" # or "DAM"
    download_public_bid_within_date_range(market, start_date, end_date, save_to_path)

if __name__ == '__main__':
    main()



