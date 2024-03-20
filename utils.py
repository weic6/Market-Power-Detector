import os
import zipfile
from datetime import datetime, timedelta
import pytz  # set time zone
from tqdm import tqdm  # add progress bar


def unzip_files(input_dir, output_dir):
    """
    unzip files inside input_dir to output_dir.
    """

    # list all the ZIP files
    zip_files = [file for file in os.listdir(input_dir) if file.endswith(".zip")]

    with tqdm(total=len(zip_files)) as p_bar:  # init the progress bar
        for file in zip_files:
            filepath = os.path.join(input_dir, file)
            p_bar.set_description(f"Extracting: {file}")
            with zipfile.ZipFile(filepath, "r") as zip_ref:

                zip_ref.extractall(output_dir)  # extract
            p_bar.update(1)  # update the progress bar
    print(f"Successfully extract files into {output_dir}")


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
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    # Generate dates
    current_date = start_date
    date_strings = []
    while current_date <= end_date:
        date_strings.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    return date_strings


def get_next_date(date_str):
    current_date = datetime.strptime(date_str, "%Y%m%d")
    next_date = current_date + timedelta(days=1)
    next_date_str = next_date.strftime("%Y%m%d")
    return next_date_str


def is_in_dst(date_str):  # return true if date_str is in DST timeframe
    """
    Determine if a given date in California (Pacific Time Zone) is within Daylight Saving Time (DST).

    Parameters:
    date_str (str): A string representing a date in the format "YYYYMMDD".

    Returns:
    True if the date is within DST, False otherwise.
    """
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    california_tz = pytz.timezone("America/Los_Angeles")  # set time zone as California
    date_in_california = california_tz.localize(
        date_obj
    )  # convert the datetime object to California timezone
    return date_in_california.dst() != timedelta(0)
