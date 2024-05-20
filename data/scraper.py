import requests
import time
import os
from tqdm import tqdm  # add progress bar
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate_dates, is_in_dst, get_next_date


def download_LMP_within_date_range(market, start_date, end_date, save_to_path):
    """
    Downloads LMP data from CAISO OASIS in market for a specified date range.

    The function generates dates between the start and end dates, checks if date is in Daylight Saving Time, then constructs the appropriate API URL for each date, and downloads the data in ZIP format.

    Parameters:
    market (str): DAM OR RUC
    start_date (str): The start date in 'YYYYMMDD' format.
    end_date (str): The end date in 'YYYYMMDD' format.
    save_to_path (str): The file path where the downloaded ZIP files will be saved.

    Returns:
    None: No return value.
    """
    dates = generate_dates(start_date, end_date)
    if len(dates) == 0:
        print("Error: start date is later than the end date.")
        return

    fail_downloads = []  # record failed downloads
    with tqdm(total=len(dates)) as p_bar:  # progress bar
        for date in dates:
            dst = is_in_dst(date)

            # get next date
            next_date = get_next_date(date)
            dst_next_date = is_in_dst(next_date)

            opentime = (
                "T07" if dst else "T08"
            )  # During the DST timeframe, use T07 instead of T08.
            opentime_next = "T07" if dst_next_date else "T08"
            # example DAM url:
            # http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=PRC_LMP&version=12&startdatetime=20231122T08:00-0000&enddatetime=20231123T08:00-0000&market_run_id=DAM&grp_type=ALL
            # http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=PRC_LMP&version=12&startdatetime=20240303T08:00-0000&enddatetime=20240304T08:00-0000&market_run_id=DAM&grp_type=ALL
            # example RUC url
            # 'http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=PRC_LMP&version=12&startdatetime=20231106T08:00-0000&enddatetime=20231107T08:00-0000&market_run_id=RUC&grp_type=ALL'
            api_url = f"http://oasis.caiso.com/oasisapi/SingleZip?resultformat=6&queryname=PRC_LMP&version=12&startdatetime={date}{opentime}:00-0000&enddatetime={next_date}{opentime_next}:00-0000&market_run_id={market}&grp_type=ALL"

            try:
                p_bar.set_description(
                    f"Downloading: {date}_{date}_{market}_LMP_GRP_N_N_v12_csv.zip"
                )  # example file: 20230312_20230312_DAM_LMP_GRP_N_N_v12_csv.zip
                response = requests.get(api_url)

                if response.status_code == 200:
                    # Save file
                    file_path = os.path.join(
                        save_to_path, f"{date}_{date}_{market}_LMP_GRP_N_N_v12_csv.zip"
                    )  # data will be stored in a ZIP file
                    with open(file_path, "wb") as file:
                        file.write(response.content)

                else:
                    print(
                        f"Warning: Unable to download file for {date}. Status code: {response.status_code}"
                    )
                    fail_downloads.append(date)

                p_bar.update(1)
            except requests.exceptions.ConnectionError:  # handle connection errors
                print(
                    f"ConnectionError for downloading data for date: {date}. Retrying in 10 seconds..."
                )
                time.sleep(10)  # Wait a bit longer before retrying
                continue

            # Wait for 5 seconds before next request
            time.sleep(5)

    if len(fail_downloads) == 0:
        print(f"Successfully downloaded data from {start_date} to {end_date}")
    else:
        print(f"Fail to download data in following dates:")
        for fail_download in fail_downloads:
            print(fail_download)


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
    if len(dates) == 0:
        print("Error: start date is later than the end date.")
        return

    fail_downloads = []  # record failed downloads
    if not os.path.exists(save_to_path):
        os.makedirs(save_to_path)
    with tqdm(total=len(dates)) as p_bar:  # progress bar
        for date in dates:
            dst = is_in_dst(date)
            opentime = (
                "T07" if dst else "T08"
            )  # During the DST timeframe, use T07 instead of T08.
            api_url = f"http://oasis.caiso.com/oasisapi/GroupZip?resultformat=6&version=3&groupid=PUB_{market}_GRP&startdatetime={date}{opentime}:00-0000"

            try:
                p_bar.set_description(
                    f"Downloading: {date}_{date}_PUB_{market}_GRP_N_N_v3_csv.zip"
                )
                response = requests.get(api_url)

                if response.status_code == 200:
                    # Save file
                    file_path = os.path.join(
                        save_to_path, f"{date}_{date}_PUB_{market}_GRP_N_N_v3_csv.zip"
                    )  # data will be stored in a ZIP file
                    with open(file_path, "wb") as file:
                        file.write(response.content)

                else:
                    print(
                        f"Warning: Unable to download file for {date}. Status code: {response.status_code}"
                    )
                    fail_downloads.append(date)

                p_bar.update(1)
            except requests.exceptions.ConnectionError:  # handle connection errors
                print(
                    f"ConnectionError for downloading data for date: {date}. Retrying in 10 seconds..."
                )
                time.sleep(10)  # Wait a bit longer before retrying
                continue

            # Wait for 5 seconds before next request
            time.sleep(5)

    if len(fail_downloads) == 0:
        print(f"Successfully downloaded data from {start_date} to {end_date}")
    else:
        print(f"Fail to download data in following dates:")
        for fail_download in fail_downloads:
            print(fail_download)


def main():
    parser = argparse.ArgumentParser(
        description="Download PUB_BID data within a specified date range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="either LMP or PUB_BID",
    )

    parser.add_argument(
        "--market", type=str, required=True, help="Market type, either DAM or RTM"
    )

    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help='The start date in "YYYYMMDD" format.',
    )
    parser.add_argument(
        "--end_date", type=str, required=True, help='The end date in "YYYYMMDD" format.'
    )
    parser.add_argument(
        "--save_to_path",
        type=str,
        required=True,
        help="The output directory for the downloaded ZIP files.",
    )

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_to_path):
        os.makedirs(args.save_to_path)

    if args.data == "LMP":
        download_LMP_within_date_range(
            args.market, args.start_date, args.end_date, args.save_to_path
        )

    elif args.data == "PUB_BID":
        download_public_bid_within_date_range(
            args.market, args.start_date, args.end_date, args.save_to_path
        )


if __name__ == "__main__":
    sys.exit(main())
