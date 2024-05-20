#!/bin/bash
'''run the script to prepare data to the database'''

_data="PUB_BID" #or "LMP"
_market="DAM"
_start_date="20230313"
_end_date="20230722"
data_folder="data/PUB_BID"

# Run the scraper to get zip data
python3 scraper.py \
    --data ${_data} \
    --market ${_market} \
    --start_date ${_start_date} \
    --end_date ${_end_date} \
    --save_to_path ${zip_data_folder}

# Unzip the data
zip_data_folder="${data_folder}/raw"
unzip_data_folder="${data_folder}/unzip"
python3 -c "from unzip import unzip_files; unzip_files('${zip_data_folder}', '${unzip_data_folder}')"

# populate database
python3 db.py
