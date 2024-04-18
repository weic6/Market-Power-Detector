#!/bin/bash

_market="DAM"
_start_date="20230313"
_end_date="20230722"
data_folder="data/PUB_BID"
zip_data_folder="${data_folder}/raw"
unzip_data_folder="${data_folder}/unzip"
# Run the scraper to get zip data
python3 scraper_CAISO_PUB-BID.py \
    --market ${_market} \
    --start_date ${_start_date} \
    --end_date ${_end_date} \
    --save_to_path ${zip_data_folder}

# Unzip the data
python3 -c "from utils import unzip_files; unzip_files('${zip_data_folder}', '${unzip_data_folder}')"
