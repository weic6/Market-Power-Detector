
_market="DAM"
_start_date="20230723"
_end_date="20230727"
data_folder="data/test"
zip_data_folder="${data_folder}/raw"

python3 scraper_CAISO_LMP.py \
    --market ${_market} \
    --start_date ${_start_date} \
    --end_date ${_end_date} \
    --save_to_path ${zip_data_folder}
