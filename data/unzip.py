import os
import sys
import zipfile
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
