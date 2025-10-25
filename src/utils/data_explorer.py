import pandas as pd
import os
from typing import List, Tuple, Union, Dict, Optional

def data_loader(path, filenames=Union[List[str], Tuple[str, ...]], verbose=True, **pandas_kwargs):
    """
    minor function to collect train, validation and test data with error check
    :param path: path to csv file
    :param filenames: set of filenames
    verbose: the parameter to continue the finding data, if not the progress to stop
    :return: train, validation and test dataset if any available, None otherwise
    The return data will be dictionary for pandas dataframes
    """
    loaded_data = {}
    file_info = []

    for filename in filenames:
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, **pandas_kwargs)
                keys = filename.replace(".csv", "_data")
                loaded_data[keys] = df
                file_info.append(f"✅ {filename}: {df.shape[0]:,} row * {df.shape[1]:,} columns")
            except Exception as e:
                if verbose:
                    print(f"❌ File {filename} not found, skipping...")
                continue
        else:
            if verbose:
                print(f"⚠️ No file {filepath} found")

    # summary the progress information
    if verbose and loaded_data:
        print("=" * 50)
        print("📁 DATA LOADER SUMMARY")
        print("=" * 50)
        for info in file_info:
            print(info)
        print(f"\n📊 Total files loaded: {len(loaded_data)}")
        print("=" * 50)

    # If file is available but no data is loaded
    if not loaded_data:
        if verbose:
            print("❌ No data files were loaded!")
        return None
    else: return loaded_data.copy()


if __name__ == "__main__":
    # try:
    #     files = os.listdir("../dataset/")
    #     print("✅ Files found:", files)
    # except FileNotFoundError:
    #     print("❌ Dataset folder not found")
    #     print("Current dir:", os.getcwd())

    data = data_loader("/mnt/DATA/10_AIO_VN/AIOVN_Main/Project 5.1_House Price Prediction/dataset/", filenames=["train.csv", "test.csv"])
    print(type(data["train_data"]))