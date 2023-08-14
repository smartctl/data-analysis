import logging
import argparse
import inspect
import pandas as pd
from pathlib import Path
from src.config import Config
from src.data_wrangling import DataWrangle

# TODO: Move extra functions to their own class
def get_filenames(srcpath: str, logger):
    """
    If a directory, returns a list of filenames with the .parquet extension
    If a filename, returns a list containing the filename if it exists
    Returns an empty list if not a file or directory.
    """
    fname = Path(srcpath)
    if fname.is_file():
        return [srcpath]
    elif fname.is_dir():
        return list(fname.rglob("*.parquet"))
    else:
        logger.error(f"[{inspect.stack()[0][3]}] {srcpath} is not a valid file or directory.")
        return []

def args_or_default(args, default):
    #if args != None:
    return args if args else default


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Only validate the mapping.")
    parser.add_argument("--mapping", help="dtypes mapping file to use. (default mapping/mapping.yaml)")
    parser.add_argument("--y-map", help="Mapping file for y-lables. (default mapping/labels-definitions.yaml)")
    parser.add_argument("--src", help="Path to source datasets (directory or file)")
    parser.add_argument("--dstfile", help="Path to write processed datasets (directory or file)")
    args = parser.parse_args()

    # Load the general mapping
    global_config = Config(
        config_file=args_or_default(args.mapping,"mapping/mapping.yaml"), 
        y_map_file=args_or_default(args.y_map,"mapping/labels-definition.yaml")
    )
    mapping = global_config.config
    logger = global_config.logger
    y_map = global_config.y_map

    src = args_or_default(args.src,"data/collection")
    dstfile = args_or_default(args.dstfile,"dtyped-data.parquet")

    # Create Data Wrangling instance
    data_wrangler=DataWrangle(mapping, y_map_set=y_map, dstdir="data/wrangle", logger=logger)

    file_names=get_filenames(src, logger)
    logger.info(f"Filenames {file_names}.")
    data_wrangler.load_and_combine_datasets(file_names)
    data_wrangler.write_dataset(dstfile)

    logger.info(f"Taks {data_wrangler} completed.")

if __name__ == "__main__":
    main()
