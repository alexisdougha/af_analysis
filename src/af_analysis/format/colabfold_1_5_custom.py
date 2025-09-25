#!/usr/bin/env python3
# coding: utf-8

import os
import logging
from tqdm.auto import tqdm
import pandas as pd
import json

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


def read_dir(directory):
    """Extract pdb list from a directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing the pdb files.

    Returns
    -------
    log_pd : pandas.DataFrame
        Dataframe containing the information extracted from the directory.

    """

    logger.info(f"Reading {directory}")

    log_dict_list = []

    pdb_files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdb")
    ]

    for pdb_file in pdb_files:
        filename = pdb_file.split("/")[-1]
        json_file = os.path.join(
            directory,
            filename.replace(".pdb", ".json").replace("_unrelaxed_", "_scores_"),
        )

        with open(json_file, "r") as f_in:
            json_dict = json.load(f_in)

        info_dict = {
            "pdb": pdb_file,
            "query": filename.split("_unrelaxed_")[0],
            "model": int(filename.split("model_")[1].split("_")[0]),
            "data_file": json_file,
        }

        info_dict.update(json_dict)
        log_dict_list.append(info_dict)

    log_pd = pd.DataFrame(log_dict_list)

    print(f"Found {len(log_pd)} predictions in {directory}")
    print(f"Found {log_pd.columns}")

    # Update column names
    log_pd = log_pd.rename(
        columns={
            # https://github.com/jwohlwend/boltz/issues/73
            # "confidence_score": "ranking_confidence",
            "ptm": "pTM",
            "iptm": "ipTM",
        }
    )
    log_pd["ranking_confidence"] = 0.2 * log_pd["pTM"] + 0.8 * log_pd["ipTM"]

    # To ensure that tests are consistent across different systems
    # we sort the dataframe by pdb
    log_pd = log_pd.sort_values(by=["pdb"]).reset_index(drop=True)
    return log_pd
