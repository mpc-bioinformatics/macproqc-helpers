#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
import json
import h5py
import pandas as pd

from mzqc import MZQCFile as qc


def argparse_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-hdf5", help="The hdf5 file created by McQuaC", required=True)
    parser.add_argument("-mzqc_out", help="Output folder for the generated mzQC file", required=True)
    return parser.parse_args()


# defines for simple table metrics the names of the columns in a dict of the struct metric_acc -> {col1_acc, col2_acc}
simple_2d_metrics = {
    "MS:4000063": ["MS:1000041", "UO:0000191"],     # "MS2 known precursor charges fractions"
    "MS:4000180": ["MS:1003044", "NCIT:C150827"],   # "table of missed cleavage counts"
}


def hdf5_entry_to_mzqc_entry(hdf5_file: h5py.File, key: str) -> qc.QualityMetric:
    # split the key to get the basic accession and name
    split_key = key.split("|")
        
    metric_accession = split_key[0]
    # possible attributes: 'qc_description', 'qc_name', 'qc_short_name', 'unit_accession', 'unit_name'
    metric_name = hdf5_file[key].attrs.get("qc_name", split_key[1])

    metric_unit = None
    unit_accession = hdf5_file[key].attrs.get("unit_accession")
    if (unit_accession is not None) and len(unit_accession) > 0:
        unit_name = hdf5_file[key].attrs.get("unit_name")
        metric_unit = {"unit_accession": unit_accession, "unit_name": unit_name}

    if type(hdf5_file[key]) == h5py.Dataset:
        if hdf5_file[key].shape[0] == 1:
            metric_value = hdf5_file[key][0]
        else:
            metric_value = list(hdf5_file[key])
    elif type(hdf5_file[key]) == h5py.Group:

        # check, if all column names start with "MS:" -> these are the column names of teh table then
        key_list = list(hdf5_file[key].keys())

        if all(str(key_list).startswith("MS:")):
            metric_value = table_from_hdf5_group(hdf5_file[key])
        elif metric_accession in simple_2d_metrics.keys():
            # For simple 2D metrics, we need to convert the group to a 2D table
            metric_value = simple_2d_table_from_hdf5(hdf5_file[key], simple_2d_metrics[metric_accession])
        else:
            metric_value = None
    
    qm = qc.QualityMetric(accession=metric_accession, name=metric_name, value=metric_value, unit=metric_unit)

    print(f"{key}")
    print(f"\t{metric_accession}\t{metric_name}\t{metric_value}")
    print(f"\t{qm}")
    
    return qm


def table_from_hdf5_group(group: h5py.Group) -> Dict[str, List[Any]]:
    column_names = list(group.keys())
    return {k: group[k][0] for k in column_names}


def simple_2d_table_from_hdf5(group: h5py.Group, column_names: List[str]) -> Dict[str, List[Any]]:
    key_list = list(group.keys())
    value_list = [group[k][0] for k in group.keys()]
    if column_names is None or len(column_names) < 2:
        column_names = ["key_table", "value_table"]
    return {column_names[0]: key_list, column_names[1]: value_list}


if __name__ == "__main__":
    args = argparse_setup()

    print(f"args: {args.hdf5} {args.mzqc_out}")

    # TODO: add these as param from the original file, or better add into the hdf5 file!
    input_filename = "EXII07156std.raw"
    fileFormat=qc.CvParameter(accession="MS:1000563", name="Thermo RAW format")
    comet_version = "v2024.01.0"

    # extract this from the given hdf5 (LOCAL:timestamp)
    input_file_completion_time = "1980-10-03-T10:18:27Z"

    input_file_raw = qc.InputFile(name=input_filename,
            location=None,
            fileFormat=fileFormat, 
            fileProperties=[qc.CvParameter(accession="MS:1000747", 
                                            name="completion time", 
                                            value=input_file_completion_time),
                            # here we could add more information, if we had them
                            ])

    # we could add information about our search engine
    comet_mzqc = qc.AnalysisSoftware(accession="MS:1002251", 
                                    name="Comet",
                                    description="Comet open-source sequence search engine developed at the University of Washington.",
                                    version=comet_version, 
                                    uri="https://uwpr.github.io/Comet/")

    # TODO: and should add information about the feature detection as well
    #anso_nb = qc.AnalysisSoftware(version="0.1.2.3", uri="file:///mylocal/jupyter/host")

    # TODO: and about MCQuaC itself
    #anso_nb = qc.AnalysisSoftware(version="0.1.2.3", uri="file:///mylocal/jupyter/host")

    quality_metrics = []
    with  h5py.File(args.hdf5, "r") as hdf5_file:

        # go through the entries in the hdf5 file and get all the metrics
        for key in hdf5_file.keys():
            qm = hdf5_entry_to_mzqc_entry(hdf5_file, key)

            # only use "valid" metrics for now
            if qm.accession.startswith("MS"):
                quality_metrics.append(qm)

    # replace values:
    # 4000054 -> 4000183
    # 4000055 -> 4000184
    # 4000056 -> 4000185
    # 4000057 -> 4000186
    # 4000058 -> 4000187

    # typo in the hdf5 file
    # "MS1 TIC quartile ratios" should be MS:4000058
    # MS:4000180 ! table of missed cleavage counts: last column name should only be a number and interpreted as "and more"

    meta = qc.MetaDataParameters(inputFiles=[input_file_raw],analysisSoftware=[comet_mzqc])
    rq = qc.RunQuality(metadata=meta, qualityMetrics=quality_metrics)
    cv_ms = qc.ControlledVocabulary(name="Proteomics Standards Initiative Mass Spectrometry Ontology",version="4.1.197", uri="https://github.com/HUPO-PSI/psi-ms-CV/blob/master/psi-ms.obo")


    mzqc = qc.MzQcFile(version="1.0.0", creationDate=datetime.now().isoformat(), runQualities=[rq], setQualities=[], controlledVocabularies=[cv_ms])

    with open(args.mzqc_out, "w") as mzqc_file:
        mzqc_file.write(json.dumps(json.loads(qc.JsonSerialisable.to_json(mzqc)), indent=2))

