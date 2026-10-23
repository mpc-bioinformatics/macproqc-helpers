#!/usr/bin/env python

import argparse
import csv
import json
from statistics import mean
from typing import Any, Dict, List

from pyteomics import proforma


def argparse_setup(subparsers: argparse._SubParsersAction):
    """
    Set up the argument parser for the create_thermorawfileparser_xic_config command.
    """
    parser = subparsers.add_parser(
        "create_thermorawfileparser_xic_config",
        help="Create a JSON file to be used with the ThermoRawFileParser or alphatims XIC extraction",
    )

    parser.add_argument("-icsv", required=True, help="The CSV-table defining the spike ins")
    parser.add_argument(
        "-iidents",
        required=False,
        default=None,
        action="append",
        help="Optional identification results in mzTAB format. If given, the RTs of identified spike ins are fine-tuned. Can be given multiple times to use several files (e.g. for unlabelled and labelled searches).",
    )
    parser.add_argument(
        "-ojson",
        required=True,
        help="The output file in JSON to be used with the ThermoRawFileParser XIC extraction (or compatible)",
    )
    parser.add_argument(
        "-oidentifications",
        required=False,
        default=None,
        help="Output mappings of the sequences to number of identifications, if a sequence is not listed, there was no (valid) ID in the files provided by -iidents (or the file was not given at all)",
    )
    parser.set_defaults(func=create)


def create(args: argparse.Namespace):
    """
    Create a JSON file to be used with the ThermoRawFileParser and alphatims XIC extractions
    """

    seq_to_data: Dict[str, Dict[str, Any]] = dict()

    with open(args.icsv, encoding="utf-8") as in_csv_file:
        # Read spike-ins CSV and get header indices
        csv_in = csv.reader(in_csv_file)
        header = next(csv_in)
        name_idx = header.index("name")
        seq_idx = header.index("sequence")
        mz_idx = header.index("mz")
        rt_idx = header.index("RT")
        mz_tol_idx = header.index("mz-tol")
        mz_tol_unit_idx = header.index("mz-tol-unit")
        rt_tol_idx = header.index("rt-tol")

        # Generate the two output structures
        for row in csv_in:
            # go line-wise through the spike-ins CSV

            rt_tol = float(row[rt_tol_idx]) / 60

            # transform the proforma sequence to a plain sequence (only amino acids, no modifications) to be able to check for identifications, which are only annotated with plain sequences
            proforma_sequence = proforma.parse(str(row[seq_idx]))
            plain_sequence = "".join(aa for aa, mods in proforma_sequence[0])

            # map from the sequence to all other data (basically our json)
            seq_to_data[plain_sequence] = {
                "comment": str(row[name_idx]),
                "mz": float(row[mz_idx]),
                "tolerance": float(row[mz_tol_idx]),
                "tolerance_unit": str(row[mz_tol_unit_idx]),
                "rt_start": max(0, (float(row[rt_idx]) / 60.0) - rt_tol)
                if (float(row[rt_idx]) >= 0)
                else -1,
                "rt_end": (float(row[rt_idx]) / 60.0) + rt_tol if (float(row[rt_idx]) >= 0) else -1,
                "rt_tol": rt_tol,
            }

    found_rts: Dict[str, List[float]] = dict()

    # only tune identifications if any ID data is provided
    if args.iidents:
        for ident_file in args.iidents:
            with open(ident_file, encoding="utf-8") as in_ident_file:
                # header information is reset for each mzTab file
                seq_idx = 0
                rt_idx = 0
                score_label = ""
                score_idx = 0
                for line in in_ident_file:
                    # iterate through mzTab

                    # Get Header Line
                    if line.startswith("MTD"):
                        cols = line.split("\t")
                        if cols[1].startswith("psm_search_engine_score") and (
                            "MS:1002257" in cols[2]
                        ):
                            # this is the "Comet:expectation value" score, remove the "psm_" prefix
                            score_label = cols[1][4:]
                    elif line.startswith("PSH"):
                        cols = line.split("\t")
                        seq_idx = cols.index("sequence")
                        rt_idx = cols.index("retention_time")
                        if len(score_label) > 0:
                            score_idx = cols.index(score_label)

                    # Check if we found an psms which actually fits to the SpikeIns and put it into the found RTs
                    if line.startswith("PSM"):
                        cols = line.split("\t")
                        if cols[seq_idx] in seq_to_data:
                            if score_idx > 1 and float(cols[score_idx]) < 0.01:
                                # either cannot filter for score or it is below 0.01
                                if cols[seq_idx] not in found_rts:
                                    found_rts[cols[seq_idx]] = []

                                found_rts[cols[seq_idx]].append(float(cols[rt_idx]))

        # adjust the RTs of found identifications (over all given files)
        for seq, vals in found_rts.items():
            # take the mean of the IDs' RT as new center
            new_rt = mean(set(vals)) / 60

            # and as new tolerance the maximum of the given or the half diff between ID RTs (-> including all IDs)
            new_tol = max((max(vals) - min(vals)) / 120, seq_to_data[seq]["rt_tol"])

            seq_to_data[seq]["rt_start"] = new_rt - new_tol
            seq_to_data[seq]["rt_end"] = new_rt + new_tol

    for _seq, data in seq_to_data.items():
        # if no ID was found and none given (rt_start < 0), measure from 0 to 9999
        if data["rt_start"] < 0:
            data["rt_start"] = 0
            data["rt_end"] = 9999
        else:
            data["rt_start"] = max(0, data["rt_start"])

        del data["rt_tol"]

    # Write Output JSON-File
    with open(args.ojson, "w", encoding="utf-8") as o_json:
        o_json.write(json.dumps([val for val in seq_to_data.values()], indent=4))

    # Writes out the fine-tuned identifications file (can be empty, if none were found)
    if args.oidentifications:
        with open(args.oidentifications, "w", encoding="utf-8") as output_identifications:
            for seq, vals in found_rts.items():
                nr_ids = len(set(vals))
                output_identifications.write(seq + "," + str(nr_ids) + "\n")
