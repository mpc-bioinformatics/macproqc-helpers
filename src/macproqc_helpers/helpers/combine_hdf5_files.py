import argparse
import os

import h5py


def argparse_setup(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "combine-hdf5",
        description="Combine multiple HDF5 files into one. The files should be in the same format (e.g. all mzQC or all macproqc output).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-hdf_out_name", help="The output of the combined HDF5 files")
    parser.add_argument(
        "files", type=check_if_file_exists, nargs="+", help="All files which should be combined"
    )
    parser.set_defaults(func=combine)


def check_if_file_exists(s: str):
    """checks if a file exists. If not: raise Exception"""
    if os.path.isfile(s):
        return s
    else:
        raise Exception(f"File '{s}' does not exists")


def combine(args: argparse.Namespace) -> None:
    with h5py.File(args.hdf_out_name, "w") as out_h5:
        # sort input files to make finished HDF5 files hash-value comparable
        for hdf in sorted(args.files, key=os.path.basename):
            with h5py.File(hdf, "r") as in_h5:
                for obj in in_h5:
                    in_h5.copy(obj, out_h5)
