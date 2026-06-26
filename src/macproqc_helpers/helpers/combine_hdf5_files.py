import argparse
from datetime import date
import os

import h5py

def argparse_setup(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "combine-hdf5",
        description="Combine multiple HDF5 files into one. The files should be in the same format (e.g. all mzQC or all McQuaC output).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-hdf_out_name", help="The output of the combined HDF5 files")
    parser.add_argument("-put_under_subdataset", default=False, action="store_true", help="Flag if data should be added under a subdataset")
    parser.add_argument("-write_metadata", default=False, action="store_true", help="Flag if the node 'METADATA' should be written")
    parser.add_argument(
        "files", type=check_if_file_exists, nargs="+",
        help="All files which should be combined"
    )
    parser.set_defaults(func=combine)


def check_if_file_exists(s: str):
    """ checks if a file exists. If not: raise Exception """
    if os.path.isfile(s):
        return s
    else:
        raise Exception("File '{}' does not exists".format(s))


def write_metadata(f):
    f.create_dataset("METADATA", shape=(1,), dtype="int64", compression="gzip")
    f["METADATA"].attrs["mzQC_format_version"] = "1.2"
    f["METADATA"].attrs["mzQC_data_version"] = "4.1.18"
    f["METADATA"].attrs["mzQC_URL"] = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/321d7731f683557c5aecf6e5f7fefe049da0ecbf/psi-ms.obo"
    f["METADATA"].attrs["creation_date"] = str(date.today)


def combine(args: argparse.Namespace) -> None:

    if not args.put_under_subdataset:
        # Write everything under "/"
        with h5py.File(args.hdf_out_name, 'w') as out_h5:
            for hdf in args.files:
                with h5py.File(hdf, 'r') as in_h5:
                    for obj in in_h5:
                        in_h5.copy(obj, out_h5)
            if args.write_metadata:
                write_metadata(out_h5)
    else: 
        # Write everything under "/filename/"
        with h5py.File(args.hdf_out_name, 'w') as out_h5:
            for hdf in args.files:
                filename = os.path.basename(hdf).split(".", 1)[0]
                out_h5.create_group("/" + filename)
                with h5py.File(hdf, 'r') as in_h5:
                    for obj in in_h5:
                        in_h5.copy(obj, out_h5["/" + filename])
            if args.write_metadata:
                write_metadata(out_h5)
