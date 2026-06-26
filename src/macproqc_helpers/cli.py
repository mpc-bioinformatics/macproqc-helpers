
"""
MacProQC Command Line Interface

Main entry point for the MacProQC module with nested argparse CLI.
Combines multiple mass spectrometry quality control analysis scripts into a unified interface.
"""

import argparse
import sys

from macproqc_helpers.helpers import (
    adjust_comet_params,
    collect_metrics_from_bruker,
    collect_metrics_from_featurexml,
    collect_metrics_from_mzml,
    collect_metrics_from_pia_output,
    collect_metrics_from_thermo,
    collect_spikein_metrics,
    combine_hdf5_files,
    convert_mztab_to_idxml,
    extract_xic_bruker,
    hdf5_to_mzqc,
    visualization,
)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="MacProQC - Mass Spectrometry Quality Control Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  macproqc adjust-comet-params -comet_params in.params -params_out out.params
  macproqc combine-hdf5 -hdf_out_name combined.hdf5 file1.hdf5 file2.hdf5
  macproqc extract-mzml -mzml sample.mzml -out_hdf5 output.hdf5 -base_peak_tic_up_to 105 -filter_threshold 0.00001 -report_up_to_charge 5
  macproqc hdf5-to-mzqc -hdf5 metrics.hdf5 -mzqc_out output.mzqc
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="Commands",
        description="Available analysis commands",
        dest="command",
        help="Command to execute"
    )
    
    # Register all command parsers
    adjust_comet_params.argparse_setup(subparsers)
    collect_metrics_from_bruker.argparse_setup(subparsers)
    collect_metrics_from_featurexml.argparse_setup(subparsers)
    collect_metrics_from_mzml.argparse_setup(subparsers)
    collect_metrics_from_pia_output.argparse_setup(subparsers)
    collect_metrics_from_thermo.argparse_setup(subparsers)
    collect_spikein_metrics.argparse_setup(subparsers)
    combine_hdf5_files.argparse_setup(subparsers)
    convert_mztab_to_idxml.argparse_setup(subparsers)
    extract_xic_bruker.argparse_setup(subparsers)
    hdf5_to_mzqc.argparse_setup(subparsers)
    visualization.argparse_setup(subparsers)

    
    return parser


def main(argv=None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # If no command is specified, print help
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    try:
        args.func(args)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
