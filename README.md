# MacProQC-Helpers Module

A self-contained Python module that combines multiple mass spectrometry quality control analysis scripts into a unified CLI interface using argparse for macproqc pipeline.

## Structure

```
macproqc/
├── __init__.py              # Package initialization
├── __main__.py              # Entry point for `python -m macproqc`
├── cli.py                   # Main CLI with nested argparse subcommands
├── commands/                # Individual command modules
│   ├── __init__.py
│   ├── ... # actual logic and their partial CLI
└── utils/                   # Shared utilities
    ├── __init__.py
    └── hdf5.py              # HDF5 helper functions
```

## Installation

From the module directory:

```bash
pip install -e .
```

Or for development with additional tools:

```bash
pip install -e ".[dev]"
```

Or via docker

```bash
docker build -t mpc/macproqc-helpers:dev .
```

## Usage

### As a command-line tool:

```bash
macproqc-helpers --help
macproqc-helpers adjust-comet-params --help
macproqc-helpers combine-hdf5 --help
```

### Run from Python directly:

```bash
python -m macproqc_helpers --help
python -m macproqc_helpers combine-hdf5 -hdf_out_name output.hdf5 file1.hdf5 file2.hdf5
```
