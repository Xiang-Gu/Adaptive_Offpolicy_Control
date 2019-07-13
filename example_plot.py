#! /usr/bin/env python
import os
import argparse


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--result_directory',
                    help='Directory to write result files to.', default=None)


def load_results(result_directory):
    """Loops over result files in a directory and returns data."""

    data = []

    for filename in os.listdir(result_directory):
        # Loop over all files in directory to read results from file.
        # Note that if your result directory contains log files then there
        # should be code here to skip over these. For example,
        if filename.endswith('.err'):
            continue
        full_path = os.path.join(result_directory, filename)
        with open(full_path, 'r') as f:
            content = f.read()
            # Read results from file and add to return data

    return data


def main():

    args = parser.parse_args()
    
    directory = args.result_directory

    # 1. Load results from file
    results = load_results(directory)

    # 2. Write code to plot results.
    # This code will depend on how the results are returned.


if __name__ == '__main__':
    main()