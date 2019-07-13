#! /usr/bin/env python
import os
import subprocess
import argparse
import time

DEBUG = True
WRITE_LOGS = False

# TODO(Xiang): your executable python script should go here. This is the
# experiments code that will be ran on condor. The python script should write
# the results to file in a way that they can be later read once all experiments
# are finished.
#
# Note that this script assumes the python script can take arguments passed to
# it. At a minimum, this code assumed the script takes an argument for a result
# file and a seed for the random number generator.
EXECUTABLE = '/u/jphanna/path/to/executable.py'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--result_directory',
                    help='Directory to write result files to.', default=None)
parser.add_argument('--num_trials', help='Number of trials to run.',
                    type=int, default=0)


# Example arguments that can be passed to all launched jobs.
args = {'--arg1': 0,
        '--arg2': ''}


def submitToCondor(seed, out_file, exp_args):

    arguments = '--result_file=%s --seed=%d' % (out_file, seed)

    for arg in args:
        arguments += ' %s=%s' % (arg, args[arg])

    for arg in exp_args:
        arguments += ' %s=%s' % (arg, exp_args[arg])

    submitFile = 'Executable = %s\nError = %s.err\n' % (EXECUTABLE, out_file)
    outputfile = errorfile = '/dev/null'
    if WRITE_LOGS:
        outputfile = '%s.out' % out_file
        errorfile = '%s.err' % out_file
    submitFile += 'Input = /dev/null\nOutput = %s\n' % outputfile
    submitFile += 'Log = %s\narguments = %s\n' % (errorfile, arguments)
    submitFile += 'requirements = InMastodon\n'
    submitFile += '+Group = "GRAD"\n+Project = "AI_ROBOTICS"\n'
    submitFile += '+ProjectDescription = "Behavior policy search"\n'
    submitFile += 'Queue'

    if DEBUG:
        print(WRAPPER, arguments)
        time.sleep(0.01)
    else:
        proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE)
        proc.stdin.write(submitFile.encode())
        proc.stdin.close()
        time.sleep(0.05)


def main():

    ct = 0
    args = parser.parse_args()
    directory = args.result_directory

    # Example argument: sometimes it is convenient to launch experiments with
    # some conditions changed. So if your executable takes an argument "--arg_name"
    # then you can loop over several different values for this argument to test
    # different conditions.
    params = [0, 5, 8]
    arg = '--arg_name'

    for param in params:

        for index in range(args.num_trials):
            base = 'experiment'
            hidden_layers, hidden_units = network

            base += '_%d' % param

            filename = os.path.join(directory, '%s_%d' % (base, index))
            if os.path.exists(filename):
                continue
            submitToCondor(index, filename, {arg: param})
            ct += 1

    print('%d jobs submitted to cluster' % ct)


if __name__ == "__main__":
    main()
