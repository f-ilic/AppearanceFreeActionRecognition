#!/usr/bin/env python3

from collections import OrderedDict
import datetime
import os
import os.path
from os.path import join
import subprocess


def setup_outdir(name, base=None):
    """
    Setup and output directory.

    args:
        name        name of the output directory
        base        base directory (or file in base directory), it not passed,
                    the parent directory of the directory in which this file is
                    in is used (assuming that we are in a utils module which is
                    a directory lying in the project base folder)

    Best practice usage:

        setup_outdir("out", __file__)

    to get subfolder in directory of main script or

        setup_outdir("out", os.path.curdir)

    to get subfolder of current working directory.
    """

    if base is not None:
        if os.path.isfile(base):
            base = os.path.dirname(base)
        elif os.path.isdir(base):
            pass
        else:
            raise IOError("need to pass a base which is either a file or a" + \
                    "directory.")
    else:
        base = os.path.dirname(os.path.dirname(__file__))

    outdir = join(base, name)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return outdir


def setup_numbered_outdir(basename, base=None, title=None):
    """
    Setup and a numbered output directory for an experiment.

    args:
        name        name of the output directory in which numbered directory is
                    created
        base        base directory (or file in base directory), it not passed,
                    the parent directory of the directory in which this file is
                    in is used (assuming that we are in a utils module which is
                    a directory lying in the project base folder)
        title       additional string to append to directory name

    Best practice usage: see documentation of setup_outdir().
    """
    outdir = setup_outdir(basename, base)

    dirs = [d for d in os.listdir(outdir) if os.path.isdir(join(outdir, d))]
    dirs_split = [d.split('_')[0] for d in dirs]

    dir_nums = []

    for d in dirs_split:
        try:
            num = int(d)
            dir_nums += [num]
        except ValueError:
            pass

    n = max(dir_nums) + 1 if len(dir_nums) > 0 else 0

    now = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
    dirname = '{0:04d}_{1:s}{2:s}'.format(n, now, '_'+title if title else '')
    outdir = setup_outdir(join(outdir, dirname), base)

    return outdir


def get_git_status():
    rev = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE).stdout
    stat = subprocess.run(['git', 'status'], stdout=subprocess.PIPE).stdout
    diff = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE).stdout

    s = '\n'.join(['revision: '+rev.decode('UTF-8'), stat.decode('UTF-8'), diff.decode('UTF-8')])

    return s


def shell_exec(command, check=False):
    run = subprocess.run(command, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return run.stdout.decode('utf-8'), run.stderr.decode('utf-8'),


def shell_exec_silent(command):
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode


def dict_to_string(d):
    assert type(d) in [dict, OrderedDict]

    if type(d) is OrderedDict:
        s = '_'.join(str(k)+str(v)
                for k, v in d.items())
    else:
        s = '_'.join(str(k)+str(d[k])
                for k in sorted(d.keys()))

    return s

