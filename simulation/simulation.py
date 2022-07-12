#!/usr/bin/env python3

import datetime
import glob
import h5py
import numpy as np
import os
import os.path
from os.path import join
import platform
import pickle as pkl
import signal
import subprocess
import sys
import tarfile
import traceback

from .helpers import dict_to_string, get_git_status, setup_outdir, setup_numbered_outdir, shell_exec, shell_exec_silent
import torch

class Simulation:
    """
    Context manager for simulations.

    Output directory is:

        output_root/sim_name/number_date_suffix/

    where suffix either is a string, or key-value pairs from a dict.
    If numbered is set to False, the output path is

        output_root/sim_name/suffix/

    unless suffix is None or '', in which

        output_root/sim_name/

    is used.

    If suspend is set, setup only paths and logging, but don't be verbose and
    don't save code status.

    """

    def __init__(self, sim_name=None, output_root='out', suffix=None, numbered=True, codedir='.', suspend=False, catch_sigint=False):
        self._sim_name = os.path.basename(sys.argv[0]) if sim_name is None else sim_name
        self._output_root = output_root
        self._suffix = suffix
        self._numbered = numbered
        self._codedir = codedir
        self._suspend = suspend
        self._catch_sigint = catch_sigint

        self._title = ''

        if suffix:
            if type(suffix) is str:
                self._title = suffix
            else:
                self._title = dict_to_string(suffix)

        # file names
        self._simfile = 'simulate.log'
        self._suspendfile = 'suspend.log'
        self._gitfile = 'git_state.log'
        self._tarfile = 'sources.tar.gz'
        self._datafile_base = 'data{0:s}.{1:s}'

    def restore_from_path(path):
        return Simulation(path, output_root='.', numbered=False, suspend=True)

    def __enter__(self):
        # create output directory
        outdir = setup_outdir(join(self._output_root, self._sim_name))

        if self._numbered:
            outdir = setup_numbered_outdir(outdir, title=self._title)
        elif self._suffix:
            outdir = setup_outdir(join(outdir, self._title))

        self._outdir = outdir

        # adjust file paths
        self._simfile = join(outdir, self._simfile)
        self._suspendfile = join(outdir, self._suspendfile)
        self._gitfile = join(outdir, self._gitfile)
        self._tarfile = join(outdir, self._tarfile)
        self._datafile = join(outdir, self._datafile_base)

        self._logfile = self._suspendfile if self._suspend else self._simfile
        mode = 'append' if self._suspend else 'write'

        # setup logging
        sys.stdout = sys.stderr = Logger(self._logfile, mode=mode)

        if not self._suspend:
            # be verbose
            print('simulation setup')
            print('codedir: ', self._codedir)
            print('outdir: ', self._outdir)

            # check for existence of git repo
            if shell_exec_silent(['git', 'rev-parse', '--is-inside-work-tree']) != 0:
                raise ValueError('not inside a git repo')

            # save git state

            git_state = get_git_status()
            with open(self._gitfile, 'w') as f:
                f.write(git_state)

            # redundancy: also save all code files
            self.__save_code()

            # check for existing data files and warn
            datafiles = glob.glob(join(outdir, self._datafile.format('*', '*')))
            if len(datafiles) > 0:
                print('WARNING: data files already exist:')
                [print('  {0:s}'.format(df) for df in datafiles)]

            # separate output from simulation output
            print('-'*70)

        # signal handling
        if self._catch_sigint:
            self._received_sigint = False

            signal.signal(signal.SIGINT, self.__signal_handler)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # explicitly print exception as it is not written to log file otherwise

        if exc_type is not None:
            sys.stdout.log('caught {0:s}: {1:s}\n'.format(exc_type.__name__, str(exc_value)))
            sys.stdout.log('traceback:\n')
            [sys.stdout.log(s) for s in traceback.format_tb(exc_traceback)]

        # quit logging
        sys.stdout, sys.stderr = sys.stdout.close()

    def create_subdirectory(self, name):
        return setup_outdir(join(self._outdir, name))

    def save_pytorch(self, data, *, subdir='models', overwrite=False, epoch=None, prefix=''):
        outdir = join(self._outdir, subdir)
        setup_outdir(outdir)

        if epoch is None:
            datafile = join(outdir, f'{prefix}checkpoint.pt')
        else:
            datafile = join(outdir, f'{prefix}checkpoint_epoch{epoch}.pt')

        if not overwrite and os.path.exists(datafile):
            raise IOError('data file already exists: ' + datafile)

        torch.save(data, datafile)
        return datafile

    def save_data(self, data, title='', *, subdir=None, overwrite=False, mode='pkl'):
        assert mode in ['pkl', 'hdf5']

        if len(title) > 0: title = '_'+title
        datafile = self._datafile_base.format(title, mode)

        if subdir is None:
            outdir = self._outdir
        else:
            outdir = join(self._outdir, subdir)
            setup_outdir(outdir)

        datafile = join(outdir, datafile)

        if not overwrite and os.path.exists(datafile):
            raise IOError('data file already exists: '+datafile)

        if mode == 'pkl':
            with open(datafile, 'wb') as f:
                pkl.dump(data, f)

        elif mode == 'hdf5':
            assert type(data) is dict

            with h5py.File(datafile, 'w' if overwrite else 'w-') as f:
                for k, v in data.items():
                    assert type(v) is np.ndarray, 'data values must be of type np.ndarray'
                    f.create_dataset(k, v.shape, v.dtype, v)
        else:
            raise ValueError()

        return datafile

    def restore_data(self, title='', *, subdir=None, mode='pkl'):
        assert mode in ['pkl', 'hdf5']

        if len(title) > 0: title = '_'+title
        datafile = self._datafile_base.format(title, mode)

        if subdir is None:
            outdir = self._outdir
        else:
            outdir = join(self._outdir, subdir)

        datafile = join(outdir, datafile)

        if not os.path.exists(datafile):
            raise IOError('data file doesn\'t exist: '+datafile)

        if mode == 'pkl':
            with open(datafile, 'rb') as f:
                data = pkl.load(f)

        elif mode == 'hdf5':
            with h5py.File(datafile, 'r') as f:
                data = {}
                for k in [*f.keys()]:
                    data[k] = np.asarray(f[k])
        else:
            raise ValueError()

        return data

    def save_data_hdf5(self, data, title='', *, subdir=None, overwrite=False):
        return self.save_data(data, title, subdir=subdir, overwrite=overwrite, mode='hdf5')

    def restore_data_hdf5(self, title='', *, subdir=None):
        return self.restore_data(title, subdir=subdir, mode='hdf5')

    def __save_code(self):
        # files on current level
        extensions = ['py', 'sh']
        files = [x for ext in extensions
                for x in glob.glob(join(self._codedir, '*.'+ext))]

        # modules
        modules = [os.path.dirname(d)
                for d in glob.glob(join(self._codedir, '*', '__init__.py'))]

        # create archive
        with tarfile.open(self._tarfile, 'w:gz', dereference=True) as tar:
            for f in files:
                print('saving file to tar: {0:s}'.format(f))
                tar.add(f, arcname=join('src', os.path.basename(f)))

            def filter(obj):
                exclude = ['.git', '__pycache__']

                if os.path.basename(obj.name) in exclude:
                    return None

                return obj

            for m in modules:
                print('saving module to tar: {0:s}'.format(m))
                tar.add(m, arcname=join('src', os.path.basename(m)), filter=filter)

    def __signal_handler(self, signum, frame):
        self._received_sigint = True

    @property
    def outdir(self):
        return self._outdir

    @property
    def received_sigint(self):
        if not self._catch_sigint:
            raise ValueError('attempting to read out sigint status, but '+
                    'catch_sigint was not set.')

        return self._received_sigint

class Logger:
    """
    Logger: replace stdout to tee output to file.

    usage:

        sys.stdout = Logger(logfile)

    to log stdout to file but not stderr, or

        sys.stdout = sys.stderr = Logger(logfile)

    to log both stdout and stderr. To close, use

        sys.stdout, sys.stderr = sys.stdout.close()
    """

    def __init__(self, filename, *, mode='write', verbose=True):
        """
        args:
            filename            log file name

        keyword-only args:
            mode                one of 'write', 'overwrite', 'append'
            verbose             write additional information to log file
        """

        self._stdout = sys.stdout
        self._stderr = sys.stderr

        if mode == 'write':
            if os.path.isfile(filename):
                errstr = 'mode is "write", but file exists: {}'.format(filename)
                raise IOError(errstr)
            m = 'w'

        elif mode == 'overwrite':
            m = 'w'

        elif mode == 'append':
            m = 'a'

        self._log = open(filename, m, buffering=1)  # line buffering

        if verbose:
            hostname = platform.node()
            self._log.write('running on host {0:s}\n'.format(hostname))

            now = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
            self._log.write('logging started at {0:s}\n'.format(now))
            self._log.write('-'*70+'\n')

        self._verbose = verbose

    def __getattr__(self, attr):
        return getattr(self._stdout, attr)

    def write(self, message):
        self._stdout.write(message)
        self._log.write(message)

    def log(self, message):
        self._log.write(message)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        if self._verbose:
            now = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
            self._log.write('-'*70+'\n')
            self._log.write('logging ended at {0:s}\n'.format(now))

        self._log.close()

        return self._stdout, self._stderr
