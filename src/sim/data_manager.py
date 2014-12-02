'''
Created on Nov 26, 2014

Manages output

@author: nick
'''
from __future__ import division, print_function; __metaclass__ = type

import logging
log = logging.getLogger('data manager')

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

import ssa
import numpy

# Default - flush data buffer every 500 iterations
BUFFER_FLUSH = 50

class DataManager:
    '''
    Handles data output
    '''
    def process_config(self):
        config = self.rc.config

        self.outfilename = config.get(['io', 'outfile'], 'out.dat')
        self.buffer_flush = config.get(['io', 'bufferflush'], BUFFER_FLUSH)

    def __init__(self, rc=None):
        self.rc = rc or ssa.rc

        # Output for sim data
        self.outfilename = None
        self.outfile = None
        self.data_buffer = None
        self.dtype = None
        self.dlength = None
        self.buffer_flush = None
        self.data_ptr = None

        # Output for sim state
        self.stateout = self.rc.stateout

        self._system = None
        self._sim_manager = None
        self.process_config()

    def __enter__(self):
        self._init_data_stream()

        return self

    def __exit__(self):
        if self.outfile:
            self.outfile.close()

    # Open data file output stream, initialize data buffer
    def _init_data_stream(self, outfilename=None, append=True):
        outfilename = outfilename or self.outfilename
        mode = 'a' if append else 'w'
        self.outfile = open(outfilename, mode)

        # Ew - have to clean up system/propensity/species reference chains ...
        species = self.system.propensity.species
        n_species = self.system.propensity.n_species
        width = n_species.shape[1]
        self.dtype = numpy.dtype([('time', numpy.float32)] +
                                 [(specie, numpy.uint32, (width,)) for specie in species])

        self.data_ptr = 0
        self.data_buffer = numpy.empty((1,), dtype=self.dtype)
        self.add_data()
        self.dump_data_buffer()

    def add_data(self):
        try:
            curr_buffer = self.data_buffer[self.data_ptr]
            curr_buffer['time'] = self.system.time
            for specie, n_species in self.system.n_species.items():
                curr_buffer[specie] = n_species

        except IndexError:
            raise

        self.data_ptr += 1

    # Output data buffer to outstream
    def dump_data_buffer(self):
        numpy.savetxt(self.outfile, self.data_buffer)
        self.data_buffer = numpy.empty((self.buffer_flush,), dtype=self.dtype)

        self.data_ptr = 0

    def dump_state(self, stateout=None):
        '''dump state info to binary file'''
        outfile = stateout or self.stateout
        self.rc.pstatus('Dumping state to {!r}'.format(outfile))

        with open(outfile, 'w') as o:
            pickle.dump(self.sim_manager.state, o, protocol=2)

    def load_state(self, statefile):
        '''Read binary statefile, load into System'''
        self.rc.pstatus('Loading state from {!r}'.format(statefile))
        self.rc.pstatus()
        with open(statefile, 'r') as f:
            state = pickle.load(f)
            self.sim_manager.load_state(state)

        self.rc.pstatus('...Loaded with money')
        self.rc.pstatus()
        self.rc.pflush()

    @property
    def system(self):
        if self._system is None:
            self._system = self.rc.get_system()

        return self._system

    @system.setter
    def system(self, system):
        self._system = system

    @property
    def sim_manager(self):
        if self._sim_manager is None:
            self._sim_manager = self.rc.get_sim_manager()

        return self._sim_manager
