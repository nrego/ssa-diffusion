'''
Created on Nov 26, 2014

Manages output

@author: Nick Rego
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
from matplotlib import pyplot
from matplotlib import animation

# Default - data buffer flush interval
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
        self.fmt = None
        self.dlength = None
        self.buffer_flush = None
        self.data_ptr = None

        self.data_field = None
        self.data_view = None
        self.data_timepts = None

        # Output for sim state
        self.stateout = self.rc.stateout

        self._system = None
        self._sim_manager = None
        self.process_config()

    def __enter__(self):
        self._init_data_stream()

        return self

    def __exit__(self, type, value, traceback):
        if self.outfile:
            self.dump_data_buffer()
            self.outfile.close()

    # Open data file output stream, initialize data buffer
    def _init_data_stream(self, outfilename=None, append=True):
        outfilename = outfilename or self.outfilename
        mode = 'a' if append else 'w'
        self.outfile = open(outfilename, mode)

        self.fmt = '%.4f'
        fmtrs = [' %d' for i in xrange(self.system.compartment_cnt)]
        for fmtstr in fmtrs:
            self.fmt += fmtstr

        log.debug('Format string: {}'.format(self.fmt))

        self.data_ptr = 0
        self.data_buffer = numpy.empty((1, self.system.species_cnt,
                                        self.system.compartment_cnt+1),
                                       dtype=numpy.float64)
        self.add_data()
        self.dump_data_buffer()

    def add_data(self):

        if self.data_ptr >= self.buffer_flush:
            self.dump_data_buffer()
        try:
            curr_buffer = self.data_buffer[self.data_ptr]
            curr_buffer[:, 0] = self.system.time
            curr_buffer[:,1:] = self.system.propensity.n_species

        except IndexError:
            raise

        self.data_ptr += 1

    # Output data buffer to outstream
    def dump_data_buffer(self):

        for timepoint in xrange(self.data_ptr):
            numpy.savetxt(self.outfile, self.data_buffer[timepoint], self.fmt)

        self.data_buffer = numpy.empty((self.buffer_flush, self.system.species_cnt,
                                        self.system.compartment_cnt+1),
                                       dtype=numpy.float64)

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

    # Read in outfile for post-simulation analysis
    def load_data(self, datafile=None):
        datafile = datafile or self.outfilename

        try:
            self.data_field = numpy.loadtxt(datafile)
        except Exception as e:
            raise IOError('Error reading datafile: {}: {}'.format(datafile, e))


    def get_data_for_species(self, species_idx, compartment_range=None):
        compartment_cnt = self.system.compartment_cnt
        if not compartment_range:
            compartment_range = range(compartment_cnt)
        cnt = self.system.species_cnt
        self.data_view = numpy.sum(self.data_field[species_idx::cnt, compartment_range], axis=1)
        self.data_timepts = self.data_field[species_idx::cnt, 0]

    def plot_animation(self, species_idx, start_pt=0):
        fig = pyplot.figure()
        ax = pyplot.axes()
        cnt = self.system.species_cnt
        self.data_view = self.data_field[species_idx::cnt, 1:]
        npts = self.data_view.shape[0]
        self.data_timepts = self.data_field[species_idx::cnt, 0]
        bin_ctrs = (self.system.compartment_bounds[1:] - self.system.compartment_bounds[:-1]) / 2 \
                    + self.system.compartment_bounds[:-1]

        rects, = ax.plot(bin_ctrs, self.data_view[start_pt])

        animation.FuncAnimation(fig, _animate, numpy.arange(start_pt, npts),
                                fargs=(rects, self, bin_ctrs), interval=20)

        pyplot.show()

    def plot_data(self, label=''):
        #fig = pyplot.figure()
        pyplot.title('Cargo Export, {:d} bins'.format(self.system.compartment_cnt), fontsize=45)

        pyplot.plot(self.data_timepts, self.data_view, label=label)
        pyplot.xlabel('Time (s)', fontsize=30)
        pyplot.ylabel('N Cargo', fontsize=30)
        #pyplot.show()

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

def _animate(i, rects, dm, bin_ctrs):
    new_data = dm.data_view[i,:]

    rects.set_data(bin_ctrs, new_data)
    return rects
