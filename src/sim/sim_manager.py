'''
Created on Nov 26, 2014

@author: Nick Rego
'''
from __future__ import division, print_function; __metaclass__ = type

import numpy
from numpy.random import random as rand
import sim
import ssa

import logging
log = logging.getLogger('simulation manager')

class SimManager:
    '''
    High level simulation task manager

    Responsible for high level algorithm routines, delegates appropriate tasks to system, data_manager
    '''
    def process_config(self):
        config = self.rc.config

        self.max_run_walltime = config.get(['rc', 'max_run_wallclock'], None)
        self.max_total_time = config.get(['rc', 'max_time'], 10)

    def __init__(self, rc=None):
        self.rc = rc or ssa.rc
        self.work_manager = self.rc.get_work_manager()
        self.data_manager = self.rc.get_data_manager()
        self.system = self.rc.get_system()

        # config items
        self.max_run_walltime = None
        self.max_total_time = None
        self.process_config()

        self.n_iter = None

    # Initialize system from parameter file
    def initialize(self, paramfile):
        self.n_iter = 0
        self.system.initialize(paramfile)

    # Load system from statefile
    def load_state(self, state):
        n_iter = state.pop('n_iter')
        self.n_iter = n_iter
        self.system.load_state(state)

    def run(self):
        '''Run SSA algorithm until specified time'''
        pstatus = self.rc.pstatus
        pflush = self.rc.pflush

        pstatus('Starting run')
        pstatus('Max total time: {!r}'.format(self.max_total_time))
        pstatus('Initial system state:')
        pstatus('Current Time: {:.4f} s'.format(self.time))
        pstatus('Current iter: {}'.format(self.n_iter))
        pflush()

        self.system.print_state()

        try:
            while (self.time < self.max_total_time):
                self.n_iter += 1
                #log.debug('Starting iteration {!r}, total time: {!r}'.format(self.n_iter, self.time))

                if self.n_iter % 5000 == 0:
                    pstatus("\rIter: {!r}  Time: {:.4f} s".format(self.n_iter, self.time))
                    pflush()

                a = self.system.alpha

                r1, r2 = rand(), rand()*a
                #log.debug('a: {}'.format(a))
                if a <= 0:
                    pstatus('a = {!r}'.format(a))
                    pstatus('Exiting...')
                    break

                tau = numpy.log(1/r1)/a

                if self.time + tau >= self.max_total_time:
                    pstatus('Iter: {!r}  Time: {:.4f} s'.format(self.n_iter, self.time))
                    pstatus('Next tau ({!r}) would put simulation over max total time ({!r} s)'.
                            format(tau, self.max_total_time))
                    pstatus('Exiting...')
                    pflush()
                    break

                self.system.run_iter(r2, tau)

        finally:
            self.system.update_state_from_propensity()
            pstatus('Ending state after {!r} iterations, {} s:'.
                    format(self.n_iter, self.time))
            self.system.print_state()

    def run_iter(self):
        self.n_iter += 1
        log.info('S')

    @property
    def time(self):
        return self.system.time

    @property
    def state(self):
        state = self.system.state
        state['n_iter'] = self.n_iter

        return state
