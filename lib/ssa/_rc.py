'''
Created on Nov 26, 2014

@author: nick

Global run control routines
'''
from __future__ import division, print_function; __metaclass__ = type

import logging
log = logging.getLogger('sim.rc')

import os, sys, errno
import ssa
from yamlcfg import YAMLConfig
from work_managers import SerialWorkManager

RC_DEFAULT_FILENAME = "sim.cfg"
SYS_DEFAULT_FILENAME = "params.cfg"
SYS_STATE_DEFAULT_FILENAME = "state.pkl"

class SysCtl:

    def __init__(self):

        self.rcfile = RC_DEFAULT_FILENAME
        self.stateout = None
        self.verbosity = None
        self._sim_manager = None
        self._data_manager = None
        self._system = None
        self.work_manager = SerialWorkManager()
        # Rc config
        self.config = YAMLConfig()
        self.process_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

        self.status_stream = sys.stdout

    def add_args(self, parser):
        group = parser.add_argument_group('general options')
        group.add_argument('-r', '--rcfile', metavar='RCFILE', dest='rcfile',
                            default=RC_DEFAULT_FILENAME,
                            help='use RCFILE as the run-time configuration file (default: %(default)s)')
        group.add_argument('-so', '--stateout', metavar='STATEOUT', dest='stateout', default='state.cpt',
                            help='Prepare simulation state to STATEFILE (default: %(default)s)')
        egroup = group.add_mutually_exclusive_group()
        egroup.add_argument('--quiet', dest='verbosity', action='store_const', const='quiet',
                             help='emit only essential information')
        egroup.add_argument('--verbose', dest='verbosity', action='store_const', const='verbose',
                             help='emit extra information')
        egroup.add_argument('--debug', dest='verbosity', action='store_const', const='debug',
                            help='enable extra checks and emit copious information')

    @property
    def verbose_mode(self):
        return (self.verbosity in ('verbose', 'debug'))

    @property
    def debug_mode(self):
        return (self.verbosity == 'debug')

    @property
    def quiet_mode(self):
        return (self.verbosity == 'quiet')

    def process_args(self, args, config_required = True):
        self.cmdline_args = args
        self.verbosity = args.verbosity

        self.stateout = args.stateout

        if args.rcfile:
            self.rcfile = args.rcfile
        try:
            self.read_config()
        except IOError as e:
            if e.errno == errno.ENOENT and not config_required:
                pass
            else:
                raise
        self.config_logging()
        self.config['args'] = {k:v for k,v in args.__dict__.iteritems() if not k.startswith('_')}
        self.process_config()

    def process_config(self):
        log.debug('config: {!r}'.format(self.config))

    def read_config(self, filename = None):
        if filename:
            self.rcfile = filename

        self.config.update_from_file(self.rcfile)

    def config_logging(self):
        import logging.config
        logging_config = {'version': 1, 'incremental': False,
                          'formatters': {'standard': {'format': '-- %(levelname)-8s [%(name)s] -- %(message)s'},
                                         'debug':    {'format': '''\
-- %(levelname)-8s %(asctime)24s PID %(process)-12d TID %(thread)-20d
   from logger "%(name)s" 
   at location %(pathname)s:%(lineno)d [%(funcName)s()] 
   ::
   %(message)s
'''}},
                          'handlers': {'console': {'class': 'logging.StreamHandler',
                                                   'stream': 'ext://sys.stdout',
                                                   'formatter': 'standard'}},
                          'loggers': {'sim': {'handlers': ['console'], 'propagate': False},
                                      'ssa': {'handlers': ['console'], 'propagate': False},
                                      'work_managers': {'handlers': ['console'], 'propagate': False},
                                      'py.warnings': {'handlers': ['console'], 'propagate': False}},
                          'root': {'handlers': ['console']}}

        logging_config['loggers'][self.process_name] = {'handlers': ['console'], 'propagate': False}

        if self.verbosity == 'debug':
            logging_config['root']['level'] = 5 #'DEBUG'
            logging_config['handlers']['console']['formatter'] = 'debug'
        elif self.verbosity == 'verbose':
            logging_config['root']['level'] = 'INFO'
        else:
            logging_config['root']['level'] = 'WARNING'

        logging.config.dictConfig(logging_config)
        logging_config['incremental'] = True
        logging.captureWarnings(True)

    def pstatus(self, *args, **kwargs):
        fileobj = kwargs.pop('file', self.status_stream)
        if kwargs.get('termonly', False) and not fileobj.isatty():
            return
        if self.verbosity != 'quiet':
            print(*args, file=fileobj, **kwargs)

    def pstatus_term(self, *args, **kwargs):
        fileobj = kwargs.pop('file', self.status_stream)
        if fileobj.isatty() and self.verbosity != 'quiet':
            print(*args, file=fileobj, **kwargs)

    def pflush(self):
        for stream in (self.status_stream, sys.stdout, sys.stderr):
            try:
                stream.flush()
            except AttributeError:
                pass

    def get_sim_manager(self):
        if self._sim_manager is None:
            self._sim_manager = self.new_sim_manager()
        return self._sim_manager

    def new_sim_manager(self):
        import sim_manager
        sim_manager = sim_manager.SimManager(rc=self)
        log.debug('loaded simulation manager {!r}'.format(sim_manager))

        return sim_manager

    def get_data_manager(self):
        if self._data_manager is None:
            self._data_manager = self.new_data_manager()

        return self._data_manager

    def new_data_manager(self):
        import data_manager
        data_manager = data_manager.DataManager()
        log.debug('loaded data manager {!r}'.format(data_manager))

        return data_manager

    def get_system(self):
        if self._system is None:
            self._system = self.new_system()

        return self._system

    def new_system(self):
        import system
        system = system.System(rc=self)
        log.debug('loaded system {!r}'.format(system))

        return system

    def get_work_manager(self):
        return self.work_manager

    sim_manager = property(get_sim_manager)
    data_manager = property(get_data_manager)
    system = property(get_system)
