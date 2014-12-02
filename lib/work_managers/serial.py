# Adapted from WESTPA work manager
#
#Copyright (C) 2013 Matthew C. Zwier and Lillian T. Chong
#
# This file is part of WESTPA.

from __future__ import division; __metaclass__ = type

import logging, sys

log = logging.getLogger(__name__)

from . import WorkManager, WMFuture

class SerialWorkManager(WorkManager):
    @classmethod
    def from_environ(cls, wmenv=None):
        return cls()
    
    def __init__(self):
        log.debug('initializing serial work manager')
        super(SerialWorkManager,self).__init__()
        self.n_workers = 1
        
    def submit(self, fn, args=None, kwargs=None):
        ft = WMFuture()
        try:
            result = fn(*(args if args is not None else ()), **(kwargs if kwargs is not None else {}))
        except Exception as e:
            ft._set_exception(e, sys.exc_info()[2])
        else:
            ft._set_result(result)
        return ft
    