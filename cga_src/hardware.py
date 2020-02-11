'''The hardware module encapsulates or hardware (GPU, CPU) handling'''
import warnings
import logging

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count, Queue

import torch
#from multiprocessing import Array


class HardwareManager():
    '''The hardware manager distributes jobs on the different CPU/GPU components'''
    _initialised = False

    @classmethod
    def add_job(cls, func, args):
        '''Adds a job to be asynchronously executed on a hardware component'''
        if not cls._all_devices:
            cls.set_devices()

        if not cls._pool:
            cls._start_pool()

        def run_job(func, args):
            device = cls._available_devices.get()
            args['device'] = device
            ret = func(**args)
            cls._available_devices.put(device)

            return ret

        future = cls._pool.submit(run_job, func, args)
        return future

    @classmethod
    def init(cls):
        '''Inits the hardware manager, will be called automatically'''
        if cls._initialised:
            return

        cls._all_devices = []
        cls._pool = None
        cls._jobs = Queue()
        cls._initialised = True

    @staticmethod
    def get_number_of_gpus():
        '''Number of GPUs currently usable'''
        return torch.cuda.device_count()

    @classmethod
    def set_devices(cls, number_of_cpus=0, gpus=None):
        '''Initialises which devices should be used'''

        if number_of_cpus == 'all':
            number_of_cpus = cpu_count()
        elif number_of_cpus < 0:
            number_of_cpus = cpu_count() - number_of_cpus

        devices = gpus or [
            'cuda:%d' % i for i in range(torch.cuda.device_count())
        ]

        # this is supposed to fix a multi-threaded initialisation error, see
        # https://github.com/pytorch/pytorch/issues/16559#issuecomment-482568464
        for device in devices:
            try:
                torch.Tensor([1.]).to(device)
            except RuntimeError:
                # occurs when we set a "fake" device which we do in the visualisations part
                pass

        if number_of_cpus:
            devices += ['cpu' for cpu in range(number_of_cpus)]

        if not devices:
            if number_of_cpus == 0:
                warnings.warn('No GPU devices found and number ' +
                              'of CPUs has been set to 0; default to one CPU')
                devices = ['cpu']

        logging.info('Hardware manager will use %d device(s): %s',
                     len(devices), devices)

        if cls._pool is not None and cls._all_devices != devices:
            cls._stop_pool()

        cls._all_devices = devices
        cls._available_devices = Queue()
        for device in devices:
            cls._available_devices.put(device)

    @classmethod
    def _start_pool(cls):
        cls._pool = ThreadPoolExecutor(
            max_workers=len(cls._all_devices))

    @classmethod
    def _stop_pool(cls):
        if not cls._pool:
            return

        cls._pool.shutdown(True)
        cls._pool = None


HardwareManager.init()
