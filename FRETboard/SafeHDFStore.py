from pandas import HDFStore
import os
import time

class SafeHDFStore(HDFStore):
    """
    write to HDFStore safely, queue if other process currently has access
    from: https://stackoverflow.com/questions/22522551/pandas-hdf5-as-a-database
    """
    def __init__(self, *args, **kwargs):
        probe_interval = kwargs.pop("probe_interval", 0.01)
        self._lock = "%s.lock" % args[0]
        while True:
            try:
                self._flock = os.open(self._lock, os.O_CREAT |
                                                  os.O_EXCL |
                                                  os.O_WRONLY)
                break
            except FileExistsError:
                time.sleep(probe_interval)

        HDFStore.__init__(self, *args, **kwargs)

    def __exit__(self, *args, **kwargs):
        HDFStore.__exit__(self, *args, **kwargs)
        os.close(self._flock)
        os.remove(self._lock)