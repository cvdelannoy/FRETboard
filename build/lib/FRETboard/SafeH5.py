from h5py import File
import os
import time

class SafeH5(File):
    """
    write to h5 file safely, queue if other process currently has access
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
        File.__init__(self, *args, **kwargs)

    def __exit__(self, *args, **kwargs):
        File.__exit__(self, *args, **kwargs)
        os.close(self._flock)
        os.remove(self._lock)