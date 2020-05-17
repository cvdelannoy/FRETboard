import h5py

class H5Walker:
    def __init__(self):
        # Store an empty list for dataset names
        self.names = []

    def __call__(self, name, h5obj):
        # only h5py datasets have dtype attribute, so we can search on this
        if hasattr(h5obj, 'dtype') and not name in self.names:
            self.names += [name]
