import os
import sys
import matplotlib.pyplot as plt

def savefig(fname, dpi='figure', format=None, delete_white = True, cache_dir = None):
    from ..utils import std
    from .image import delete_white
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), 'pyerp')
    std.mkdir(cache_dir)
    
    basename = os.path.basename(fname)
    
    plt.savefig(fname=os.path.join(cache_dir, basename), dpi = dpi, format = format)
    
    delete_white(file=os.path.join(cache_dir, basename), file_out=fname)
    
    
    