import os
import json

def save_json(fname, data):
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            _data = json.load(f)
        data = {**_data, **data} #overwrite by the right argument.
    
    with open(fname, 'w') as f:
        json.dump(data, f, indent = 4)