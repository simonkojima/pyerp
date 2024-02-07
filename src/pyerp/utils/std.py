import os
import sys
import re

def get_swap_dict(d):
    return {v: k for k, v in d.items()}

def mkdir(dir):
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        
def save_json(file, data):
    import json
    with open(file, "w") as f:
        json.dump(data, f, indent = 4)

def input_binary(message, default = None, suffix = True):
    if suffix:
        if default is None:
            message += " (y/n) : "
        elif type(default) != str:
            raise ValueError("default can only take None, 'y', 'n'")
        elif default.lower() == 'y':
            message += " ([y]/n) : "
        elif default.lower() == 'n':
            message += " (y/[n]) : "
        else:
            raise ValueError("default can only take None, 'y', 'n'")
    while True:
        ans = input(message)
        if ans != "":
            ans = ans.lower()
            if ans == 'y':
                return True
            elif ans == 'n':
                return False
        else:
            if default == 'y':
                return True
            elif default == 'n':
                return False
            elif default == None:
                pass

def print_list(data):
    for line in data:
        print(line)

def sort_list(data):
    return sorted(data, key=natural_keys)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]