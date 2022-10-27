import os
import sys
import re

def mkdir(dir):
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)

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

def sort_list(data):
    return sorted(data, key=natural_keys)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]