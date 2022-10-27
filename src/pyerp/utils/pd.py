from logging import getLogger
import os
try:
    import pandas as pd
except:
    print("Warning: pandas is not installed")


def create_df_from_evoked(evoked, subject, event, units=None):
    import numpy as np
    ch_names = evoked.ch_names
    times = evoked.times
    N = len(times)
    df = list()
    for ch in ch_names:
        data = dict()
        data['time'] = times
        data['signal'] = np.squeeze(evoked.get_data(picks=[ch], units=units))
        data['event'] = [event for m in range(N)]
        data['subject'] = [subject for m in range(N)]
        data['channel'] = [ch for m in range(N)]
        df.append(pd.DataFrame.from_dict(data))
    return pd.concat(df, ignore_index=True)
    

        
class quickSave(object):
    def __init__(self, file_dir):
        if ".pkl" in file_dir:
            self.file_dir = file_dir
        else:
            self.file_dir = file_dir + ".pkl"
        self.is_exists = os.path.exists(self.file_dir)
        self.df = None

    def add(self, data, index = None, columns = None, html=False, csv=False):
        if self.is_exists:
            df = pd.read_pickle(self.file_dir)
            if columns is not None:
                print("Warning (pylibs) : column value cannot overwrite, original culumns will be used.")
            columns = df.columns
            df_add = pd.DataFrame(data=data, columns=columns, index=index)
            df = pd.concat([df, df_add])
        else:
            df = pd.DataFrame(data=data, index=index, columns=columns)
            self.is_exists = True
        _ = df.to_pickle(self.file_dir)
        self.df = df
        if html:
            self.save_html()
        if csv:
            self.save_csv()
    
    def save_csv(self):
        self.df.to_csv(self.file_dir[:-4] + ".csv")

    def save_html(self):
        self.df.to_html(self.file_dir[:-4] + ".html")
    
    def print(self):
        df = pd.read_pickle(self.file_dir)
        print(df)