import numpy as np

try:
    import mne
    from mne import events_from_annotations, annotations_from_events, merge_events, create_info
    from mne.io import RawArray
except:
    print("Warning: mne is not installed")

def read_raw_xdf(fname):
    import pyxdf
    streams, header = pyxdf.load_xdf(fname)
    print(len(streams))
    print(streams[0]['info'])
    import sys
    sys.exit()
    data = streams[0]["time_series"].T
    #print(data.shape)
    #assert data.shape[0] == 5  # four raw EEG plus one stim channel
    #data[:4:2] -= data[1:4:2]  # subtract (rereference) to get two bipolar EEG
    #data = data[::2]  # subselect
    #data[:2] *= (1e-6 / 50 / 2)  # uV -> V and preamp gain
    #sfreq = float(streams[0]["info"]["nominal_srate"][0])
    #if sfreq == 0:
    #    sfreq = 100
    #print(sfreq)
    sfreq = 100
    n_ch, n_samples = data.shape
    info = mne.create_info(n_ch, sfreq)
    raw = mne.io.RawArray(data, info)
    return raw

def plot_sensors(ch_names, show_names=False):
    n_ch = len(ch_names)
    fs = 100
    data = np.zeros((n_ch, fs))
    info = create_info(ch_names, fs, ch_types='eeg')

    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020")

    fig = raw.plot_sensors(kind='topomap',ch_type = 'eeg', show=True, show_names = show_names)


def replace_event_id(raw, replaced_id, new_id, verbose=None):
    events, event_id = events_from_annotations(raw, verbose=verbose)
    events_list = events[:,-1]
    events_list = np.unique(events_list)
    event_desc = dict()
    for event_code in events_list:
        if event_code == 99999:
            event_desc[event_code] = 'New Segment/'
        else:
            for k, v in event_id.items():
                if v == event_code:
                    if v == replaced_id:
                        v = new_id
                    k = k[0:-3]
                    k += '{: =3d}'.format(v)
                    event_desc[v] = k
    sfreq = raw.info['sfreq']
    events = merge_events(events, [replaced_id], new_id)
    annot = annotations_from_events(events, sfreq, event_desc=event_desc, verbose=verbose)
    raw.set_annotations(annot, verbose=verbose)
    return raw

def delete_match_events(event_id, events, desc_to_delete):
    # delete element in events array.
    new_event_id = dict()
    ids_comment = list()
    for desc in list(event_id):
        if desc_to_delete.lower() in desc.lower():
            ids_comment.append(event_id[desc])
            continue
        new_event_id[desc] = event_id[desc]

    events = events.copy()
    if len(ids_comment) > 0:
        events_list = events[:, -1]
        for id_comment in ids_comment:
            idx = np.where(events_list == id_comment)[0] # first ele of tuple
            events = np.delete(events, idx, 0)
    return events, new_event_id

def event_id2desc(event_id):
    event_desc = dict()
    for desc in list(event_id):
        id = event_id[desc]
        event_desc[id] = desc
    return event_desc

def get_event_list(events):
    events_list = events[:,-1]
    events_list = np.unique(events_list)
    return events_list

def save_raw(raw, file_name, verbose=None, overwrite=False, delete_comment_events=True):
    events, event_id = events_from_annotations(raw, verbose=verbose)
    if delete_comment_events:
        events, event_id = delete_match_events(event_id, events, 'comment')
    event_desc = event_id2desc(event_id)
    sfreq = raw.info['sfreq']
    annot = annotations_from_events(events, sfreq, event_desc=event_desc, verbose=verbose)
    raw.set_annotations(annot, verbose=verbose)
    raw.save(file_name, overwrite=overwrite, verbose=verbose)