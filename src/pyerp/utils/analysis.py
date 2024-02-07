import os
import sys

import numpy as np
import mne

def get_tags(epochs):
    tags = list(epochs.event_id.keys())
    return tags

def get_run_list(epochs):
    tags = list(epochs.event_id.keys())
    runs = list()
    for tag in tags:
        runs.append(int(get_val_in_tag(tag, 'run')))
    val = np.unique(runs).tolist()
    return val

def get_task_list(epochs):
    tags = list(epochs.event_id.keys())
    tasks = list()
    for tag in tags:
        tasks.append(get_val_in_tag(tag, 'task'))
    val = np.unique(tasks).tolist()
    return val

def get_event_id(epochs, tags):
    return list(epochs['/'.join(tags)].event_id.values())

def get_event_id_by_type(marker, type):
    event_id = list()
    for mrk in marker:
        if type in mrk[1].split('/'):
            event_id.append(mrk[0])
    if len(event_id) == 0:
        raise ValueError("Could not find event type '%s' in marker"%(type))
    return event_id

def get_target_from_trial(raws, marker_target):
    """
    marker_target : list
    """
    target = list()

    for raw in raws:
        events, event_id = mne.events_from_annotations(raw)
        e = np.unique(events[:,2])
        t = set(e) & set(marker_target)
        if len(t) > 1:
            raise ValueError("WARNING : trial has more than 2 types of target event.")
        target += list(t)
    return target

def get_n_trials_in_run(epochs, run):
    tags = list(epochs['run:%d'%run].event_id.keys())
    n_trials = 0
    for tag in tags:
        if 'run:%d'%run in tag.split('/'):
            trial_num = int(get_val_in_tag(tag, 'trial'))
            if trial_num > n_trials:
                n_trials = trial_num   
    return n_trials

def get_n_runs(epochs):
    tags = list(epochs.event_id.keys())
    n_runs = list()
    for tag in tags:
        n_runs.append(int(get_val_in_tag(tag, 'run')))
    return max(n_runs)

def get_n_runs_in_task(epochs, task):
    run_list = get_run_list_in_task(epochs, task)
    return len(run_list)

def get_run_list_in_task(epochs, task):
    tags = list(epochs.event_id.keys())
    run_list = list()
    if task.startswith('task:'):
        keyword = task
    else:
        keyword = 'task:%s'%task
    for tag in tags:
        if keyword in tag.split('/'):
            run_list.append(int(get_val_in_tag(tag, 'run')))
    run_list = np.unique(run_list)
    return run_list

def get_val_in_tag(tag, key):
    tag_list = list(tag.split('/'))
    for tag in tag_list:
        if key in tag:
            return tag.split(':')[1]

def get_tags_binary(tags):
    if tags is not None:
        tags_target = list()
        tags_nontarget = list()
        if type(tags[0]) == list:
            for tag in tags:
                tmp = tag.copy()
                tmp.append('target')
                tags_target.append(tmp)

                tmp = tag.copy()
                tmp.append('nontarget')
                tags_nontarget.append(tmp)
        else:
            tmp = tags.copy()
            tmp.append('target')
            tags_target = tmp

            tmp = tags.copy()
            tmp.append('nontarget')
            tags_nontarget = tmp
        #tags_target = process_tags(tags_target)
        #tags_nontarget = process_tags(tags_nontarget)
    else:
        tags_target = 'target'
        tags_nontarget = 'nontarget'

    return tags_target, tags_nontarget


def process_tags(tags):
    processed_tags = list()
    if type(tags[0]) == list:
        for tag in tags:
            processed_tags.append('/'.join(tag))
    else:
        if type(tags) == list:
            processed_tags = '/'.join(tags)
        else:
            processed_tags = tags
    return processed_tags


def get_binary_epochs(epochs, tags=None):
    """
    Parameters
    ==========
    tags : list of str, default = None, e.g. tags = ['event:stim1', 'task:count']
    """

    tags_target, tags_nontarget = get_tags_binary(tags)

    tags_target = process_tags(tags_target)
    tags_nontarget = process_tags(tags_nontarget)

    id_target = list(epochs[tags_target].event_id.values())
    id_nontarget = list(epochs[tags_nontarget].event_id.values())

    if tags is not None:
        X = epochs[process_tags(tags)].copy()
    else:
        X = epochs.copy()
    Y = X.events
    Y = mne.merge_events(Y, id_target, 10)
    Y = mne.merge_events(Y, id_nontarget, 1)
    Y = Y[:, -1]

    return X, Y

def print_event_in_epochs(epochs):
    for key in epochs.event_id.keys():
        print(key)

def print_events_in_epochs(epochs):
    event_id = epochs.event_id
    events = epochs.events

    for event in events:
        print("%d, %s" %(event[0], event_name_from_id(event_id, event[2])))

def event_name_from_id(event_ids, id):
    for key in event_ids:
       if id == event_ids[key] :
        return key

def get_target_of_trial(epochs):
    event_keys = list(epochs.event_id.keys())
    for event_key in event_keys:
        if 'target' in event_key.split('/'):
            target = get_val_in_tag(event_key, 'event')
    return target

def get_events(epochs):
    events = list()
    for tag in epochs.event_id:
        val = get_val_in_tag(tag, 'event')
        events.append(val)
    return events

def split_cv(run_list):
    train = list()
    test = list()
    for test_run in run_list:
        test.append(test_run)
        train_run = list()
        for run in run_list:
            if run != test_run:
                train_run.append(run)
        train.append(train_run)
    val = dict()
    val['train'] = train
    val['test'] = test
    return val