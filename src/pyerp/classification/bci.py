def bci_simulation(epochs,
                   task,
                   vectorizer,
                   clf,
                   enable_dynamic_stopping = True,
                   dynamic_stopping_min_stimulus = 5,
                   dynamic_stopping_mode = 'best-rest',
                   dynamic_stopping_p_value = 0.05, 
                   dynamic_stopping_alternative = 'two-sided',
                   adaptation = False,
                   clf_adaptation = None,
                   soa = None,
                   reject = None):
    from scipy import stats
    import numpy as np

    from ..utils.analysis import (get_binary_epochs, get_target_of_trial, get_events, get_n_trials_in_run,
                            event_name_from_id, get_val_in_tag, split_cv, get_run_list_in_task)
    from .metrices import calc_itr
    from sklearn.metrics import accuracy_score

    run_list = get_run_list_in_task(epochs, task)
    cv = split_cv(run_list)

    scores = dict()
    scores['labels'] = list()
    scores['preds'] = list()
    scores['score'] = list()
    scores['itr'] = list()
    scores['n_channels'] = list()

    scores_dynamic_stopping = dict()
    scores_dynamic_stopping['labels'] = list()
    scores_dynamic_stopping['preds'] = list()
    scores_dynamic_stopping['score'] = list()
    scores_dynamic_stopping['pvalue'] = list()
    scores_dynamic_stopping['n_stimulus'] = list()
    scores_dynamic_stopping['itr'] = list()
    scores_dynamic_stopping['n_channels'] = list()

    for cv_idx, train in enumerate(cv['train']):
        test = cv['test'][cv_idx]
        tags_test = list()
        for train_run in train:
            tags_test.append(['run:%d'%train_run])
        X, Y = get_binary_epochs(epochs, tags_test)
        
        if reject is not None:
            X.drop_bad(reject=reject)

        X.pick_types(eeg=True)

        X = vectorizer.transform(X)
        clf.fit(X, Y)
        n_trials = get_n_trials_in_run(epochs, test)

        labels = list()
        preds = list()

        preds_dynamic_stopping = list()
        pvalue = list()
        n_stimulus = list()

        for trial_num in range(1, n_trials+1):

            epochs_trial = epochs.copy()
            epochs_trial = epochs_trial['run:%d/trial:%d'%(test, trial_num)]
            target = get_target_of_trial(epochs_trial)
            labels.append(target)

            events_in_trial = get_events(epochs_trial)

            distances = dict()
            for event in events_in_trial:
                distances[event] = list()

            dynamic_stopping_triggered = False

            #print_events_in_epochs(epochs_trial)

            n_epochs = epochs_trial.__len__()

            events = epochs_trial.events

            for idx_epoch in range(n_epochs):

                tag = event_name_from_id(epochs_trial.event_id, events[idx_epoch, 2])
                event = get_val_in_tag(tag, 'event')

                X = vectorizer.transform(epochs_trial[idx_epoch])
                
                if adaptation:
                    if target == event:
                        y = np.array([1])
                    else:
                        y = np.array([0])
                    clf_adaptation.adaptation(X, y)

                distance = clf.decision_function(X)
                distances[event].append(distance)

                if dynamic_stopping_triggered == False and enable_dynamic_stopping and dynamic_stopping_min_stimulus <= idx_epoch+1:
                    distances_middle = distances.copy()
                    distances_mean = dict()
                    for event in events_in_trial:
                        distances_middle[event] = np.array(distances_middle[event])
                        distances_mean[event] = np.mean(distances_middle[event])
                    
                    v = list(distances_mean.values())
                    k = list(distances.keys())
                    best_event = k[v.index(max(v))]

                    if dynamic_stopping_mode == 'best-rest':
                        rest_events = events_in_trial.copy()
                        rest_events.remove(best_event)
                        best = distances_middle[best_event]
                        rest = list()
                        for rest_event in rest_events:
                            rest.append(distances_middle[rest_event])
                        best = np.array(best)
                        rest = np.concatenate(rest)

                        t_score, p = stats.ttest_ind(best, rest, equal_var = False, alternative = dynamic_stopping_alternative)

                        if p <= dynamic_stopping_p_value:
                            preds_dynamic_stopping.append(best_event)
                            pvalue.append(p)
                            n_stimulus.append(idx_epoch+1)
                            dynamic_stopping_triggered = True
                            
                    elif dynamic_stopping_mode == 'best-second':
                        pass
                    elif dynamic_stopping_mode == 'target-bestnontarget':
                        target_distance = distances[target]
                        
                        distances_nontarget = distances.copy()
                        distances_nontarget.pop(target)

                        v = list(distances_mean.values())
                        k = list(distances.keys())
                        bestnontarget_event = k[v.index(max(v))]
                        
                        bestnontarget_distance = distances[bestnontarget_event]

                        t_score, p = stats.ttest_ind(target_distance, bestnontarget_distance, equal_var = False, alternative = dynamic_stopping_alternative)
                        
                        if p <= dynamic_stopping_p_value:
                            preds_dynamic_stopping.append(target)
                            pvalue.append(p)
                            n_stimulus.append(idx_epoch+1)
                            dynamic_stopping_triggered = True

                    else:
                        raise ValueError("Unknown dynamic stopping mode : %s" %dynamic_stopping_mode)

                    if idx_epoch+1 == n_epochs and dynamic_stopping_triggered == False:
                        preds_dynamic_stopping.append(best_event)
                        pvalue.append(p)
                        n_stimulus.append(idx_epoch+1)
                        
            if adaptation:
                clf_adaptation.apply_adaptation()

            # normal bci simulation (without dynamic stopping)        
            for event in events_in_trial:
                distances[event] = np.array(distances[event])
                distances[event] = np.mean(distances[event])
            v = list(distances.values())
            k = list(distances.keys())
            decoded = k[v.index(max(v))]
            preds.append(decoded) 
        scores['labels'].append(labels)
        scores['preds'].append(preds)
        scores['score'].append(accuracy_score(labels, preds))
        req_time = np.mean(n_epochs)*soa + (epochs.tmax - soa)
        req_time = req_time / 60
        scores['itr'].append(calc_itr(6, accuracy_score(labels, preds), req_time))
        scores['n_channels'].append(len(epochs_trial.ch_names))

        scores_dynamic_stopping['labels'].append(labels)
        scores_dynamic_stopping['preds'].append(preds_dynamic_stopping)
        #print(labels)
        #print(preds_dynamic_stopping)
        scores_dynamic_stopping['score'].append(accuracy_score(labels, preds_dynamic_stopping))
        scores_dynamic_stopping['pvalue'].append(pvalue)
        scores_dynamic_stopping['n_stimulus'].append(n_stimulus)
        scores_dynamic_stopping['n_channels'].append(len(epochs_trial.ch_names))
        req_time = np.mean(n_stimulus)*soa + (epochs.tmax - soa)
        req_time = req_time / 60
        scores_dynamic_stopping['itr'].append(calc_itr(len(events_in_trial), accuracy_score(labels, preds_dynamic_stopping), req_time))

    r = dict()
    r['normal'] = scores
    r['dynamic_stopping'] = scores_dynamic_stopping

    return r 