import os
import sys

import numpy as np
import mne

def ica_eog_removal(raw,
                l_freq = 1.0,
                h_freq = None,
                visualize = True,
                n_components = None,
                noise_cov = None,
                random_state = None,
                method = 'fastica',
                fit_params = None,
                max_iter = 'auto',
                allow_ref_meg = False,
                ch_name = None,
                threshold = 3.0,
                verbose=None):
    from mne.preprocessing import (ICA, create_eog_epochs)
    filt_raw = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
    ica = ICA(n_components = n_components,
              noise_cov = noise_cov,
              random_state = random_state,
              method = method,
              fit_params = fit_params,
              max_iter = max_iter,
              allow_ref_meg = allow_ref_meg,
              verbose = verbose)
    ica.fit(filt_raw)
    eog_indices, eog_scores = ica.find_bads_eog(raw,
                                    ch_name = ch_name,
                                    threshold = threshold,
                                    verbose = verbose)

    if len(eog_indices) == 0:
        ica.plot_sources(raw, show_scrollbars=False)
        ica.plot_scores(eog_scores)
        eog_indices = input("input the index of EOG. (start from 0)\n")
        eog_indices = eval(eog_indices)
        if type(eog_indices) == int:
            eog_indices = [eog_indices]

    ica.exclude = eog_indices
    print("ICA Exclude : %s" %(str(ica.exclude)))

    if visualize:
        ica.plot_scores(eog_scores)
        ica.plot_sources(raw, show_scrollbars=False)
        input()

    return ica