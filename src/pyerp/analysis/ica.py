import numpy as np
import scipy
import mne

from .signal import apply_sosfilter
from .raw import concatenate_raws_vhdr, reconstruct_raw

def find_bad_eog(raw, ica, filter = [1, 10], threshold = 0.9):
    """
    Parameters
    ==========

    raw : raw instance contains eog channels.
    ica : ica instance
    filter : filter range will be used for eog channels
    threshold, numerical or 'max': 
    
    """
    #raw_eeg_ = mne.io.RawArray(data = raw.get_data(), info = mne.create_info(raw_eeg.ch_names, Fs))

    Fs = raw.info['sfreq']

    raw_eog = raw.copy().pick(picks = ['eog'])
    raw_eeg = raw.copy().pick(picks = ['eeg'])

    raw_eog = reconstruct_raw(raw_eog) 
    raw_eeg = reconstruct_raw(raw_eeg)

    IC = ica.get_sources(raw_eeg)
    
    #raw = mne.io.RawArray(raw.get_data(), mne.create_info(raw.ch_names, Fs))

    if filter is not None:
        sos = scipy.signal.butter(2,np.array(filter)/(Fs/2), btype = 'bandpass', output='sos')
        raw_eog.apply_function(apply_sosfilter, picks = 'all', n_jobs = -1, channel_wise = True, sos = sos, zero_phase = True)

    scores = list()
    indices = list()
    for ch in raw_eog.ch_names:
        data_eog = raw_eog.get_data(picks = ch)

        score = list() 
        for idx, ic in enumerate(IC.ch_names):
            #data_ic = ica.get_data(picks = ic)
        
            a = scipy.stats.pearsonr(x = np.squeeze(data_eog), y = np.squeeze(IC.get_data(picks = ic)))

            score.append(a[0])
            
        if threshold == 'max':
            I = np.argmax(np.absolute(np.array(score)))
            indices.append(I)
        else:
            I = np.where(np.absolute(np.array(score)) >= threshold)
            indices += I[0].tolist()
                
        scores.append(score)

    scores = np.array(scores)
    
    return scores, indices
    

def ica_from_vhdr(files,
                  picks = ['eeg'],
                  eog_channels = ['vEOG', 'hEOG'],
                  l_freq = 1.0,
                  len_transition = 1.0,
                  n_components = 15,
                  max_iter = "auto",
                  random_state = 42,
                  save = False,
                  overwrite = False):
    
    raw = mne.io.read_raw_brainvision(vhdr_fname = files[0], preload=False)
    Fs = raw.info['sfreq']
    sos = scipy.signal.butter(2, np.array(l_freq)/(Fs/2), btype = 'highpass', output='sos')

    raw = concatenate_raws_vhdr(files=files, sos=sos, eog_channels=eog_channels, len_transition=len_transition, save = False) 

    ica = mne.preprocessing.ICA(n_components=n_components, max_iter=max_iter, random_state=random_state)
    ica.fit(raw.copy().pick(picks = picks))

    if save is not False:
        ica.save(fname = save, overwrite = overwrite)
        
    return ica