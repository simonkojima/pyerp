import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from mne import BaseEpochs, concatenate_epochs, epochs    

def classify_binary(epochs,
                    clf,
                    ivals,
                    tags=None,
                    cv = 3,
                    perm = True,
                    n_perm = 100,
                    scoring = 'balanced_accuracy',
                    n_jobs = None,
                    verbose = 0):
    """
    Parameters
    ==========
    """
    from ..utils.analysis import get_binary_epochs
    from sklearn.model_selection import cross_val_score, permutation_test_score

    r = dict()
    r['n_channels'] = len(epochs.ch_names)

    X, Y = get_binary_epochs(epochs, tags) 
    
    vectorizer = EpochsVectorizer(ivals=ivals)
    X = vectorizer.transform(X)

    r['n_features'] = X.shape[1]
    r['n_samples'] = dict()

    for y in np.unique(Y):
        r['n_samples'][str(y)] = np.count_nonzero(Y == y)

    scores = cross_val_score(estimator = clf,
                            X = X,
                            y = Y,
                            scoring = scoring,
                            cv = cv,
                            n_jobs = n_jobs,
                            verbose = verbose)
    r['scores'] = scores
    if perm:
        score, permutation_scores, pvalue = permutation_test_score(estimator = clf,
                                                                X = X,
                                                                y = Y,
                                                                cv = cv,
                                                                n_permutations = n_perm,
                                                                n_jobs = n_jobs,
                                                                verbose = verbose,
                                                                scoring = scoring)
        r['score'] = score
        r['permutation_scores'] = permutation_scores
        r['pvalue'] = pvalue
        r['n_permutations'] = n_perm
    r['cv'] = cv
    r['scoring'] = scoring
    try:
        r['estimator'] = str(clf)
    except:
        r['could not be parsed.']

    return r

class EpochsVectorizer():
    def __init__(self,
                 ivals,
                 channel_prime = True,
                 type = 'mne',
                 tmin = None,
                 tmax = None,
                 include_tmax = True,
                 fs = None):
        self.ivals = ivals
        self.channel_prime = channel_prime
        self.type = type
        self.tmin = tmin
        self.tmax = tmax
        self.include_tmax = include_tmax
        self.fs = fs
        
        """
        
        type: 'mne', 'ndarray'
        
        channel_prime:
        if True,  vec = [ch1_t1, ch2_t1, ch3_t1, ch1_t2, ch2_t2, ch3_t2, ...]
        if False, vec = [ch1_t1, ch1_t2, ch1_t3, ch2_t1, ch2_t2, ch2_t3, ...]


        """

    def fit(self, X, y=None):
        """fit."""
        return self

    def transform(self, X):
        
        if self.type == 'ndarray':
            tmin = self.tmin
            tmax = self.tmax
            fs = self.fs
            data = X
            if self.include_tmax:
                times = np.linspace(start=tmin, stop=tmax, num=int((tmax-tmin)*fs)+1)
            else:
                times = np.linspace(start=tmin, stop=tmax, num=int((tmax-tmin)*fs))
            if data.shape[2] != times.size:
                raise ValueError("number of time samples is invalid.")
        elif self.type == 'mne':
            tmin = X.times[0]
            tmax = X.times[-1]
            data = X.get_data(copy = True)
        else:
            raise ValueError("Unknown type: %s"%str(self.type))
        
        
        if np.min(np.array(self.ivals).flatten()) < tmin:
            raise ValueError("minimum time value of ivals belows tmin of epochs.")
        
        if np.max(np.array(self.ivals).flatten()) > tmax:
            raise ValueError("maximum time value of ivals exceeds tmax of epochs.")
        
        vec = np.zeros((data.shape[0], data.shape[1], len(self.ivals)))

        for m, ival in enumerate(self.ivals):
            if self.type == 'ndarray':
                idx = list()
                for t in ival:
                    I = np.argmin(np.absolute(times - t))
                    idx.append(I)
            elif self.type == 'mne':
                idx = X.time_as_index(ival)
            idx = list(range(idx[0], idx[1]+1))
            vec[:, :, m] = np.mean(data[:, :, idx], axis=2)

        if self.channel_prime:
            vec = np.reshape(vec, (vec.shape[0], -1), order='F')
        else:
            vec = np.reshape(vec, (vec.shape[0], -1), order='C')
            
        return vec 
        
        

class _EpochsVectorizer(BaseEstimator, TransformerMixin):

    """
        Original code of this class by implemented by Jan Sosulski. Modified by Simon Kojima.

        --
        https://github.com/jsosulski/toeplitzlda
        Copyright (c) 2022 Jan Sosulski
        All rights reserved.

        Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
        * Neither the name of Jan Sosulski nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

        NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    def __init__(
        self,
        sfreq = None,
        t_ref = None,
        permute_channels_and_time=True,
        select_ival=None,
        jumping_mean_ivals=None,
        averaging_samples=None,
        rescale_to_uv=False,
        mne_scaler=None,
        pool_times=False,
        to_np_only=False,
        copy=True
    ):

        self.sfreq = sfreq
        self.t_ref = t_ref
        self.permute_channels_and_time = permute_channels_and_time
        self.jumping_mean_ivals = jumping_mean_ivals
        self.select_ival = select_ival
        self.averaging_samples = averaging_samples
        self.rescale_to_uv = rescale_to_uv
        self.scaling = 1e6 if self.rescale_to_uv else 1
        self.pool_times = pool_times
        self.to_np_only = to_np_only
        self.copy = copy
        self.mne_scaler = mne_scaler
        if self.select_ival is None and self.jumping_mean_ivals is None:
            raise ValueError("jumping_mean_ivals or select_ival is required")

    def fit(self, X, y=None):
        """fit."""
        return self

    def transform(self, X):
        """transform."""
        e = X.copy() if self.copy else X
        type_e = type(e)
        if type_e is epochs.EpochsFIF:
            type_e = BaseEpochs
        elif type_e is epochs.Epochs:
            type_e = BaseEpochs
        elif type_e is epochs.EpochsArray:
            type_e = BaseEpochs

        if type_e is not BaseEpochs and type_e is not np.ndarray:
            if type_e is list and type(e[0]) is BaseEpochs:
                e = concatenate_epochs(e, add_offset=False) # Is it ok to fix this to add_offset=Fasle ?
            else:
                raise ValueError("argument X has unknown type : " + str(type_e))
        if self.to_np_only:
            if type_e is BaseEpochs:
                X = e.get_data() * self.scaling
                return X
            else:
                raise ValueError("argument X is already np_array")
        if self.jumping_mean_ivals is not None:
            self.averaging_samples = np.zeros(len(self.jumping_mean_ivals))
            if type_e is BaseEpochs:
                X = e.get_data() * self.scaling
            else:
                X = e * self.scaling
                if self.sfreq is None:
                    raise ValueError("specify the sampling frequency")
                if self.t_ref is None:
                    raise ValueError("specify the time reference")
            new_X = np.zeros((X.shape[0], X.shape[1], len(self.jumping_mean_ivals)))
            for i, ival in enumerate(self.jumping_mean_ivals):
                if type_e is BaseEpochs:
                    np_idx = e.time_as_index(ival)
                else:
                    np_idx = time_as_index(ival, self.t_ref, self.sfreq)
                idx = list(range(np_idx[0], np_idx[1]))
                self.averaging_samples[i] = len(idx)
                new_X[:, :, i] = np.mean(X[:, :, idx], axis=2)
            X = new_X
        elif self.select_ival is not None:
            if type_e is BaseEpochs:
                e.crop(tmin=self.select_ival[0], tmax=self.select_ival[1])
                X = e.get_data() * self.scaling
            else:
                if self.sfreq is None:
                    raise ValueError("specify the sampling frequency")
                if self.t_ref is None:
                    raise ValueError("specify the time reference")
                X = e * self.scaling
                t_idx = time_as_index(self.select_ival, self.t_ref, self.sfreq)
                X = X[:,:,t_idx[0]:(t_idx[1]+1)] # t_idx[1]+1 to be same as e.crop()
        elif self.pool_times:
            if type_e is BaseEpochs:
                X = e.get_data() * self.scaling
            else:
                X = e * self.scaling
            raise ValueError("This should never be entered though.")
        else:
            raise ValueError(
                "In the constructor, pass either select ival or jumping means."
            )
        if self.mne_scaler is not None:
            X = self.mne_scaler.fit_transform(X)
        if self.permute_channels_and_time and not self.pool_times:
            X = X.transpose((0, 2, 1))
        if self.pool_times:
            X = np.reshape(X, (-1, X.shape[1]))
        else:
            X = np.reshape(X, (X.shape[0], -1))
        return X

def time_as_index(times, t_ref, sfreq, use_rounding=False):

    """Convert time to indices.
    Parameters
    ----------
    times : list-like | float | int
        List of numbers or a number representing points in time.
    t_ref : float | int
        time reference to compute index of each time point.
        usually, it is the first time index of concerned data.
    sfreq : float | int
        sampling frequency of concerned data.
    use_rounding : bool
        If True, use rounding (instead of truncation) when converting
        times to indices. This can help avoid non-unique indices.
    Returns
    -------
    index : ndarray
        Indices corresponding to the times supplied.

    ---
    This function is originally from MNE-Python project.
    Modified by Simon Kojima

    Copyright © 2011-2022, authors of MNE-Python
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    index = (np.atleast_1d(times) - t_ref) * sfreq
    if use_rounding:
        index = np.round(index)
    return index.astype(int)