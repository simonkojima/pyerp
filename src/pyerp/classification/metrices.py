import numpy as np

def calc_itr(n_classes, accuracy, trial_duration=None):
    """
    compute ITR (Information Transfer Rate).

    Parameters
    ==========
    n_classes : int
        Number of classes of bci system
    accuracy : float
        Classification accuracy
    trial_duration : float, default=None
        Time duration of single trial in minutes. If it's NOT None, ITR will be returned in unit of (bits/min).
        Otherwise, it will be (bit/trial). Default value is None.    

    References
    ==========
    [1] Jonathan R. Wolpaw, Niels Birbaumer, et al., 
        Brain–Computer Interface Technology: A Review of the First International Meeting, 
        IEEE TRANSACTIONS ON REHABILITATION ENGINEERING, VOL. 8, NO. 2, JUNE 2000
    """
    N = n_classes
    P = accuracy

    log2 = np.log2

    if P == 1:
        itr = log2(N)
    elif P == 0:
        itr = log2(N) + (1-P)*log2((1-P)/(N-1))
    else:
        itr = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))

    if trial_duration is not None:
        return itr / trial_duration
    else:
        return itr