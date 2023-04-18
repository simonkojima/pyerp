def apply_tukey_window(data, fs, L, t_w):
    import scipy
    w = scipy.signal.windows.tukey(M = L, alpha = (2*t_w*fs)/L)
    data = data*w
    return data

    