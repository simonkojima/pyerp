import numpy as np
import scipy
import matplotlib.pyplot as plt

def apply_sosfilter(data, sos, zero_phase = True):
    # channel_wise should be set True, when it's called from mne_instance.apply_function().
    if zero_phase:
        r = scipy.signal.sosfiltfilt(sos, data)
    else:
        r = scipy.signal.sosfilt(sos, data)
    return r

def apply_filter(data, b, a, zero_phase = True):
    # channel_wise should be set True, when it's called from mne_instance.apply_function().
    if zero_phase:
        r = scipy.signal.filtfilt(b = b, a = a, x = data)
    else:
        r = scipy.signal.lfilter(b = b, a = a, x = data)
    return r

def plot_freqz(sos=None, b=None, a=None, fs = None, xlim = None, show = True):
    fig = plt.figure()

    if sos is not None: 
        w,h = scipy.signal.sosfreqz(sos)
    else:
        w,h = scipy.signal.freqz(b = b, a = a)
    
    # frequency response
    plt.subplot(2, 1, 1)
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    
    if fs is None:
        plt.plot(w/np.pi, db)
    else:
        plt.plot(w/np.pi*fs/2, db)
    plt.ylim(-75, 5)
    plt.grid(True)
    plt.yticks([0, -20, -40, -60])
    plt.ylabel('Gain [dB]')
    plt.title('Frequency Response')
    if xlim is not None:
        plt.xlim(xlim)
        
    # phase characteristic
    plt.subplot(2, 1, 2)
    if fs is None:
        plt.plot(w/np.pi, np.angle(h))
    else:
        plt.plot(w/np.pi*fs/2, np.angle(h))
    plt.grid(True)
    plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.ylabel('Phase [rad]')
    if fs is None:
        plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    else:
        plt.xlabel('Frequency (Hz)')
    if xlim is not None:
        plt.xlim(xlim)
    
    if show:
        plt.show()
    return fig

def plot_response(sos = None, b = None, a = None, length = 1000, fs = None, xlim = None, ylim = None, show = True):
    
    # impulse response
    imp = scipy.signal.unit_impulse(length, 'mid')
    
    if sos is not None:
        response = scipy.signal.sosfilt(sos, imp)
        response_zero = scipy.signal.sosfiltfilt(sos, imp)
    else:
        response = scipy.signal.lfilter(b = b, a = a, x = imp)
        response_zero = scipy.signal.filtfilt(b = b, a = a, x = imp)

    if fs is None:
        x = np.arange(-length/2, length/2)
    else:
        x = np.arange(-length/2, length/2)/fs
    
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x, imp)
    plt.plot(x, response)


    plt.subplot(2, 2, 2)
    plt.plot(x, imp)
    plt.plot(x, response_zero)
    

    # heaviside response
    step = np.heaviside(np.arange(-length/2,length/2), 1)
    
    if sos is not None:
        response = scipy.signal.sosfilt(sos, step)
        response_zero = scipy.signal.sosfiltfilt(sos, step)
    else:
        response = scipy.signal.lfilter(b = b, a = a, x = step)
        response_zero = scipy.signal.filtfilt(b = b, a = a, x = step)

    plt.subplot(2, 2, 3)
    plt.plot(x, step)
    plt.plot(x, response)

    plt.subplot(2, 2, 4)
    plt.plot(x, step)
    plt.plot(x, response_zero)

    
    if xlim is not None:
        if type(xlim[0]) == list:
            plt.subplot(2,2,1)
            plt.xlim(xlim[0])
            plt.subplot(2,2,2)
            plt.xlim(xlim[0])
            plt.subplot(2,2,3)
            plt.xlim(xlim[1])
            plt.subplot(2,2,4)
            plt.xlim(xlim[1])
        else:
            for m in range(1, 5):
                plt.subplot(2,2,m)
                plt.xlim(xlim)

    if ylim is not None:
        if type(ylim[0]) == list:
            plt.subplot(2,2,1)
            plt.ylim(ylim[0])
            plt.subplot(2,2,2)
            plt.ylim(ylim[0])
            plt.subplot(2,2,3)
            plt.ylim(ylim[1])
            plt.subplot(2,2,4)
            plt.ylim(ylim[1])
        else:
            for m in range(1,5):
                plt.subplot(2,2,m)
                plt.ylim(ylim)

    plt.subplot(2,2,1)
    plt.title("one pass")

    plt.subplot(2,2,2)
    plt.title("forward backward")

    plt.subplot(2,2,3)
    if fs is None:
        plt.xlabel('Sample')
    else:
        plt.xlabel('Seconds')
    plt.ylabel('Amplitude')
    
    if show:
        plt.show()
    
    return fig