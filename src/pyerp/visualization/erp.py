import numpy as np
import matplotlib.pyplot as plt

def plot_erp_general(tmin, tmax, fontsize = 12):
    plt.xlim(tmin, tmax)
    plt.xlabel("Time (s)", fontsize=fontsize)
    plt.ylabel('Potential ($\mu$V)', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)

def plot_2ch_tnt(epochs, picks = ['Cz', 'F3'], reject = None, tags = None, linewidth = 2, sns = True):
    from ..utils.analysis import get_binary_epochs
    X, _ = get_binary_epochs(epochs, tags)
    if sns:
        import seaborn as sns
        sns.set()
    times = epochs.times
    fig = plt.figure()
    colors = ['tab:orange', 'tab:blue']
    N_t = X['target'].__len__()
    N_nt = X['nontarget'] .__len__()
    for idx, ch in enumerate(picks):
        plt.plot(times,
                np.squeeze(X['target'].average().get_data(picks=[ch], units='uV')),
                color = colors[idx],
                linestyle = '-',
                linewidth = linewidth,
                label="target(%s,N=%d)" %(ch, N_t))
        plt.plot(times,
                np.squeeze(X['nontarget'].average().get_data(picks=[ch], units='uV')),
                color = colors[idx],
                linestyle = '--',
                linewidth = linewidth,
                label = "nontarget(%s,N=%d)" %(ch, N_nt))
    plot_erp_general(X.tmin, X.tmax)
    return fig

def plot_tnt(epochs, picks = ['Cz'], tags=None, linewidth=2, sns=True):
    """
    Parameters
    ==========
    tags : list of str, default = None, e.g. tags = ['event:stim1', 'task:count']
    """
    
    from ..utils.analysis import get_binary_epochs
    X, _ = get_binary_epochs(epochs, tags)
    
    if sns:
        import seaborn as sns
        sns.set()

    N_t = X['target'].__len__()
    N_nt = X['nontarget'] .__len__()

    times = epochs.times
    figs = list()
    for ch in picks:
        figs.append(plt.figure())
        plt.title(ch)
        plt.plot(times,
                np.squeeze(X['target'].average().get_data(picks=[ch], units='uV')),
                color = 'tab:orange',
                linewidth=linewidth,
                label='target(N=%d)'%N_t)
        plt.plot(times,
                np.squeeze(X['nontarget'].average().get_data(picks=[ch], units='uV')),
                color = 'tab:blue',
                linewidth=linewidth,
                label='nontarget(N=%d)'%N_nt)
        plot_erp_general(X.tmin, X.tmax)
    return figs
