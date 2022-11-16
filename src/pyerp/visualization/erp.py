import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_erp_general(tmin, tmax, fontsize = 12, legend_loc = 'best'):
    plt.xlim(tmin, tmax)
    plt.xlabel("Time (s)", fontsize=fontsize)
    plt.ylabel('Potential ($\mu$V)', fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc=legend_loc)
    plt.tick_params(labelsize=fontsize)

def plot_2ch_tnt(epochs, picks = ['Cz', 'F3'], reject = None, tags = None, figsize = [6.4, 4.8], linewidth = 2, legend_loc = 'best', sns = True):
    from ..utils.analysis import get_binary_epochs
    from ..analysis.erp import signed_square_r

    X, _ = get_binary_epochs(epochs, tags)
    if sns:
        import seaborn as sns
        sns.set()
    times = epochs.times
    fs = epochs.info['sfreq']
    fig = plt.figure(figsize=figsize)
    colors = ['tab:orange', 'tab:blue']
    N_t = X['target'].__len__()
    N_nt = X['nontarget'] .__len__()
    
    gs = gridspec.GridSpec(2,2, height_ratios=[10,1], width_ratios=[10,0.5])

    ax0 = plt.subplot(gs[0,0])
    for idx, ch in enumerate(picks):
        ax0.plot(times,
                np.squeeze(X['target'].average().get_data(picks=[ch], units='uV')),
                color = colors[idx],
                linestyle = '-',
                linewidth = linewidth,
                label="target(%s,N=%d)" %(ch, N_t))
        ax0.plot(times,
                np.squeeze(X['nontarget'].average().get_data(picks=[ch], units='uV')),
                color = colors[idx],
                linestyle = '--',
                linewidth = linewidth,
                label = "nontarget(%s,N=%d)" %(ch, N_nt)) 
    plt.setp(ax0.get_xticklabels(), visible = False)
    plt.xlim(X.tmin, X.tmax)
    plt.ylabel('Potential ($\mu$V)', fontsize=12)
    plt.legend(fontsize=12, loc=legend_loc)
    plt.tick_params(labelsize=12)

    r2 = signed_square_r(X['target'].get_data(picks=picks, units='uV'),
                         X['nontarget'].get_data(picks=picks, units='uV'))
    ax1 = plt.subplot(gs[1,0], sharex = ax0)
    pc = ax1.pcolormesh(np.atleast_2d(np.append(times, times[-1]+1/fs)), [2,1,0], r2, cmap='seismic', vmin=-0.03, vmax=0.03)
    plt.setp(ax1.get_yticklabels(), visible = False)
    plt.ylabel('\n'.join(picks), rotation = 'horizontal', horizontalalignment='right', verticalalignment='center')
    #plt.ylabel("abc\ndef", rotation = 'horizontal', horizontalalignment='right', verticalalignment='center')
    plt.xlabel("Time (s)", fontsize=12)
    plt.tick_params(labelsize=12)

    axes = plt.subplot(gs[0:2,1])
    plt.colorbar(pc, cax = axes)

    plt.subplots_adjust(hspace=.0)
    plt.subplots_adjust(wspace=.02)

    return fig

def plot_tnt(epochs, picks = ['Cz'], tags=None, figsize = [6.4, 4.8], linewidth=2, legend_loc = 'best', sns=True):
    """
    Parameters
    ==========
    tags : list of str, default = None, e.g. tags = ['event:stim1', 'task:count']
    """
    
    from ..utils.analysis import get_binary_epochs
    from ..analysis.erp import signed_square_r
    X, _ = get_binary_epochs(epochs, tags)
    
    if sns:
        import seaborn as sns
        sns.set()

    N_t = X['target'].__len__()
    N_nt = X['nontarget'] .__len__()


    fs = epochs.info['sfreq']
    times = epochs.times
    figs = list()
    for ch in picks:
        r2 = signed_square_r(X['target'].get_data(picks=[ch], units='uV'),
                             X['nontarget'].get_data(picks=[ch], units='uV'))

        figs.append(plt.figure(figsize=figsize))

        gs = gridspec.GridSpec(2,2, height_ratios=[10,1], width_ratios=[10,0.5])

        ax0 = plt.subplot(gs[0,0])
        plt.title(ch)
        ax0.plot(times,
                np.squeeze(X['target'].average().get_data(picks=[ch], units='uV')),
                color = 'tab:orange',
                linewidth=linewidth,
                label='target(N=%d)'%N_t)
        ax0.plot(times,
                np.squeeze(X['nontarget'].average().get_data(picks=[ch], units='uV')),
                color = 'tab:blue',
                linewidth=linewidth,
                label='nontarget(N=%d)'%N_nt)
        plt.setp(ax0.get_xticklabels(), visible = False)
        plt.xlim(X.tmin, X.tmax)
        plt.ylabel('Potential ($\mu$V)', fontsize=12)
        plt.legend(fontsize=12, loc=legend_loc)
        plt.tick_params(labelsize=12)

        ax1 = plt.subplot(gs[1,0], sharex = ax0)
        pc = ax1.pcolormesh(np.atleast_2d(np.append(times, times[-1]+1/fs)), range(2), r2, cmap='seismic', vmin=-0.03, vmax=0.03)
        plt.setp(ax1.get_yticklabels(), visible = False)
        plt.ylabel(ch, rotation = 'horizontal', horizontalalignment='right', verticalalignment='center')
        plt.xlabel("Time (s)", fontsize=12)
        plt.tick_params(labelsize=12)

        axes = plt.subplot(gs[0:2,1])
        plt.colorbar(pc, cax = axes)

        plt.subplots_adjust(hspace=.0)
        plt.subplots_adjust(wspace=.02)

    return figs
