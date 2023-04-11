import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_erp_general(tmin, tmax, fontsize = 12, legend_loc = 'best'):
    plt.xlim(tmin, tmax)
    plt.xlabel("Time (s)", fontsize=fontsize)
    plt.ylabel('Potential ($\mu$V)', fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc=legend_loc)
    plt.tick_params(labelsize=fontsize)

def make_label_last(labels, label_last):
    """
    change order of items in legend.
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = make_label_last(labels, 'baseline')
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    """
    order = list()
    for idx, label in enumerate(labels):
        if label == label_last:
            idx_target_label = idx
        else:
            order.append(idx)
    order.append(idx_target_label)

    return order

def legend_in_order(handles, labels, order, loc='best', fontsize='medium'):
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=loc, fontsize=fontsize)

def plot_evo_2ch_tnt(target, nontarget, picks = ['Cz', 'F3'], vlim = None, figsize = [6.4, 4.8], linewidth = 2, legend_loc = 'best', sns = True):
    if sns:
        import seaborn as sns
        sns.set()
    times = target.times
    baseline = target.baseline
    fig = plt.figure(figsize=figsize)
    colors = ['tab:orange', 'tab:blue']
    
    if baseline is not None:
        plt.plot(baseline, [0,0], color = 'tab:gray', linestyle = '-', linewidth = linewidth, label = 'baseline')

    for idx, ch in enumerate(picks):
        plt.plot(times,
                 np.squeeze(target.get_data(picks=[ch], units='uV')),
                 color = colors[idx],
                 linestyle = '-',
                 linewidth = linewidth,
                 label="target(%s)"%ch)
        plt.plot(times,
                 np.squeeze(nontarget.get_data(picks=[ch], units='uV')),
                 color = colors[idx],
                 linestyle = '--',
                 linewidth = linewidth,
                 label="nontarget(%s)"%ch)
    
    plt.xlim(target.tmin, target.tmax)
    if vlim is not None:
        plt.ylim(vlim[0], vlim[1])
    plt.ylabel('Potential ($\mu$V)', fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    if baseline is not None:
        order = make_label_last(labels, 'baseline')
        legend_in_order(handles, labels, order, fontsize=12, loc=legend_loc)
    else:
        plt.legend(loc=legend_loc, fontsize=12)
    plt.tick_params(labelsize=12)

    return fig

def plot_2ch_tnt(epochs, picks = ['Cz', 'F3'], tags = None, vlim = None, figsize = [6.4, 4.8], linewidth = 2, legend_loc = 'best', sns = True):
    from ..utils.analysis import get_binary_epochs
    from ..analysis.erp import signed_square_r

    X, _ = get_binary_epochs(epochs, tags)

    if sns:
        import seaborn as sns
        sns.set()
    times = epochs.times
    baseline = epochs.baseline
    fs = epochs.info['sfreq']
    fig = plt.figure(figsize=figsize)
    colors = ['tab:orange', 'tab:blue']
    N_t = X['target'].__len__()
    N_nt = X['nontarget'] .__len__()
    
    gs = gridspec.GridSpec(2,2, height_ratios=[10,1], width_ratios=[10,0.5])

    ax0 = plt.subplot(gs[0,0])
    if baseline is not None:
        plt.plot(baseline, [0,0], color = 'tab:gray', linestyle = '-', linewidth = linewidth, label='baseline')
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
    if vlim is not None:
        plt.ylim(vlim[0], vlim[1])
    plt.ylabel('Potential ($\mu$V)', fontsize=12)
    if baseline is not None:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = make_label_last(labels, 'baseline')
        legend_in_order(handles, labels, order, fontsize=12, loc=legend_loc)
    else:
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

def plot_tnt(epochs, picks = ['Cz'], tags=None, vlim = None, figsize = [6.4, 4.8], linewidth=2, legend_loc = 'best', sns=True):
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
    baseline = epochs.baseline
    figs = list()
    for ch in picks:
        r2 = signed_square_r(X['target'].get_data(picks=[ch], units='uV'),
                             X['nontarget'].get_data(picks=[ch], units='uV'))

        figs.append(plt.figure(figsize=figsize))

        gs = gridspec.GridSpec(2,2, height_ratios=[10,1], width_ratios=[10,0.5])

        ax0 = plt.subplot(gs[0,0])
        if baseline is not None:
            plt.plot(baseline, [0,0], color = 'tab:gray', linestyle = '-', linewidth = linewidth, label='baseline')
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
        if vlim is not None:
            plt.ylim(vlim[0], vlim[1])
        plt.ylabel('Potential ($\mu$V)', fontsize=12)
        plt.tick_params(labelsize=12)
        if baseline is not None:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = make_label_last(labels, 'baseline')
            legend_in_order(handles, labels, order, fontsize=12, loc=legend_loc)
        else:
            plt.legend(fontsize=12, loc=legend_loc)

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
