import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_butterfly(epochs, picks = None, figsize=[6.4, 4.8]):
    times = epochs.times
    
    epochs = epochs.copy()

    if picks is not None:
        epochs.pick(picks = picks)
        
    
    data = epochs.get_data(units = 'uV')
    data = np.squeeze(data)

    if data.ndim == 3:
        raise ValueError("select single channel.")

    fig = plt.figure(figsize=figsize)
    for m in range(data.shape[1]):
        plt.plot(times, data[m,:], color = 'tab:gray')
    
    plt.plot(times, np.mean(data, axis=0), color = 'tab:orange')
    
    return fig

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

def plot_evo_2ch_tnt(target, nontarget, picks = ['Cz', 'F3'], vlim = None, figsize = [6.4, 4.8], linewidth = 2, fontsize=12, legend_loc = 'best', sns = True):
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
    plt.ylabel('Potential ($\mu$V)', fontsize=fontsize)
    plt.xlabel('Time (sec)', fontsize=fontsize)
    handles, labels = plt.gca().get_legend_handles_labels()
    if baseline is not None:
        order = make_label_last(labels, 'baseline')
        legend_in_order(handles, labels, order, fontsize=fontsize, loc=legend_loc)
    else:
        plt.legend(loc=legend_loc, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)

    return fig

def plot_2ch_squared_r(target, nontarget, channels, r2, times, fs, baseline = None, mesh = None, N = None, vlim = None, xticks = None, maplimits = None, figsize = [6.4, 4.8], linewidth = 2, fontsize=12, fontsize_legend = 12, legend_loc = 'best', sns = True):
    """
    Parameters
    ==========
    
    target: ndarray, shape of (2, n_samples)
        target[0]: channel 1
        target[1]: channel 2
        
    nontarget: ndarray, shape of (2, n_samples)
        nontarget[0]: channel 1
        nontarget[1]: channel 2

    channels: channel labels, array-like

    r2: signed-r^2 value, ndarray
        shape of (2, n_samples)
        r2[0]: channel 1
        r2[1]: channel 2

    times: time samples

    fs: sampling frequency

    baseline: baseline range in time
    
    mesh: time ranges to plot mesh., list has length of number of mesh
        mesh[0] = [t0, t1]: area between time [t0, t1] will be meshed.
    
    mesh_colors

    N: number of samples to obtain the averaged responses
        N[0][0]: ch1, target
        N[0][1]: ch1, nontarget
        N[1][0]: ch2, target
        N[1][1]: ch2, nontarget
    
    """
    if sns:
        import seaborn as sns
        sns.set()
        
    params = dict()

    fig = plt.figure(figsize=figsize)
    colors = ['tab:orange', 'tab:blue']
    
    gs = gridspec.GridSpec(2,2, height_ratios=[10,1], width_ratios=[10,0.5])

    ax0 = plt.subplot(gs[0,0])
    if baseline is not None:
        plt.plot(baseline, [0,0], color = 'tab:gray', linestyle = '-', linewidth = linewidth, label='baseline')

    for idx, ch in enumerate(channels):
        label = [None, None]
        if N is None:
            label[0] = "target(%s)"%ch
            label[1] = "nontarget(%s)"%ch
        else:
            label[0] = "target(%s,N=%d)"%(ch, N[idx][0])
            label[1] = "nontarget(%s,N=%d)"%(ch, N[idx][1])
        ax0.plot(times,
                target[idx],
                color = colors[idx],
                linestyle = '-',
                linewidth = linewidth,
                label=label[0])
        ax0.plot(times,
                nontarget[idx],
                color = colors[idx],
                linestyle = '--',
                linewidth = linewidth,
                label = label[1]) 

    if xticks is not None:
        ax0.set_xticks(xticks)
    plt.setp(ax0.get_xticklabels(), visible = False)
    plt.xlim(times[0], times[-1])
    plt.ylabel('Potential ($\mu$V)', fontsize=fontsize)
    if baseline is not None:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = make_label_last(labels, 'baseline')
        legend_in_order(handles, labels, order, fontsize=fontsize_legend, loc=legend_loc)
    else:
        plt.legend(fontsize=fontsize_legend, loc=legend_loc)
    plt.tick_params(labelsize=fontsize)
    if vlim is not None:
        plt.ylim(vlim[0], vlim[1])
    else:
        plt.ylim(plt.ylim())
    params['ylim'] = plt.ylim()
    if mesh is not None:
        #from matplotlib.transforms import Bbox
        ylim = plt.ylim()
        for t in mesh:
            x = [t[0], t[0], t[1], t[1], t[0]]
            y = [ylim[0], ylim[1], ylim[1], ylim[0], ylim[0]]
            ax0.fill(x, y, color = t[2], alpha = 0.2, linestyle = 'None')

    #r2 = signed_square_r(X['target'].get_data(picks=picks, units='uV'),
    #                     X['nontarget'].get_data(picks=picks, units='uV'))
    ax1 = plt.subplot(gs[1,0], sharex = ax0)
    if maplimits is None:
        maplimits = [-0.03, 0.03]
    elif maplimits == 'auto':
        _max = np.max(np.absolute(r2))
        maplimits = [-1*_max, _max]
    params['maplimits'] = maplimits
    
    pc = ax1.pcolormesh(np.atleast_2d(np.append(times, times[-1]+1/fs)), [2,1,0], r2, cmap='seismic', vmin=maplimits[0], vmax=maplimits[1])
    plt.setp(ax1.get_yticklabels(), visible = False)
    plt.ylabel('\n'.join(channels), rotation = 'horizontal', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
    #plt.ylabel("abc\ndef", rotation = 'horizontal', horizontalalignment='right', verticalalignment='center')
    plt.xlabel("Time (s)", fontsize=fontsize)
    if xticks is not None:
        ax1.set_xticks(xticks)
    plt.tick_params(labelsize=fontsize)

    axes = plt.subplot(gs[0:2,1])
    cbar = plt.colorbar(pc, cax = axes)
    cbar.set_ticks([maplimits[0], 0, maplimits[1]])
    cbar.ax.tick_params(labelsize=fontsize)
    #plt.colorbar().a

    plt.subplots_adjust(hspace=.0)
    plt.subplots_adjust(wspace=.02)

    return fig, params

def plot_2ch_tnt(epochs, picks = ['Cz', 'F3'], tags = None, vlim = None, xlim = None, maplimits = None, figsize = [6.4, 4.8], linewidth = 2, fontsize=12, fontsize_legend = 12, legend_loc = 'best', sns = True):
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
    if xlim is None:
        plt.xlim(X.tmin, X.tmax)
    else:
        plt.xlim(xlim)
    if vlim is not None:
        plt.ylim(vlim[0], vlim[1])
    plt.ylabel('Potential ($\mu$V)', fontsize=fontsize)
    if baseline is not None:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = make_label_last(labels, 'baseline')
        legend_in_order(handles, labels, order, fontsize=fontsize_legend, loc=legend_loc)
    else:
        plt.legend(fontsize=fontsize_legend, loc=legend_loc)
    plt.tick_params(labelsize=fontsize)

    r2 = signed_square_r(X['target'].get_data(picks=picks, units='uV'),
                         X['nontarget'].get_data(picks=picks, units='uV'))
    ax1 = plt.subplot(gs[1,0], sharex = ax0)
    if maplimits is None:
        maplimits = [-0.03, 0.03]
    pc = ax1.pcolormesh(np.atleast_2d(np.append(times, times[-1]+1/fs)), [2,1,0], r2, cmap='seismic', vmin=maplimits[0], vmax=maplimits[1])
    plt.setp(ax1.get_yticklabels(), visible = False)
    plt.ylabel('\n'.join(picks), rotation = 'horizontal', horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
    #plt.ylabel("abc\ndef", rotation = 'horizontal', horizontalalignment='right', verticalalignment='center')
    plt.xlabel("Time (s)", fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)

    axes = plt.subplot(gs[0:2,1])
    cbar = plt.colorbar(pc, cax = axes)
    cbar.set_ticks([maplimits[0], 0, maplimits[1]])
    cbar.ax.tick_params(labelsize=fontsize)
    #plt.colorbar().a

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
