import numpy as np
import matplotlib.pyplot as plt

def plot_erp_general(tmin, tmax, fontsize = 12):
    plt.xlim(tmin, tmax)
    plt.xlabel("Time (s)", fontsize=fontsize)
    plt.ylabel('Potential ($\mu$V)', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)

def plot_tnt(epochs, picks = ['Cz'], tags=None, linewidth=2, sns=True):
    """
    Parameters
    ==========
    tags : list of str, default = None, e.g. tags = ['event:stim1', 'task:count']
    """
    if tags is not None:
        tags_target = tags.copy()        
        tags_target.append('target')

        tags_nontarget = tags.copy()
        tags_nontarget.append('nontarget')

        tags_target = '/'.join(tags_target)
        tags_nontarget = '/'.join(tags_nontarget)
    else:
        tags_target = 'target'
        tags_nontarget = 'nontarget'

    target = epochs[tags_target]
    nontarget = epochs[tags_nontarget]

    target_evoked = target.average()
    nontarget_evoked = nontarget.average()

    if sns:
        import seaborn as sns
        sns.set()

    times = epochs.times
    for ch in picks:
        plt.figure()
        plt.title(ch)
        plt.plot(times,
                np.squeeze(target_evoked.get_data(picks=[ch], units='uV')),
                color = 'tab:orange',
                linewidth=linewidth,
                label='target')
        plt.plot(times,
                np.squeeze(nontarget_evoked.get_data(picks=[ch], units='uV')),
                color = 'tab:blue',
                linewidth=linewidth,
                label='nontarget')
        plot_erp_general(target.tmin, target.tmax)
    plt.show()

