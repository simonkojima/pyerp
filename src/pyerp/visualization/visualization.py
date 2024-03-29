import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    enable_seaborn = True
except:
    enable_seaborn = False
    print("Warning: seaborn is not installed")

def barplot(data, 
            err=None,
            avg_parents=True,
            avg_children=True,
            parents=None,
            children=None,
            color=None,
            width=None,
            ylabel=None,
            xlabel=None,
            ylim = None,
            title=None,
            suptitle=None,
            show=True,
            space_before_avg = False,
            legend_loc = 'upper right',
            figsize=[6.4, 4.8],
            fontsize = 12,
            fname=None):
    """
    Parameters
    ==========
    data : array-like, shape of (n_parents, n_children)
    err : None, array, default=None
        shape of (n_parents, n_children)
    avg : boolen, default=True
    fname : path to file to save or None, default=None
        if not None, save figure as a file.
    """

    n_parents, n_children = data.shape
    if parents is None:
        parents = [str(i) for i in range(1, n_parents+1)]

    if children is None:
        children = [str(i) for i in range(1, n_children+1)]        
    
    if avg_parents:
        mean = np.mean(data, axis=0, keepdims=True)
        if space_before_avg:
            data = np.vstack((data, np.zeros((1, n_children))))
            n_parents += 1
            parents = parents.copy()
            parents.append('')
        data = np.vstack((data, mean))
        n_parents += 1
        parents = parents.copy()
        parents.append('avg')
        
    if avg_children:
        data = np.hstack((data, np.mean(data, axis=1, keepdims=True)))
        n_children += 1
        children = children.copy()
        children.append('avg')

    if color is None:
        color = [None for i in range(n_children)]
    
    if width is None:
        ratio = 0.9
        width = ratio*0.5*2/n_children

    x = np.arange(n_parents)
    #xs = np.arange(n_parents)

    if enable_seaborn:
        sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    #for idx, child in enumerate(children):
    #    ax.bar(x-(n_children*width/2)+(idx*width), data[:, idx], width, label=child, color=color[idx])
    for idx, child in enumerate(children):
            #if err is None or err[idx_x, idx] is None:
            if err is None:
                ax.bar(x = x-(n_children*width/2)+(idx*width), height = data[:, idx], width = width, label=child, color=color[idx])
            else:
                ax.bar(x = x-(n_children*width/2)+(idx*width), height = data[:, idx], width = width, label=child, color=color[idx], yerr = err[:, idx])
    if suptitle is not None:
        fig.suptitle(suptitle)
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize = fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize = fontsize)
    if ylim is not None:
        plt.ylim(ylim)
    #ax.set_xlabel('Subject', fontsize=18)
    ax.set_xticks(x, parents, fontsize = fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    #ax.legend(children, fontsize=12)
    ax.legend(children, fontsize = fontsize, loc = legend_loc)
    #plt.tick_params(labelsize=18)
    fig.tight_layout()

    if fname is not None:
        plt.savefig(fname)

    if show:
        plt.show()
    
    return fig