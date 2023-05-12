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
        data = np.vstack((data, np.mean(data, axis=0, keepdims=True)))
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

    if enable_seaborn:
        sns.set()
    fig, ax = plt.subplots()
    for idx, child in enumerate(children):
        ax.bar(x-(n_children*width/2)+(idx*width), data[:, idx], width, label=child, color=color[idx])
    if suptitle is not None:
        fig.suptitle(suptitle)
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylim is not None:
        plt.ylim(ylim)
    #ax.set_xlabel('Subject', fontsize=18)
    ax.set_xticks(x, parents)
    #ax.legend(children, fontsize=12)
    ax.legend(children)
    #plt.tick_params(labelsize=18)
    fig.tight_layout()

    if fname is not None:
        plt.savefig(fname)

    if show:
        plt.show()
    
    return fig