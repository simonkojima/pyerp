import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except:
    print("Warning: seaborn is not installed")

def barplot(data, 
            err=None,
            avg=True,
            parents=None,
            children=None,
            color=None,
            width=None,
            ylabel=None,
            xlabel=None,
            title=None,
            suptitle=None,
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
    
    if avg:
        data = np.hstack((data, np.mean(data, axis=1, keepdims=True)))
        data = np.vstack((data, np.mean(data, axis=0, keepdims=True)))
        n_parents += 1
        n_children += 1
        parents = parents.copy()
        children = children.copy()
        parents.append('avg')
        children.append('avg')

    if color is None:
        color = [None for i in range(n_children)]
    
    if width is None:
        ratio = 0.9
        width = ratio*0.5*2/n_children

    x = np.arange(n_parents)

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
    #ax.set_xlabel('Subject', fontsize=18)
    ax.set_xticks(x, parents)
    #ax.legend(children, fontsize=12)
    ax.legend(children)
    #plt.tick_params(labelsize=18)
    fig.tight_layout()

    if fname is not None:
        plt.savefig(fname)

    plt.show()