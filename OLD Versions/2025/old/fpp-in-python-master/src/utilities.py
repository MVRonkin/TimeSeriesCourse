from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def corrfunc_annotate(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)


def show_pairplot(df, height=6.27, aspect=11.7/8.27, diag_kind='kde'):
    g = sns.pairplot(df, diag_kind=diag_kind)
    g.map_lower(corrfunc_annotate)
    plt.show()
