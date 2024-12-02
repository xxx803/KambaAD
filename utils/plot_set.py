

def set_canvas(ax):
    for edge in ['top', 'right', 'bottom', 'left']:
        ax.spines[edge].set_edgecolor('#E2E2E2')
        ax.spines[edge].set_linewidth(2)
    ax.tick_params(axis='both', which='both', direction='in', length=4, width=3, color='#E8E8E8')
    return ax
