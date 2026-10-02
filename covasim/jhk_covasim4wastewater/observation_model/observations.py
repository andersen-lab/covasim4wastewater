from covasim.plotting import (
    handle_args,
    create_figs,
    title_grid_legend,
    reset_ticks,
    tidy_up,
)
from covasim.settings import options as cvo

import numpy as np


def observe(sim, p_test, p_seq, delay_pmf):
    # sim: covasim simulation object
    # p_test: dignosis probability (out of infected)
    # p_seq: sequencing probability (out of dignosed)
    # delay_pmf: distribution of getting dignosed at day k of infected (eg. [0.5, 0.3, 0.2])
    T = sim.npts
    new_inf = sim.results['new_infections']
    new_diag = np.zeros(T)
    new_seq = np.zeros(T)

    for day in range(T):
        n = new_inf[day]
        if n == 0:
            continue
        n_diag = np.random.binomial(n, p_test)
        diag_by_delay = np.random.multinomial(n_diag, delay_pmf)

        for k, count in enumerate(diag_by_delay):
            diag_day = day + k
            if diag_day >= T:
                break
            new_diag[diag_day] += count
            new_seq[diag_day] += np.random.binomial(count, p_seq)

    total_diag = int(np.cumsum(new_diag)[-1])
    total_seq  = int(np.cumsum(new_seq)[-1])

    print(f"{total_diag} cumulative diagnoses (obs model)")
    print(f"{total_seq} cumulative sequenced (obs model)")

    return new_diag, new_seq

def plot_sim_vs_observed(sim, new_diag, new_seq,
                         fig_args=None, plot_args=None, axis_args=None,
                         date_args=None, style_args=None,
                         grid=True, commaticks=True, setylim=True,
                         do_show=None, do_save=False, fig_path=None,
                         fig=None, ax=None, **kwargs):

    # Covasim helper args
    args = handle_args(fig_args=fig_args, plot_args=plot_args,
                       axis_args=axis_args, date_args=date_args,
                       style_args=style_args, do_show=do_show, **kwargs)

    sep_figs = False

    # Extract Covasim results
    dates = sim.results['date']
    new_inf = sim.results['new_infections'].values
    new_diag_cova = sim.results['new_diagnoses'].values

    # color palette
    colors = {
        "inf": "#4C72B0",      # soft blue
        "diag_cova": "#DD8452", # soft orange
        "diag_obs": "#55A868",  # soft green
        "seq_obs": "#C44E52",   # soft red
    }

    with cvo.with_style(args.style):
        fig, figs = create_figs(args, sep_figs, fig, ax)

        if ax is None:
            try:
                ax = fig.axes[0]
            except:
                ax = fig.add_subplot(111, label='ax1')

        # True infections (Covasim)
        ax.plot(dates, new_inf,
                label='True infections (Covasim)',
                color=colors["inf"], lw=1.5, alpha=0.9)

        # Covasim diagnoses
        ax.plot(dates, new_diag_cova,
                label='Diagnoses (Covasim)',
                color=colors["diag_cova"], lw=1.5, alpha=0.9)

        # Observed diagnoses (obs model)
        ax.plot(dates, new_diag,
                label='Observed diagnoses (obs model)',
                color=colors["diag_obs"], lw=1.8, alpha=0.9)

        # Observed sequenced (obs model)
        ax.plot(dates, new_seq,
                label='Observed sequenced (obs model)',
                color=colors["seq_obs"], lw=1.8, alpha=0.9)

        # Apply Covasim style helpers
        title_grid_legend(
            ax,
            'Simulated vs Observed',
            grid, commaticks, setylim,
            args.legend, args.show
        )

        # Make grid lighter
        ax.grid(alpha=0.25)

        reset_ticks(ax, sim, args.date)

    tidy_up(fig, figs, sep_figs, do_save, fig_path, args)

    return None
