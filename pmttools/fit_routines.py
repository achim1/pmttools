"""
Some basic routines to fit models

"""

from pyosci import tools
from pyosci import plotting as plt

from pyevsel.fitting import gauss, Model

from functools import reduce

import pylab as p
import numpy as np
import dashi as d
import seaborn.apionly as sb
d.visual()

from collections import namedtuple

from . import characteristics as c
from . import charge_response_model as crm

PALETTE = sb.color_palette()

def pedestal_fit(filename, nbins, fig=None):
    """
    Fit a pedestal to measured waveform data
    One shot function for
    * integrating the charges
    * making a histogram
    * fitting a simple gaussian to the pedestal
    * calculating mu
        P(hit) = (N_hit/N_all) = exp(QExCExLY)
        where P is the probability for a hit, QE is quantum efficiency,
        CE is the collection efficiency and
        LY the (unknown) light yield

    Args:
        filename (str): Name of the file with waveform data
        nbins (int): number of bins for the underlaying charge histogram

    """

    head, wf = tools.load_waveform(filename)
    charges = -1e12 * tools.integrate_wf(head, wf)
    plt.plot_waveform(head, tools.average_wf(wf))
    p.savefig(filename.replace(".npy", ".wf.pdf"))
    one_gauss = lambda x, n, y, z: n * gauss(x, y, z, 1)
    ped_mod = Model(one_gauss, (1000, -.1, 1))
    ped_mod.add_data(charges, nbins, create_distribution=True, normalize=False)
    ped_mod.fit_to_data(silent=True)
    fig = ped_mod.plot_result(add_parameter_text=((r"$\mu_{{ped}}$& {:4.2e}\\", 1), \
                                                  (r"$\sigma_{{ped}}$& {:4.2e}\\", 2)), \
                              xlabel=r"$Q$ [pC]", ymin=1, xmax=8, model_alpha=.2, fig=fig, ylabel="events")

    ax = fig.gca()
    n_hit = abs(ped_mod._distribution.bincontent - ped_mod.prediction(ped_mod.xs)).sum()
    ax.grid(1)
    bins = np.linspace(min(charges), max(charges), nbins)
    data = d.factory.hist1d(charges, bins)
    n_pedestal = ped_mod._distribution.stats.nentries - n_hit

    mu = -1 * np.log(n_pedestal / ped_mod._distribution.stats.nentries)

    print("==============")
    print("All waveforms: {:4.2f}".format(ped_mod._distribution.stats.nentries))
    print("HIt waveforms: {:4.2f}".format(n_hit))
    print("NoHit waveforms: {:4.2f}".format(n_pedestal))
    print("mu = -ln(N_PED/N_TRIG) = {:4.2e}".format(mu))

    ax.fill_between(ped_mod.xs, 1e-4, ped_mod.prediction(ped_mod.xs),\
                    facecolor=PALETTE[2], alpha=.2)
    p.savefig(filename.replace(".npy", ".pdf"))

    return ped_mod


def create_charge_response_from_file(name):
    """
    One shot function to create a default charge spectrum
    from a file with waveform data

    Args:
        name (str): path to a file with numpy readable waveform data
    """

    head, wf = tools.load_waveform(name)
    plt.plot_waveform(head, tools.average_wf(wf))
    all_charges = 1e12 * np.array([-1 * tools.integrate_wf(head, w) for w in wf])
    # all_charges = all_charges[all_charges > -0.53]
    mu = c.calculate_mu(all_charges, 200)
    nhit, nall = c.get_n_hit(all_charges, 200)
    charge_response_model = crm.construct_charge_response_model(all_charges,\
                                                            model_2PE_response=True,\
                                                            convolved_exponential=True,\
                                                            lowest_mpe_contrib=2)
    fitparams = namedtuple("fitparams", ["N_i", "mu_p", "sigma_p", "p", "A", "mu", "sigma"])

    startparams = fitparams(1, -.2, .1, .8, .3, 3, .2)
    bounds = ((0, -3, 0, .0, 0, 0, .05),
              (1, 0, .5, 1, 100, 5, 5))

    model = fit_model(all_charges, charge_response_model, startparams, rej_outliers=False,\
                                   bounds=bounds)
    fig = model.plot_result(ymin=1e-4, xmax=5, xlabel=r"$Q$ [pC]",\
                            model_alpha=.8,\
                            add_parameter_text=((r"$\sigma_{{ped}}$& {:4.2e}\\", 2),
                                                (r"$\mu_{{SPE}}$& {:4.2e}\\", 5),\
                                                (r"$\sigma_{{SPE}}$& {:4.2e}\\", 6),\
                                                (r"$p_{{exp}}$& {:4.2e}\\", 3)))


    return model, fig

