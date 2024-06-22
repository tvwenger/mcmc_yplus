"""
plots.py - Plotting helper utilities.

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Changelog:
Trey Wenger - March 2024
"""

import warnings

import arviz as az
import arviz.labels as azl

import matplotlib.pyplot as plt
import numpy as np

from mcmc_yplus.utils import gaussian


def plot_predictive(
    data: dict,
    predictive: az.InferenceData,
    plot_fname: str,
    posterior: az.InferenceData = None,
):
    """
    Generate plots of predictive checks.

    Inputs:
        data :: dictionary
            Model data dictionary
        predictive :: az.InferenceData
            Predictive samples
        plot_fname :: string
            Plot filename
        posterior :: az.InferenceData
            If not None, plot individual posterior predictions
            for individual clouds

    Returns: Nothing
    """
    fig, ax = plt.subplots()
    num_chains = len(predictive.chain)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, num_chains)))

    for chain in predictive.chain:
        c = next(color)
        if posterior is not None:
            for draw in posterior.draw:
                # baseline
                coeffs = posterior["coeffs"].sel(chain=chain, draw=draw).data
                baseline_norm = np.sum(
                    [
                        coeff * data["velocity_norm"] ** i
                        for i, coeff in enumerate(coeffs)
                    ],
                    axis=0,
                )

                # spectrum
                H_spec = gaussian(
                    data["velocity"][:, None],
                    posterior["H_amplitude"].sel(chain=chain, draw=draw).data,
                    posterior["H_center"].sel(chain=chain, draw=draw).data,
                    posterior["H_fwhm"].sel(chain=chain, draw=draw).data,
                ).sum(axis=-1)
                He_spec = gaussian(
                    data["velocity"][:, None],
                    posterior["He_amplitude"].sel(chain=chain, draw=draw).data,
                    posterior["He_center"].sel(chain=chain, draw=draw).data,
                    posterior["He_fwhm"].sel(chain=chain, draw=draw).data,
                ).sum(axis=-1)

                # un-normalize
                baseline = baseline_norm * data["noise"]
                spectrum_mu = H_spec.eval() + He_spec.eval() + baseline

                ax.plot(
                    data["velocity"],
                    spectrum_mu,
                    linestyle="-",
                    color=c,
                    alpha=0.1,
                    linewidth=1,
                )

        # plot predictives
        outcomes = predictive["pred_spectrum"].sel(chain=chain).data
        ax.plot(
            data["velocity"],
            outcomes.T,
            linestyle="-",
            color=c,
            alpha=0.5,
            linewidth=2,
        )

    # plot data
    ax.plot(
        data["velocity"],
        data["spectrum"],
        "k-",
    )
    ax.set_ylabel("Spectrum")
    ax.set_xlabel("Velocity")
    fig.savefig(plot_fname, bbox_inches="tight")
    plt.close(fig)


def plot_pair(trace, var_names, label, fname, labeller: azl.MapLabeller = None):
    """
    Pair plot helper.

    Inputs:
        trace :: az.InferenceData
            Samples to plot
        var_names :: list of strings
            variables from trace to plot
        label :: string
            Label for plot
        fname :: string
            Save plot to this filename
        cloud :: integer or None
            If None, combine all clouds into one. Otherwise, plot only
            this cloud.
        labeller :: azl.MapLabeller or None
            If not None, use apply these labels

    Returns: Nothing
    """
    size = int(2.0 * (len(var_names) + 1))
    textsize = int(np.sqrt(size)) + 8
    fontsize = 2 * size
    with az.rc_context(rc={"plot.max_subplots": None}):
        with warnings.catch_warnings(action="ignore"):
            axes = az.plot_pair(
                trace,
                var_names=var_names,
                combine_dims={"cloud"},
                kind="kde",
                figsize=(size, size),
                labeller=labeller,
                marginals=True,
                marginal_kwargs={"color": "k"},
                textsize=textsize,
                kde_kwargs={
                    "hdi_probs": [
                        0.3,
                        0.6,
                        0.9,
                    ],  # Plot 30%, 60% and 90% HDI contours
                    "contourf_kwargs": {"cmap": "Grays"},
                    "contour_kwargs": {"colors": "k"},
                },
                backend_kwargs={"layout": "constrained"},
            )
    # drop y-label of top left marginal
    axes[0][0].set_ylabel("")
    for ax in axes.flatten():
        ax.grid(False)
    fig = axes.ravel()[0].figure
    fig.text(0.7, 0.8, label, ha="center", va="center", fontsize=fontsize)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
