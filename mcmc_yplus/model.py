"""
model.py - Model definition

Copyright(C) 2023-2024 by
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
Trey Wenger - September 2023
Trey Wenger - May 2024 - Updates to mimic "caribou" framework
"""

import os
import warnings

import numpy as np
from numpy.polynomial import Polynomial

import pymc as pm
from pymc.variational.callbacks import CheckParametersConvergence
import pytensor.tensor as pt

import arviz as az
import arviz.labels as azl

from scipy.stats import norm

import matplotlib.pyplot as plt
import graphviz

from mcmc_yplus.nuts import init_nuts
from mcmc_yplus import plots
from mcmc_yplus.cluster_posterior import cluster_posterior
from mcmc_yplus.utils import gaussian


class Model:
    """
    Model definition
    """

    def __init__(
        self,
        data: dict,
        n_clouds: int,
        baseline_degree: int = 0,
        seed: int = 1234,
        verbose: bool = False,
    ):
        """
        Initialize a new model

        Inputs:
            data :: dictionary
                Dictionary with keys "velocity", "spectrum", "noise"
                where
                data["velocity"] contains the spectral axis (km/s)
                data["spectrum"] contains the spectrum (brightness temp, K)
                data["noise"] contains the spectral rms noise (brightness temp, K)
            n_clouds :: integer
                Number of Gaussian components
            baseline_degree :: integer
                Degree of polynomial baseline
            seed :: integer
                Random seed
            verbose :: boolean
                Print extra info
        """
        self.n_clouds = n_clouds
        self.baseline_degree = baseline_degree
        self.seed = seed
        self.verbose = verbose
        self.data = data

        # Center and normalize velocity
        self.data["velocity_norm"] = (
            self.data["velocity"] - np.mean(self.data["velocity"])
        ) / np.std(self.data["velocity"])

        # Normalize data by the noise
        self.data["spectrum_norm"] = self.data["spectrum"] / self.data["noise"]

        # Initialize the model
        self.model = pm.Model(
            coords={
                "vel": self.data["velocity"],
                "coeff": range(self.baseline_degree + 1),
                "cloud": range(self.n_clouds),
            }
        )
        with self.model:
            for key, value in self.data.items():
                _ = pm.Data(key, value, dims="vel")
        self._n_data = len(self.data["velocity"])

        # Model parameters
        self.baseline_params = ["coeffs"]
        self.hyper_params = []
        self.cloud_params = [
            "H_amplitude",
            "H_center",
            "H_fwhm",
            "He_H_fwhm_ratio",
            "yplus",
        ]
        self.deterministics = ["He_amplitude", "He_center", "He_fwhm"]

        # Parameters used for posterior clustering
        self._cluster_features = ["H_amplitude", "H_center", "H_fwhm"]

        # Arviz labeller map
        self.var_name_map = {
            "coeffs": r"$\beta$",
            "yplus": r"$y^+$",
            "He_H_fwhm_ratio": r"$\Delta V_{\rm He}/\Delta V_{\rm H}$",
            "H_amplitude": r"$T_{L,\rm H}$",
            "H_center": r"$V_{\rm LSR, H}$",
            "H_fwhm": r"$\Delta V_{\rm H}$",
            "He_amplitude": r"$T_{L,\rm He}$",
            "He_center": r"$V_{\rm LSR, He}$",
            "He_fwhm": r"$\Delta V_{\rm He}$",
        }

        # Reset results and convergence checks
        self.reset_results()

    @property
    def _n_params(self):
        """
        Determine the number of model parameters.
        """
        return (
            len(self.cloud_params) * self.n_clouds
            + len(self.baseline_params) * (self.baseline_degree + 1)
            + len(self.hyper_params)
        )

    @property
    def _get_unique_solution(self):
        """
        Return the unique solution index (0) if there is a unique
        solution, otherwise raise an exception.
        """
        if not self.unique_solution:
            raise ValueError("There is not a unique solution. Must supply solution.")
        return 0

    @property
    def labeller(self):
        """
        Get the arviz labeller.
        """
        return azl.MapLabeller(var_name_map=self.var_name_map)

    @property
    def unique_solution(self):
        """
        Check if posterior samples suggest a unique solution
        """
        if self.solutions is None or len(self.solutions) == 0:
            raise ValueError("No solutions. Try solve()")
        return len(self.solutions) == 1

    def _add_likelihood(self):
        """
        Add the likelihood to the model.

        Inputs: None
        Returns: Nothing
        """
        with self.model:
            # baseline
            baseline_norm = pt.sum(
                [
                    self.model["coeffs"][i] * self.model["velocity_norm"] ** i
                    for i in range(self.baseline_degree + 1)
                ],
                axis=0,
            )

            # H components
            H_spec = gaussian(
                self.model["velocity"][:, None],
                self.model["H_amplitude"],
                self.model["H_center"],
                self.model["H_fwhm"],
            ).sum(axis=-1)

            # He components
            He_spec = gaussian(
                self.model["velocity"][:, None],
                self.model["He_amplitude"],
                self.model["He_center"],
                self.model["He_fwhm"],
            ).sum(axis=-1)

            # Normalized likelihood
            pred_spectrum = H_spec + He_spec
            spectrum_mu = pred_spectrum / self.model["noise"] + baseline_norm
            _ = pm.Normal(
                "pred_spectrum_norm",
                mu=spectrum_mu,
                sigma=1.0,
                observed=self.model["spectrum_norm"],
                dims="vel",
            )

    def reset_results(self):
        """
        Reset results and convergence checks.

        Inputs: None
        Returns: Nothing
        """
        self.approx: pm.Approximation = None
        self.trace: az.InferenceData = None
        self.solutions = None
        self._good_chains = None
        self._chains_converged: bool = None

    def null_bic(self):
        """
        Evaluate the BIC for the null hypothesis (baseline only, no clouds)

        Inputs: None
        Returns: Nothing
        """
        # fit polynomial baseline
        baseline = Polynomial.fit(
            self.data["velocity_norm"],
            self.data["spectrum_norm"],
            self.baseline_degree,
        )(self.data["velocity_norm"])

        # evaluate likelihood
        emission = self.data["spectrum_norm"] - baseline
        lnlike = norm.logpdf(emission).sum()

        n_params = self.baseline_degree + 1
        return n_params * np.log(self._n_data) - 2.0 * lnlike

    def lnlike_mean_point_estimate(self, chain: int = None, solution: int = None):
        """
        Evaluate model log-likelihood at the mean point estimate of posterior samples.

        Inputs:
            chain :: None or integer
                If None (default), evaluate BIC across all chains using
                clustered posterior samples. Otherwise, evaluate BIC for
                this chain only using un-clustered posterior samples.
            solution :: None or integer
                Solution index
                If chain is None and solution is None:
                    If there is a unique solution, use that
                    Otherwise, raise an exception
                If chain is None and solution is not None:
                    Use this solution index
                If chain is not None:
                    This parameter has no effect

        Returns: lnlike
            lnlike :: scalar
                Log likelihood at point
        """
        if chain is None and solution is None:
            solution = self._get_unique_solution

        # mean point estimate
        if chain is None:
            point = self.trace[f"solution_{solution}"].mean(dim=["chain", "draw"])
        else:
            point = self.trace.posterior.sel(chain=chain).mean(dim=["draw"])

        # RV names and transformations
        params = {}
        for rv in self.model.free_RVs:
            name = rv.name
            param = self.model.rvs_to_values[rv]
            transform = self.model.rvs_to_transforms[rv]
            if transform is None:
                params[param] = point[name].data
            else:
                params[param] = transform.forward(
                    point[name].data, *rv.owner.inputs
                ).eval()

        return float(self.model.logp().eval(params))

    def bic(self, chain: int = None, solution: int = None):
        """
        Calculate the Bayesian information criterion at the mean point estimate.

        Inputs:
            chain :: None or integer
                If None (default), evaluate BIC across all chains using
                clustered posterior samples. Otherwise, evaluate BIC for
                this chain only using un-clustered posterior samples.
            solution :: None or integer
                Solution index
                If chain is None and solution is None:
                    If there is a unique solution, use that
                    Otherwise, raise an exception
                If chain is None and solution is not None:
                    Use this solution index
                If chain is not None:
                    This parameter has no effect

        Returns: bic
            bic :: scalar
                Bayesian information criterion
        """
        try:
            lnlike = self.lnlike_mean_point_estimate(chain=chain, solution=solution)
            return self._n_params * np.log(self._n_data) - 2.0 * lnlike
        except ValueError as e:
            print(e)
            return np.inf

    def good_chains(self, mad_threshold: float = 5.0):
        """
        Identify bad chains as those with deviant BICs.

        Inputs:
            mad_threshold :: scalar
                Chains are good if they have BICs within {mad_threshold} * MAD of the median BIC.

        Returns: good_chains
            good_chains :: 1-D array of integers
                Chains that appear converged
        """
        if self.trace is None:
            raise ValueError("Model has no posterior samples. Try fit() or sample().")

        # check if already determined
        if self._good_chains is not None:
            return self._good_chains

        # if the trace has fewer than 2 chains, we assume they're both ok so we can run
        # convergence diagnostics
        if len(self.trace.posterior.chain) < 3:
            self._good_chains = self.trace.posterior.chain.data
            return self._good_chains

        # per-chain BIC
        bics = np.array(
            [self.bic(chain=chain) for chain in self.trace.posterior.chain.data]
        )
        mad = np.median(np.abs(bics - np.median(bics)))
        good = np.abs(bics - np.median(bics)) < mad_threshold * mad

        self._good_chains = self.trace.posterior.chain.data[good]
        return self._good_chains

    def set_priors(
        self,
        prior_H_amplitude: float = 10.0,
        prior_H_center: list[float] = [0.0, 10.0],
        prior_H_fwhm: float = 20.0,
        prior_yplus: float = 0.1,
        prior_He_H_fwhm_ratio: float = 0.1,
    ):
        """
        Add priors to model.

        Inputs:
            prior_H_amplitude :: scalar (K)
                Width of the half-normal prior distribution on H amplitude
            prior_H_center :: list of two scalar (km/s)
                Center and width of the normal prior distribution on the H center
            prior_H_fwhm :: scalar (km/s)
                Mode of the k=4 Gamma prior distribution on the H FWHM
            prior_yplus :: scalar
                Width of the half-normal prior distribution on y+
            prior_He_H_fwhm_ratio :: scalar
                Width of the unity-centered, truncated normal prior distribution
                on the He/H FWHM ratio

        Returns: Nothing
        """
        with self.model:
            # (normalized) polynomial baseline coefficients
            _ = pm.Normal(
                "coeffs",
                mu=0.0,
                sigma=1.0,
                dims="coeff",
            )

            # H parameters
            H_amplitude = pm.HalfNormal(
                "H_amplitude", sigma=prior_H_amplitude, dims="cloud"
            )
            H_center = pm.Normal(
                "H_center", mu=prior_H_center[0], sigma=prior_H_center[1], dims="cloud"
            )
            H_fwhm = pm.Gamma(
                "H_fwhm", alpha=4.0, beta=3.0 / prior_H_fwhm, dims="cloud"
            )

            # He parameters
            yplus = pm.HalfNormal("yplus", sigma=prior_yplus, dims="cloud")
            He_H_fwhm_ratio = pm.TruncatedNormal(
                "He_H_fwhm_ratio",
                mu=1.0,
                sigma=prior_He_H_fwhm_ratio,
                lower=0.0,
                dims="cloud",
            )

            # Deterministic He parameters
            _ = pm.Deterministic(
                "He_amplitude", H_amplitude * yplus / He_H_fwhm_ratio, dims="cloud"
            )
            _ = pm.Deterministic("He_center", H_center - 122.15, dims="cloud")
            _ = pm.Deterministic("He_fwhm", H_fwhm * He_H_fwhm_ratio, dims="cloud")

        # add likelihood
        self._add_likelihood()

    def prior_predictive_check(self, samples: int = 50, plot_fname: str = None):
        """
        Generate prior predictive samples, and optionally plot the outcomes.

        Inputs:
            samples :: integer
                Number of prior predictive samples to generate
            plot_fname :: string
                If not None, generate a plot of the outcomes over
                the data, and save to this filename.

        Returns: predictive
            predictive :: InferenceData
                Object containing prior and prior predictive samples
        """
        with self.model:
            trace = pm.sample_prior_predictive(samples=samples, random_seed=self.seed)
            # add un-normalized predictive
            trace.prior_predictive["pred_spectrum"] = (
                trace.prior_predictive["pred_spectrum_norm"] * self.data["noise"]
            )

        if plot_fname is not None:
            plots.plot_predictive(self.data, trace.prior_predictive, plot_fname)

        return trace

    def posterior_predictive_check(
        self, solution: int = None, thin: int = 100, plot_fname: str = None
    ):
        """
        Generate posterior predictive samples, and optionally plot the outcomes.

        Inputs:
            solution :: integer
                If None, generate posterior predictive samples from the un-clustered posterior
                samples. Otherwise, generate predictive samples from this solution index.
            thin :: integer
                Thin posterior samples by keeping one in {thin}
            plot_fname :: string
                If not None, generate a plot of the outcomes over
                the data, and save to this filename.

        Returns: predictive
            predictive :: InferenceData
                Object containing posterior and posterior predictive samples
        """
        with self.model:
            if solution is None:
                posterior = self.trace.posterior.sel(
                    chain=self.good_chains(), draw=slice(None, None, thin)
                )
            else:
                posterior = self.trace[f"solution_{solution}"].sel(
                    draw=slice(None, None, thin)
                )
            trace = pm.sample_posterior_predictive(
                posterior,
                extend_inferencedata=True,
                random_seed=self.seed,
            )
            # add un-normalized predictive
            trace.posterior_predictive["pred_spectrum"] = (
                trace.posterior_predictive["pred_spectrum_norm"] * self.data["noise"]
            )

        if plot_fname is not None:
            plots.plot_predictive(
                self.data,
                trace.posterior_predictive,
                plot_fname,
                posterior=posterior,
            )

        return trace

    def fit(
        self,
        n: int = 100_000,
        draws: int = 1_000,
        rel_tolerance: float = 0.03,
        abs_tolerance: float = 0.03,
        learning_rate: float = 1e-2,
        **kwargs,
    ):
        """
        Fit posterior using variational inference (VI). If you get NaNs
        during optimization, try increasing the learning rate.

        Inputs:
            n :: integer
                Number of VI iterations
            draws :: integer
                Number of samples to draw from fitted posterior
            rel_tolerance :: scalar
                Relative parameter tolerance for VI convergence
            abs_tolerance :: scalar
                Absolute parameter tolerance for VI convergence
            learning_rate :: scalar
                adagrad_window learning rate. Try increasing if you get NaNs
            **kwargs :: additional keyword arguments
                Additional arguments passed to pymc.fit
                (method)

        Returns: Nothing
        """
        # reset convergence checks
        self.reset_results()

        with self.model:
            callbacks = [
                CheckParametersConvergence(tolerance=rel_tolerance, diff="relative"),
                CheckParametersConvergence(tolerance=abs_tolerance, diff="absolute"),
            ]
            self.approx = pm.fit(
                n=n,
                random_seed=self.seed,
                progressbar=self.verbose,
                callbacks=callbacks,
                obj_optimizer=pm.adagrad_window(learning_rate=learning_rate),
                **kwargs,
            )
            self.trace = self.approx.sample(draws)

    def sample(
        self,
        init: str = "advi+adapt_diag",
        n_init: int = 100_000,
        chains: int = 4,
        init_kwargs: dict = None,
        nuts_kwargs: dict = None,
        **kwargs,
    ):
        """
        Sample posterior distribution using MCMC.

        Inputs:
            init :: string
                Initialization strategy
            n_init :: integer
                Number of initialization iterations
            chains :: integer
                Number of chains
            init_kwargs :: dictionary
                Keyword arguments passed to init_nuts
                (tolerance, learning_rate)
            nuts_kwargs :: dictionary
                Keyword arguments passed to pm.NUTS
                (target_accept)
            **kwargs :: additional keyword arguments
                Keyword arguments passed to pm.sample
                (cores, tune, draws)

        Returns: Nothing
        """
        # reset convergence checks
        self.reset_results()

        if init == "auto":
            init = "jitter+adapt_diag"

        if init_kwargs is None:
            init_kwargs = {}
        if nuts_kwargs is None:
            nuts_kwargs = {}

        # attempt custom initialization
        initial_points, step = init_nuts(
            self.model,
            init=init,
            n_init=n_init,
            chains=chains,
            nuts_kwargs=nuts_kwargs,
            seed=self.seed,
            verbose=self.verbose,
            **init_kwargs,
        )

        # if we're using custom initialization, then drop nuts
        # arguments from pm.sample
        if initial_points is not None:
            nuts_kwargs = {}

        with self.model:
            self.trace = pm.sample(
                init=init,
                initvals=initial_points,
                step=step,
                chains=chains,
                progressbar=self.verbose,
                discard_tuned_samples=False,
                compute_convergence_checks=False,
                random_seed=self.seed,
                **nuts_kwargs,
                **kwargs,
            )

        # diagnostics
        if self.verbose:
            # converged chains
            good_chains = self.good_chains()
            if len(good_chains) < len(self.trace.posterior.chain):
                print(f"Only {len(good_chains)} chains appear converged.")

            # divergences
            num_divergences = self.trace.sample_stats.diverging.sel(
                chain=self.good_chains()
            ).data.sum()
            if num_divergences > 0:
                print(f"There were {num_divergences} divergences in converged chains.")

    def solve(self, p_threshold=0.9):
        """
        Cluster posterior samples and determine unique solutions. Adds
        new groups to self.trace called "solution_{idx}" for the posterior
        samples of each unique solution.

        Inputs:
            p_threshold :: scalar
                p-value threshold for considering a unique solution

        Returns: Nothing
        """
        # Drop solutions if they already exist in trace
        for group in list(self.trace.groups()):
            if "solution" in group:
                del self.trace[group]

        self.solutions = []
        solutions = cluster_posterior(
            self.trace.posterior.sel(chain=self.good_chains()),
            self.n_clouds,
            self._cluster_features,
            p_threshold=p_threshold,
            seed=self.seed,
        )
        if len(solutions) < 1 and self.verbose:
            print("No solution found!")

        # convergence check
        unique_solution = len(solutions) == 1
        if self.verbose:
            if unique_solution:
                print("GMM converged to unique solution")
            else:
                print(f"GMM found {len(solutions)} unique solutions")
                for solution_idx, solution in enumerate(solutions):
                    print(
                        f"Solution {solution_idx}: chains {list(solution['chains'].keys())}"
                    )

        # labeling degeneracy check
        for solution_idx, solution in enumerate(solutions):
            chain_order = np.array(
                [chain["label_order"] for chain in solution["chains"].values()]
            )
            if self.verbose and not np.all(chain_order == solution["label_order"]):
                print(f"Chain label order mismatch in solution {solution_idx}")
                for chain, order in solution["chains"].items():
                    print(f"Chain {chain} order: {order['label_order']}")
                print(f"Adopting (first) most common order: {solution['label_order']}")

            # Add solution to the trace
            with warnings.catch_warnings(action="ignore"):
                self.trace.add_groups(
                    **{
                        f"solution_{solution_idx}": solution["posterior_clustered"],
                        "coords": solution["coords"],
                        "dims": solution["dims"],
                    }
                )
                self.solutions.append(solution_idx)

    def plot_graph(self, dotfile: str, ext: str):
        """
        Generate dot plot of model graph.

        Inputs:
            dotfile :: string
                Where graphviz source is saved
            ext :: string
                Rendered image is {dotfile}.{ext}

        Returns: Nothing
        """
        gviz = pm.model_to_graphviz(self.model)
        gviz.graph_attr["rankdir"] = "TB"
        gviz.graph_attr["splines"] = "ortho"
        gviz.graph_attr["newrank"] = "false"
        unflat = gviz.unflatten(stagger=3)

        # clean up
        source = []
        for line in unflat.source.splitlines():
            # rename normalized data/likelihood vars
            line = line.replace("velocity_norm", "velocity")
            line = line.replace("spectrum_norm", "spectrum")
            line = line.replace("pred_spectrum_norm", "pred_spectrum")
            source.append(line)

        # save and render
        with open(dotfile, "w", encoding="ascii") as f:
            f.write("\n".join(source))
        graphviz.render("dot", ext, dotfile)

    def plot_traces(self, plot_fname: str, warmup: bool = False):
        """
        Plot traces for all chains.

        Inputs:
            plot_fname :: string
                Plot filename
            warmup :: boolean
                If True, plot warmup samples instead

        Returns: Nothing
        """
        posterior = self.trace.warmup_posterior if warmup else self.trace.posterior
        with az.rc_context(rc={"plot.max_subplots": None}):
            var_names = [rv.name for rv in self.model.free_RVs]
            axes = az.plot_trace(
                posterior.sel(chain=self.good_chains()),
                var_names=var_names,
            )
            fig = axes.ravel()[0].figure
            fig.tight_layout()
            fig.savefig(plot_fname, bbox_inches="tight")
            plt.close(fig)

    def plot_pair(self, plot_fname: str, solution: int = None):
        """
        Generate pair plots from clustered posterior samples.

        Inputs:
            plot_fname :: string
                Figure filename with the format: {basename}.{ext}
                Several plots are generated:
                {basename}.{ext}
                    Pair plot of non-clustered cloud parameters
                {basename}_determ.{ext}
                    Pair plot of non-clustered cloud deterministic parameters
                {basename}_{cloud}.{ext}
                    Pair plot of clustered cloud with index {cloud} parameters
                {basename}_{num}_determ.{ext}
                    Pair plot of clustered cloud with index {cloud} deterministic parameters
                {basename}_other.{ext}
                    Pair plot of baseline and hyper parameters
            solution :: None or integer
                Plot the posterior samples associated with this solution index. If
                solution is None and there is a unique solution, use that.
                Otherwise, raise an exception.

        Returns: Nothing
        """
        if solution is None:
            solution = self._get_unique_solution
        trace = self.trace[f"solution_{solution}"]

        basename, ext = os.path.splitext(plot_fname)

        # All cloud free parameters
        plots.plot_pair(
            trace,
            self.cloud_params,
            "All Clouds\nFree Parameters",
            plot_fname,
        )
        # All cloud deterministic parameters
        plots.plot_pair(
            trace,
            self.deterministics,
            "All Clouds\nDerived Quantities",
            basename + "_determ" + ext,
        )
        # Baseline & hyper parameters
        plots.plot_pair(
            trace,
            self.baseline_params + self.hyper_params,
            "All Clouds\nDerived Quantities",
            basename + "_other" + ext,
        )
        # Cloud quantities
        for cloud in range(self.n_clouds):
            plots.plot_pair(
                trace.sel(cloud=cloud),
                self.cloud_params,
                f"Cloud {cloud}\nFree Parameters",
                basename + f"_{cloud}" + ext,
            )
            plots.plot_pair(
                trace.sel(cloud=cloud),
                self.deterministics,
                f"Cloud {cloud}\nDerived Quantities",
                basename + f"_{cloud}_determ" + ext,
            )
