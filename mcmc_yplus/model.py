"""
model.py - Model definition

Copyright(C) 2023 by
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
"""

import os

import numpy as np

import pymc as pm
import pytensor.tensor as pt
import arviz as az

from scipy.stats import norm

from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import corner


def gaussian(x, amp, center, fwhm):
    """
    Evaluate a Gaussian.

    Inputs:
        x :: scalar
            Position at which to evaluate
        amp, center, fwhm :: scalars
            Gaussian parameters
    Returns: y
        y :: scalar
            Evaluated Gaussian at x
    """
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amp * np.exp(-0.5 * (x - center) ** 2.0 / (2.0 * sigma**2.0))


class Model:
    """
    Model definition
    """

    def __init__(
        self, n_gauss: int, n_poly: int = 1, seed: int = 1234, verbose: bool = False
    ):
        """
        Initialize a new model

        Inputs:
            n_gauss :: integer
                Number of Gaussian components
            n_poly :: integer
                Number of baseline polynomial coefficients
                (i.e., polynomial order + 1)
            seed :: integer
                Random seed
            verbose :: boolean
                Print extra info
        """
        self.n_gauss = n_gauss
        self.n_poly = n_poly
        self.seed = seed
        self.verbose = verbose
        self.data = None

        # Model parameters
        self._baseline_params = ["baseline_coeffs"]
        self._H_gauss_params = ["H_amplitude", "H_center", "H_fwhm"]
        self._other_params = ["yplus", "He_H_fwhm_ratio"]
        self._parameters = (
            self._baseline_params + self._H_gauss_params + self._other_params
        )

        # Deterministic quantities
        self._He_gauss_determs = ["He_amplitude", "He_center", "He_fwhm"]

        # Number of model parameters = 3 * n_gauss + n_poly + 2
        self._n_params = 3 * self.n_gauss + self.n_poly + 2
        self._n_data = 0

        # Initialize the model
        self.model = pm.Model(
            coords={"poly_coeff": range(self.n_poly), "gauss_comp": range(self.n_gauss)}
        )
        with self.model:
            _ = pm.MutableData("channel", [])
            _ = pm.MutableData("obs_spectrum", [])
            _ = pm.MutableData("he_offset", 0.0)
            _ = pm.MutableData("rms", 0.0)

        # Storage for results
        self.trace = None
        self.posterior_samples = None

        # reset convergence checks
        self._reset()

    def _reset(self):
        """
        Reset convergence checks.

        Inputs: None
        Returns: Nothing
        """
        self._gmm_n_gauss = None
        self._cluster_converged = None
        self._chains_converged = None
        self._good_chains = None
        self._has_divergences = None

    def _cluster_posterior(self, cluster_more=3, rel_bic_threshold=0.1):
        """
        Break the labeling degeneracy (each chain could have a different
        order of components) by identifying clusters in posterior samples.
        Determine if optimal number of components matches n_gauss as
        a convergence test.

        Inputs:
            cluster_more :: integer
                Try fitting Gaussian Mixture Model with n_components between
                1 and n_gauss + cluster_more
            rel_bic_threshold :: scalar
                Identify optimal number of components, n, when
                (BIC[n+1] - BIC[n]) < rel_bic_threshold * BIC[n]

        Returns: Nothing
        """
        # Get posterior samples for H parameters, flatten
        good_chains = self.good_chains()
        features = np.array(
            [
                self.trace.posterior[param].sel(chain=good_chains).data.flatten()
                for param in self._H_gauss_params
            ]
        ).T

        # Use a Gaussian Mixture Model to cluster posterior samples. Test
        # different number of clusters as a convergence check.
        if self.verbose:
            print("Clustering posterior samples...")
        gmms = {}
        max_clusters = self.n_gauss + cluster_more
        n_clusters = [i for i in range(1, max_clusters + 1)]
        for n_cluster in n_clusters:
            gmms[n_cluster] = GaussianMixture(
                n_components=n_cluster,
                max_iter=100,
                init_params="kmeans",
                n_init=10,
                verbose=False,
                random_state=self.seed,
            )
            gmms[n_cluster].fit(features)

        # identify knee in BIC distribution
        self._gmm_bics = np.array(
            [gmms[n_cluster].bic(features) for n_cluster in n_clusters]
        )
        # first time when relative BIC change is less than threshold
        rel_bic_diff = np.abs(np.diff(self._gmm_bics) / self._gmm_bics[:-1])
        best = np.where(rel_bic_diff < rel_bic_threshold)[0]
        if len(best) == 0:
            self._gmm_n_gauss = -1
        else:
            self._gmm_n_gauss = n_clusters[best[0]]

        # check if posterior clusters matches n_gauss
        self._cluster_converged = self._gmm_n_gauss == self.n_gauss

        if self.verbose:
            if self._cluster_converged:
                print(f"GMM converged at n_gauss = {self._gmm_n_gauss}")
            elif self._gmm_n_gauss == -1:
                print(f"GMM prefers n_gauss > {self.n_gauss + cluster_more}")
            else:
                print(f"GMM prefers n_gauss = {self._gmm_n_gauss}")

        # Save clustered posterior samples for n_clusters = n_gauss
        self.posterior_samples = {
            param: self.trace.posterior[param].sel(chain=good_chains).data.flatten()
            for param in self._other_params
        }
        for param in self._baseline_params:
            self.posterior_samples[param] = {}
            for i in range(self.n_poly):
                self.posterior_samples[param][i] = (
                    self.trace.posterior[param]
                    .sel(poly_coeff=i, chain=good_chains)
                    .data.flatten()
                )
        labels = gmms[self.n_gauss].predict(features)
        for param in self._H_gauss_params + self._He_gauss_determs:
            posterior = (
                self.trace.posterior[param].sel(chain=good_chains).data.flatten()
            )
            self.posterior_samples[param] = {}
            for i in range(self.n_gauss):
                self.posterior_samples[param][i] = posterior[labels == i]

    def _plot_predictive(
        self,
        predictive: az.InferenceData,
        plot_fname: str,
        xlabel: str,
        ylabel: str,
    ):
        """
        Generate plots of predictive checks.

        Inputs:
            predictive :: InferenceData
                Predictive samples
            plot_fname :: string
                Plot filename
            xlabel :: string
                x-axis label
            ylabels :: list of strings
                y-axis labels

        Returns: Nothing
        """
        fig, ax = plt.subplots()
        num_chains = len(predictive.chain)
        color = iter(plt.cm.rainbow(np.linspace(0, 1, num_chains)))
        for chain in predictive.chain:
            c = next(color)
            outcomes = predictive["spectrum"].sel(chain=chain).data
            ax.plot(
                self.data["channel"],
                outcomes.T,
                linestyle="-",
                color=c,
                alpha=0.1,
            )
        ax.plot(
            self.data["channel"],
            self.data["obs_spectrum"],
            "k-",
        )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        fig.savefig(plot_fname, bbox_inches="tight")
        plt.close(fig)

    def set_data(self, data: dict):
        """
        Set or update the data.

        Inputs:
            data :: dictionary
                Dictionary with keys "channel", "obs_spectrum", "rms", and "he_offset"
                where
                data['channel'] contains the (normalized) spectral axis
                data['obs_spectrum'] contains the (normalized) spectral intensity
                data['rms'] contains the estimated (normalized) spectral rms
                data['he_offset'] is the (normalized) spectral offset between the He
                and H RRL such that channel_He = channel_H - data['he_offset']


        Returns: Nothing
        """
        self.data = data
        with self.model:
            pm.set_data(self.data)
        self._n_data = len(self.data["channel"])

    def null_bic(self):
        """
        Evaluate the BIC for the null hypothesis (no baseline, no components)

        Inputs: None
        Returns: Nothing
        """
        return (
            -2.0 * norm.logpdf(self.data["obs_spectrum"], scale=self.data["rms"]).sum()
        )

    def set_priors(
        self,
        prior_coeffs: float = 0.1,
        prior_amplitude: float = 0.5,
        prior_center: float = 0.05,
        prior_fwhm: float = 0.1,
        prior_yplus: float = 0.1,
        prior_He_H_fwhm_ratio: float = 1.0,
    ):
        """
        Add a priors to the model.

        Inputs:
            prior_coeffs :: scalar
                Standard deviation of the normalized spectral baseline normal prior
            prior_amplitude :: scalar
                Standard deviation of the normalized H amplitude half-normal prior
            prior_center :: scalar
                Standard deviation of the normalized H centeroid normal prior
            prior_fwhm :: scalar
                Standard deviation of the normalized H FWHM half-normal prior
            prior_yplus :: scalar
                Standard deviation of the y+ half-normal prior
            prior_He_H_fwhm_ratio :: scalar
                Standard deviation of the He/H FWHM ratio half-normal prior

        Returns: Nothing
        """
        with self.model:
            # polynomial baseline coefficients
            _ = pm.Normal(
                "baseline_coeffs", mu=0.0, sigma=prior_coeffs, dims="poly_coeff"
            )

            # Hydrogen components
            h_amp = pm.HalfNormal(
                "H_amplitude", sigma=prior_amplitude, dims="gauss_comp"
            )
            h_chan = pm.Normal("H_center", mu=0, sigma=prior_center, dims="gauss_comp")
            h_fwhm = pm.HalfNormal("H_fwhm", sigma=prior_fwhm, dims="gauss_comp")

            # Helium parameters
            yplus = pm.HalfNormal("yplus", sigma=prior_yplus)
            he_h_fwhm_ratio = pm.HalfNormal(
                "He_H_fwhm_ratio", sigma=prior_He_H_fwhm_ratio
            )

            # Helium components
            _ = pm.Deterministic(
                "He_amplitude", h_amp * yplus / he_h_fwhm_ratio, dims="gauss_comp"
            )
            _ = pm.Deterministic(
                "He_center", h_chan - self.data["he_offset"], dims="gauss_comp"
            )
            _ = pm.Deterministic("He_fwhm", h_fwhm * he_h_fwhm_ratio, dims="gauss_comp")

    def add_likelihood(self):
        """
        Add the likelihood to the model.

        Inputs: None
        Returns: Nothing
        """
        with self.model:
            # baseline
            baseline = pt.sum(
                [
                    self.model["baseline_coeffs"][i] * self.model["channel"] ** i
                    for i in range(self.n_poly)
                ],
                axis=0,
            )

            # H components
            h_spec = gaussian(
                self.model["channel"][:, None],
                self.model["H_amplitude"],
                self.model["H_center"],
                self.model["H_fwhm"],
            ).sum(axis=-1)

            # He components
            he_spec = gaussian(
                self.model["channel"][:, None],
                self.model["He_amplitude"],
                self.model["He_center"],
                self.model["He_fwhm"],
            ).sum(axis=-1)

            # Predicted spectrum
            spectrum_mu = baseline + h_spec + he_spec

            # likelihood
            _ = pm.Normal(
                "spectrum",
                mu=spectrum_mu,
                sigma=self.model["rms"],
                observed=self.model["obs_spectrum"],
            )

    def cluster_converged(self):
        """
        Check if GMM clusters appear converged.

        Inputs: None

        Returns: converged
            converged :: boolean
                True if clusters are converged
        """
        if self._cluster_converged is None:
            raise ValueError("Cluster converge not checked. try model.fit()")
        return self._cluster_converged

    def chains_converged(self, frac_good_chains=0.6, mad_threshold=5.0):
        """
        Check if chains appear converged.

        Inputs:
            frac_good_chains :: scalar
                Chains are converged if the number of good chains exceeds
                {frac_good_chains} * {num_chains}
                Default = 0.6
            mad_threshold :: scalar
                Chains are converged if all have BICs within
                {mad_threshold} * MAD of the clustered BIC
                Default = 5.0

        Returns: converged
            converged :: boolean
                True if chains appear converged
        """
        if self.trace is None:
            raise ValueError("No trace. try model.fit()")

        # check if already determined
        if self._chains_converged is not None:
            return self._chains_converged

        good_chains = self.good_chains()
        num_good_chains = len(good_chains)
        num_chains = len(self.trace.posterior.chain)
        if num_good_chains <= frac_good_chains * num_chains:
            return False

        # per-chain BIC
        bics = np.array([self.bic(chain=chain) for chain in good_chains])
        mad = np.median(np.abs(bics - np.median(bics)))

        # BIC of clustered chains
        clustered_bic = self.bic()

        self._chains_converged = np.all(
            np.abs(bics - clustered_bic) < mad_threshold * mad
        )
        return self._chains_converged

    def has_divergences(self, threshold=10):
        """
        Check if the number of divergences in the good chains exceeds
        the threshold.

        Inputs:
            threshold :: integer
                Number of divergences to indicate a problem

        Returns: divergences
            divergences :: boolean
                True if the model has any divergences
        """
        # check if already determined
        if self._has_divergences is not None:
            return self._has_divergences

        self._has_divergences = (
            self.trace.sample_stats.diverging.sel(chain=self.good_chains()).data.sum()
            > threshold
        )
        return self._has_divergences

    def good_chains(self, mad_threshold=5.0):
        """
        Identify bad chains as those with deviant BICs.

        Inputs:
            mad_threshold :: scalar
                Chains are good if they have BICs within
                {mad_threshold} * MAD of the median BIC.
                Default = 5.0

        Returns: good_chains
            good_chains :: 1-D array of integers
                Chains that appear converged
        """
        if self.trace is None:
            raise ValueError("No trace. try model.fit()")

        # check if already determined
        if self._good_chains is not None:
            return self._good_chains

        # per-chain BIC
        bics = np.array(
            [self.bic(chain=chain) for chain in self.trace.posterior.chain.data]
        )
        mad = np.median(np.abs(bics - np.median(bics)))
        good = np.abs(bics - np.median(bics)) < mad_threshold * mad

        self._good_chains = self.trace.posterior.chain.data[good]
        return self._good_chains

    def prior_predictive_check(self, samples=50, plot_fname: str = None):
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
                Object containing prior predictive samples
        """
        rng = np.random.default_rng(self.seed)
        with self.model:
            trace = pm.sample_prior_predictive(samples=samples, random_seed=rng)

        if plot_fname is not None:
            xlabel = "Normalized Channel"
            ylabel = "Normalized Antenna Temperature"
            self._plot_predictive(trace.prior_predictive, plot_fname, xlabel, ylabel)

        return trace.prior_predictive

    def posterior_predictive_check(self, thin=1, plot_fname: str = None):
        """
        Generate posterior predictive samples, and optionally plot the outcomes.

        Inputs:
            thin :: integer
                Thin posterior samples by keeping one in {thin}
            plot_fname :: string
                If not None, generate a plot of the outcomes over
                the data, and save to this filename.

        Returns: predictive
            predictive :: InferenceData
                Object containing posterior predictive samples
        """
        rng = np.random.default_rng(self.seed)
        with self.model:
            thinned_trace = self.trace.sel(
                chain=self.good_chains(), draw=slice(None, None, thin)
            )
            trace = pm.sample_posterior_predictive(
                thinned_trace, extend_inferencedata=True, random_seed=rng
            )

        if plot_fname is not None:
            xlabel = "Normalized Channel"
            ylabel = "Normalized Antenna Temperature"
            self._plot_predictive(
                trace.posterior_predictive, plot_fname, xlabel, ylabel
            )

        return trace.posterior_predictive

    def fit(
        self,
        init: str = "adapt_diag",
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.8,
        chains: int = 4,
        cores: int = None,
        cluster_more: int = 3,
        rel_bic_threshold: float = 0.1,
    ):
        """
        Sample posterior distribution.

        Inputs:
            init :: string
                pymc initialization strategy
            draws :: integer
                Number of samples
            tune :: integer
                Number of tuning samples
            target_accept :: scalar
                Target sampling acceptance rate. Default = 0.8
            chains :: integer
                Number of chains. Default = 4
            cores :: integer
                Number of cores to run chains in parallel.
                If None, then cores = min(4, num_cpus)
                where num_cpus is the number of CPUs in the system
            cluster_more :: integer
                Try fitting Gaussian Mixture Model with n_components between
                1 and n_gauss + cluster_more
            rel_bic_threshold :: scalar
                Identify optimal number of components, n, when
                (BIC[n+1] - BIC[n]) < rel_bic_threshold * BIC[n]


        Returns: Nothing
        """
        # check that we have enough chains for convergence checks
        if chains < 4:
            raise ValueError("You should use at least 4 chains!")

        # reset convergence checks
        self._reset()

        rng = np.random.default_rng(self.seed)
        with self.model:
            self.trace = pm.sample(
                init=init,
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                progressbar=self.verbose,
                target_accept=target_accept,
                discard_tuned_samples=False,
                compute_convergence_checks=False,
                random_seed=rng,
            )

        # check how many chains converged
        if self.verbose:
            good_chains = self.good_chains()
            if len(good_chains) < chains:
                print(f"Only {len(good_chains)} chains appear converged.")

        # check if there were any divergences
        if self.verbose:
            num_divergences = self.trace.sample_stats.diverging.data.sum()
            if num_divergences > 0:
                print(f"There were {num_divergences} divergences.")

        # cluster posterior samples
        self._cluster_posterior(
            cluster_more=cluster_more,
            rel_bic_threshold=rel_bic_threshold,
        )

    def point_estimate(
        self,
        stats=["mean", "std", "hdi"],
        hdi_prob=0.68,
        chain=None,
    ):
        """
        Get point estimate and other statistics from trace

        Inputs:
            stats :: list of strings
                Statstics to return. Options include "mean", "median",
                "std" (standard deviation), "mad" (median absolute deviation),
                "hdi" (highest density interval)
            hdi_prob :: scalar
                Highest density interval probability to evaluate
                (e.g., stats=['hdi'], hdi_prob=0.68 will calculate
                the 68% highest density interval)
            chain :: None or integer
                If None (default), evaluate statistics across all chains using
                clustered posterior samples. Otherwise, evaluate statistics for
                this chain only.

        Returns: point_estimate
            point_estimate :: dictionary
                Statistics for each parameter
        """
        if chain is None and self.posterior_samples is None:
            raise ValueError("Model has no posterior samples. try model.fit()")

        point_estimate = {}
        for param in self._other_params:
            if chain is None:
                posterior = self.posterior_samples[param]
            else:
                posterior = self.trace.posterior[param].sel(chain=chain).data.T

            # storage for parameter
            point_estimate[param] = {}
            for stat in stats:
                if stat == "mean":
                    point_estimate[param][stat] = np.mean(posterior, axis=0)
                elif stat == "median":
                    point_estimate[param][stat] = np.median(posterior, axis=0)
                elif stat == "std":
                    point_estimate[param][stat] = np.std(posterior, axis=0)
                elif stat == "mad":
                    median = np.median(posterior, axis=0)
                    point_estimate[param][stat] = np.median(
                        np.abs(posterior - median), axis=0
                    )
                elif stat == "hdi":
                    point_estimate[param][stat] = az.hdi(posterior, hdi_prob=hdi_prob)

        for param in (
            self._baseline_params + self._H_gauss_params + self._He_gauss_determs
        ):
            if chain is None:
                posterior = self.posterior_samples[param]
            else:
                posterior = self.trace.posterior[param].sel(chain=chain).data.T

            # storage for parameter
            point_estimate[param] = {}
            for stat in stats:
                if param in self._baseline_params:
                    samples = [posterior[i] for i in range(self.n_poly)]
                else:
                    samples = [posterior[i] for i in range(self.n_gauss)]

                if stat == "mean":
                    point_estimate[param][stat] = [
                        np.mean(sample, axis=0) for sample in samples
                    ]
                elif stat == "median":
                    point_estimate[param][stat] = [
                        np.median(sample, axis=0) for sample in samples
                    ]
                elif stat == "std":
                    point_estimate[param][stat] = [
                        np.std(sample, axis=0) for sample in samples
                    ]
                elif stat == "mad":
                    medians = [np.median(sample, axis=0) for sample in samples]
                    point_estimate[param][stat] = [
                        np.median(np.abs(sample - median), axis=0)
                        for sample, median in zip(samples, medians)
                    ]
                elif stat == "hdi":
                    point_estimate[param][stat] = [
                        az.hdi(sample, hdi_prob=hdi_prob) for sample in samples
                    ]

        return point_estimate

    def lnlike_mean_point_estimate(self, chain=None):
        """
        Evaluate model log-likelihood at the mean point estimate.

        Inputs:
            chain :: None or integer
                If None (default), determine point estimate across all chains using
                clustered posterior samples. Otherwise, get point estimate for
                this chain only.

        Returns: lnlike
            lnlike :: scalar
                Log likelihood at point
        """
        # mean point estimate
        point = self.point_estimate(stats=["mean"], chain=chain)

        # RV names and transformations
        params = {}
        for rv in self.model.free_RVs:
            name = rv.name
            param = self.model.rvs_to_values[rv]
            transform = self.model.rvs_to_transforms[rv]
            if transform is None:
                params[param] = point[name]["mean"]
            else:
                params[param] = transform.forward(point[name]["mean"]).eval()

        return float(self.model.observedlogp.eval(params))

    def bic(self, chain=None):
        """
        Calculate the Bayesian information criterion at the mean point
        estimate.

        Inputs:
            chain :: integer
                If None (default), evaluate BIC across all chains using
                clustered posterior samples. Otherwise, evaluate BIC for
                this chain only.

        Returns: bic
            bic :: scalar
                Bayesian information criterion
        """
        lnlike = self.lnlike_mean_point_estimate(chain=chain)
        return self._n_params * np.log(self._n_data) - 2.0 * lnlike

    def plot_traces(self, plot_fname: str, warmup=False):
        """
        Plot traces.

        Inputs:
            plot_fname :: string
                Plot filename
            warmup :: boolean
                If True, plot warmup samples instead

        Returns: Nothing
        """
        posterior = self.trace.warmup_posterior if warmup else self.trace.posterior
        axes = az.plot_trace(posterior, var_names=self._parameters)
        fig = axes.ravel()[0].figure
        fig.savefig(plot_fname, bbox_inches="tight")
        plt.close(fig)

    def plot_corner(self, plot_fname: str):
        """
        Generate corner plots with optional truths

        Inputs:
            plot_fname :: string
                Figure filename that includes extension. Several plots are generated
                from this basename:
                {plot_fname}
                    Corner plot of non-clustered Gaussian parameters
                {plot_fname-extension}_{num}.{extension}
                    Corner plot of clustered Gaussian parameters
                {plot_fname-extension}_other.{extension}
                    Corner plot of only non-Gaussian related parameters

        Returns: Nothing
        """
        # Clustered Gaussian parameters
        labels = []
        posteriors = []
        for param in self._H_gauss_params:
            labels += [f"{param}"]
            posteriors += [
                np.concatenate(
                    [self.posterior_samples[param][i] for i in range(self.n_gauss)]
                )
            ]
        fig = corner.corner(np.array(posteriors).T, labels=labels)
        fig.savefig(plot_fname, bbox_inches="tight")
        plt.close(fig)

        # Individual Gaussian components
        for i in range(self.n_gauss):
            labels = []
            posteriors = []
            for param in self._H_gauss_params:
                labels += [f"{param}[{i}]"]
                posteriors += [self.posterior_samples[param][i]]

            fig = corner.corner(np.array(posteriors).T, labels=labels)
            fname = os.path.splitext(plot_fname)
            fname = fname[0] + f"_{i}" + fname[1]
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)

        # Other parameters
        fig = corner.corner(
            self.trace, var_names=self._baseline_params + self._other_params
        )
        fname = os.path.splitext(plot_fname)
        fname = fname[0] + "_other" + fname[1]
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
