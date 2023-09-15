"""
optimize.py - Fit spectra with MCMC and determine optimal number of
spectral components.

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

from .model import Model


class OptimizeModel:
    """
    OptimizeModel class definition
    """

    def __init__(self, max_n_gauss=5, n_poly=3, seed=1234, verbose=False):
        """
        Initialize a new OptimizeModel instance.

        Inputs:
            max_n_gauss :: integer
                Maximum number of Gaussian components to consider.
            n_poly :: integer
                Number of baseline polynomial coefficients
            seed :: integer
                Random seed
            verbose :: boolean
                If True, print info

        Returns: optimize_model
            optimize_model :: OptimizeModel
                New OptimizeModel instance
        """
        self.max_n_gauss = max_n_gauss
        self.verbose = verbose
        self.n_gauss = [i for i in range(1, max_n_gauss + 1)]
        self.seed = seed
        self.ready = False

        # Initialize models
        self.models = {}
        for n_gauss in self.n_gauss:
            self.models[n_gauss] = Model(
                n_gauss, n_poly, seed=seed, verbose=self.verbose
            )
        self.best_model = None

    def set_data(self, data):
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
        for n_gauss in self.n_gauss:
            self.models[n_gauss].set_data(data)

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
        for n_gauss in self.n_gauss:
            self.models[n_gauss].set_priors(
                prior_coeffs=prior_coeffs,
                prior_amplitude=prior_amplitude,
                prior_center=prior_center,
                prior_fwhm=prior_fwhm,
                prior_yplus=prior_yplus,
                prior_He_H_fwhm_ratio=prior_He_H_fwhm_ratio,
            )

    def add_likelihood(self):
        """
        Add the likelihood to the model.

        Inputs: None
        Returns: Nothing
        """
        for n_gauss in self.n_gauss:
            self.models[n_gauss].add_likelihood()
        self.ready = True

    def fit_all(
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
        Fit all of the models until clusters and chains converege, or until
        divergences occur, or until the BIC increases twice ina  row.

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
        if not self.ready:
            raise RuntimeError(
                "You must first add the priors and likelihood to the model"
            )

        # reset best model
        self.best_model = None

        minimum_bic = self.models[1].null_bic()
        last_bic = minimum_bic
        if self.verbose:
            print(f"Null hypothesis BIC = {minimum_bic}")
            print()

        num_increase = 0
        for n_gauss in self.n_gauss:
            if self.verbose:
                print(f"Fitting n_gauss = {n_gauss}")
            self.models[n_gauss].fit(
                init=init,
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                cluster_more=cluster_more,
                rel_bic_threshold=rel_bic_threshold,
            )
            current_bic = self.models[n_gauss].bic()
            if self.verbose:
                print(f"Current BIC = {current_bic}")

            # update minimum BIC
            if current_bic < minimum_bic:
                minimum_bic = current_bic
                self.best_model = self.models[n_gauss]
                num_increase = 0

            # Check if BIC is increasing
            if current_bic > last_bic:
                num_increase += 1
            else:
                num_increase = 0

            # Check stopping conditions
            if self.models[n_gauss].has_divergences():
                if self.verbose:
                    print("Model divergences. Stopping.")
                break

            if (
                self.models[n_gauss].cluster_converged()
                and self.models[n_gauss].chains_converged()
            ):
                if num_increase < 2:
                    if self.verbose:
                        print("Model converged, but BIC might decrease. Continuing.")
                else:
                    print("Model converged. Stopping.")

            if num_increase > 1:
                if self.verbose:
                    print("BIC increasing. Stopping.")
                break

            # update BIC
            last_bic = current_bic
            if self.verbose:
                print()
        else:
            print("Reached maximum n_gauss. Stopping.")
