"""
optimize.py - Fit spectra with MCMC and determine optimal number of
spectral components.

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
Trey Wenger - June 2024 - Updates to mimic "caribou" framework
"""

import numpy as np

from .model import Model


class Optimize:
    """
    Optimize class definition
    """

    def __init__(
        self,
        data: dict,
        max_n_clouds: int,
        baseline_degree: int = 0,
        seed: int = 1234,
        verbose: bool = False,
    ):
        """
        Initialize a new Optimize instance.

        Inputs:
            data :: dictionary
                Dictionary with keys "velocity", "spectrum", "noise"
                where
                data["velocity"] contains the spectral axis (km/s)
                data["spectrum"] contains the spectrum (brightness temp, K)
                data["noise"] contains the spectral rms noise (brightness temp, K)
            max_n_clouds :: integer
                Maximum number of Gaussian components
            baseline_degree :: integer
                Degree of polynomial baseline
            seed :: integer
                Random seed
            verbose :: boolean
                Print extra info

        Returns: optimize_model
            optimize_model :: Optimize
                New Optimize instance
        """
        self.max_n_clouds = max_n_clouds
        self.verbose = verbose
        self.n_clouds = [i for i in range(1, self.max_n_clouds + 1)]
        self.seed = seed
        self.data = data

        # Initialize models
        self.models = {}
        for n_cloud in self.n_clouds:
            self.models[n_cloud] = Model(
                self.data, n_cloud, baseline_degree, seed=seed, verbose=self.verbose
            )
        self.best_model = None

    def set_priors(self, **kwargs):
        """
        Add a priors to the model.

        Inputs:
            See model.set_priors

        Returns: Nothing
        """
        for n_cloud in self.n_clouds:
            self.models[n_cloud].set_priors(**kwargs)

    def fit_all(self, **kwargs):
        """
        Fit posterior of all models using variational inference.

        Inputs:
            see model.fit

        Returns: Nothing
        """
        if self.verbose:
            print(f"Null hypothesis BIC = {self.models[1].null_bic():.3e}")

        for n_cloud in self.n_clouds:
            if self.verbose:
                print(f"Approximating n_cloud = {n_cloud} posterior...")
            self.models[n_cloud].fit(**kwargs)
            self.models[n_cloud].solve()
            if self.verbose:
                print(f"n_cloud = {n_cloud} BIC = {self.models[n_cloud].bic():.3e}")
                print()

    def optimize(self, bic_threshold: float = 10.0, fit_kwargs={}, sample_kwargs={}):
        """
        Determine the optimal number of clouds by minimizing the BIC
        using variational inference, and then sample the best model using
        MCMC and solve the labeling degeneracy.

        Inputs:
            bic_threshold :: scalar
                Sample the first model that is within min(BIC)+bic_threshold
            fit_kwargs :: dictionary
                Arguments passed to fit()
            sample_kwargs :: dictionary
                Arguments passed to sample()

        Returns: Nothing
        """
        # fit all with VI
        self.fit_all(**fit_kwargs)

        # get best model
        model_bics = np.array([self.models[n_cloud].bic() for n_cloud in self.n_clouds])
        best_n_clouds = self.n_clouds[
            np.where(model_bics < (np.nanmin(model_bics) + bic_threshold))[0][0]
        ]
        self.best_model = self.models[best_n_clouds]

        # sample best
        if self.verbose:
            print(f"Sampling best model (n_cloud = {self.best_model.n_clouds})...")
        self.best_model.sample(**sample_kwargs)
        self.best_model.solve()
