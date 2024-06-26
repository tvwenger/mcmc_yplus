"""
cluster_posterior.py - Utilities for clustering posterior samples with
Gaussian Mixture Models.

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

from typing import Iterable

import arviz as az
import numpy as np
from scipy.stats import chi2, mode
from scipy.spatial.distance import mahalanobis
from sklearn.mixture import GaussianMixture


def cluster_posterior(
    trace: az.InferenceData,
    n_clusters: int,
    cluster_features: Iterable[str],
    p_threshold: float = 0.9,
    seed: int = 1234,
):
    """
    Each chain (1) could have a different order of clouds (labeling
    degeneracy) or (2) could have converged to a different solution.
    This function uses a Gaussian Mixture Model (GMM) to:
        (1) Determine if there are multiple solutions
        (2) Break the labeling degeneracy.
    Adds new groups to model.trace called "solution_{idx}" where
    idx = 0, 1, ... is the solution index.

    Inputs:
        trace :: az.InferenceData
            Posterior samples
        n_clusters :: integer
            Number of clusters
        cluster_features :: list of strings
            Features to use for clustering
        p_threshold :: scalar
            p-value threshold for considering a unique solution
        seed :: integer
            Random seed

    Returns: Nothing
    """
    # Determine if a chain prefers a unique solution suggested by
    # a significant difference between a GMM fit to only this chain compared
    # to the GMM of previous solutions
    solutions = []
    for chain in trace.chain.data:
        features = np.array(
            [trace[param].sel(chain=chain).data.flatten() for param in cluster_features]
        ).T
        gmm = GaussianMixture(
            n_components=n_clusters,
            max_iter=100,
            init_params="random_from_data",
            n_init=10,
            verbose=False,
            random_state=seed,
        )
        gmm.fit(features)

        # Calculate multivariate z-score between this GMM and other
        # solution means to determine if this is a unique solution.
        # Each GMM could have a different label order, so we compare all
        # combinations of solution GMM cluster and this GMM cluster
        for solution in solutions:
            # cluster_zscore shape (solution clusters, GMM clusters)
            cluster_zscore = np.ones((n_clusters, n_clusters)) * np.nan
            for sol_cluster in range(n_clusters):
                for gmm_cluster in range(n_clusters):
                    # The z-score for MVnormal is mahalanobis distance
                    cov = (
                        solution["gmm"].covariances_[sol_cluster]
                        + gmm.covariances_[gmm_cluster]
                    )
                    inv_cov = np.linalg.inv(cov)
                    zscore = mahalanobis(
                        solution["gmm"].means_[sol_cluster],
                        gmm.means_[gmm_cluster],
                        inv_cov,
                    )
                    cluster_zscore[sol_cluster, gmm_cluster] = zscore

            # calculate significance from z-score
            matched = cluster_zscore**2.0 < chi2.ppf(
                p_threshold, df=len(cluster_features)
            )

            # if all GMM clusters are matched to a solution
            # cluster, then this is NOT a unique solution
            if np.all(np.any(matched, axis=0)):
                # adopt GMM labeling from matched solution
                solution_labels = mode(
                    solution["gmm"].predict(features).reshape(-1, n_clusters),
                    axis=0,
                ).mode
                # ensure all labels present
                if len(np.unique(solution_labels)) == len(solution_labels):
                    solution["chains"][chain] = {"label_order": solution_labels}
                    break
        # This is a unique solution
        else:
            labels = mode(gmm.predict(features).reshape(-1, n_clusters), axis=0).mode
            # ensure all labels present
            if len(np.unique(labels)) == len(labels):
                solution = {
                    "gmm": gmm,
                    "chains": {chain: {"label_order": labels}},
                }
                solutions.append(solution)

    # Each solution now has the labeling degeneracy broken, in that
    # each cloud has been assigned to a unique GMM cluster. We must
    # now determine which order of GMM clusters is preferred
    good_solutions = []
    for solution in solutions:
        chain_order = np.array(
            [chain["label_order"] for chain in solution["chains"].values()]
        )
        # no chains have unique feature labels, abort!
        if len(chain_order) == 0:
            continue
        unique_chain_orders, counts = np.unique(
            chain_order,
            axis=0,
            return_counts=True,
        )
        solution["label_order"] = unique_chain_orders[np.argmax(counts)]

        # Determine the order of clouds needed to match the adopted label order.
        for chain in solution["chains"].values():
            xorder = np.argsort(chain["label_order"])
            chain["cloud_order"] = xorder[
                np.searchsorted(chain["label_order"][xorder], solution["label_order"])
            ]

        # Add solutions to the trace
        coords = dict(trace.coords)
        coords["chain"] = list(solution["chains"].keys())
        dims = {}
        posterior_clustered = {}
        for param, samples in trace.data_vars.items():
            if "cloud" in samples.coords:
                # break labeling degeneracy
                posterior_clustered[param] = np.array(
                    [
                        samples.sel(chain=chain, cloud=order["cloud_order"]).data
                        for chain, order in solution["chains"].items()
                    ]
                )
            else:
                posterior_clustered[param] = np.array(
                    [
                        samples.sel(chain=chain).data
                        for chain in solution["chains"].keys()
                    ]
                )
            dims[param] = list(samples.coords)
        solution["posterior_clustered"] = posterior_clustered
        solution["coords"] = coords
        solution["dims"] = dims
        good_solutions.append(solution)
    return good_solutions
