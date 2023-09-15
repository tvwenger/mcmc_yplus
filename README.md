# mcmc_yplus
Infer helium abundance and other parameters for a RRL spectrum using MCMC.

Given a RRL spectrum, `mcmc_yplus` uses a Monte Carlo Markov Chain analysis
to infer optimal number of Gaussian components and their parameters, including
the helium RRL line width and `yplus`, the helium to hydrogen abundance ratio
by number. Here is a basic outline of the algorithm:

0. First, we calculate the Bayesian Information Criterion (BIC) over the
data for the null hypothesis (no baseline, no Gaussian components).

1. Starting with one component, we sample the posterior distribution using
MCMC with at least four independent Markov chains.

2. Because of the degeneracies related to fitting Gaussians to data, it is
possible that chains get stuck in a local maximum of the posterior
distribution. This is especially likely when the number of components is less
than the "true" number of components, in which case each chain may decide
to fit a different subset of components. We check if the chains appear
converged by evaluating the BIC over the data using the mean point
estimate per chain. Any deviant chains are discarded.

3. There also exists a labeling degeneracy: each chain could decide to fit the components
in a different order. To break the degeneracy, we use a Gaussian Mixture Model (GMM)
to cluster the posterior samples of all chains into the same number of groups as there are
expected components. It also tests fewer and more clusters and evaluates the BIC for each
number of clusters in order to determine how many clusters appears optimal to explain the
posterior samples.

4. Once completed, we check to see if the chains appear converged (by comparing 
the BIC of each chain's mean point estimate to that of the combined posterior samples) and
if the number of components seems converged (by comparing the ideal GMM cluster count to
the model number of components). If both convergence checks are passed, then we
stop.

5. We also check to see if there were any divergences in the posterior sampling.
Divergences indicate that the model number of components exceeds the true
number of components present in the data. If there are divergences, then we 
stop.

6. If the BIC of the mean point estimate has decreased compared to the previous iteration,
then we increment the number of Gaussian components by one and continue.

7. If the BIC of the mean point estimate increases two iterations in a row, then we stop.

## Installation
```bash
conda create --name mcmc_yplus -c conda-forge pymc
conda activate mcmc_yplus
pip install git+https://github.com/tvwenger/mcmc_yplus.git
```

## Usage
In general, try `help(function)` for a thorough explanation of
parameters, return values, and other information related to `function`.

### Single model demonstration

If the number of spectral components is known or assumed a prior, then
a single model may be fit.

```python
import numpy as np
import matplotlib.pyplot as plt

from mcmc_yplus.model import Model

rng = np.random.RandomState(seed=1234)

# create some synthetic data
poly_coeffs = [1.0e-7, -2.0e-6, 0.0, -2.5]
H_centers = [-15.0, 20.0]
H_fwhms = [30.0, 35.0]
H_peaks = [10.0, 20.0]
He_H_fwhm_ratio = 0.9
yplus = 0.08
He_offset = 122.15

channel = np.linspace(-300.0, 200.0, 1000)
rms = 1.0
spectrum = rng.normal(loc=0.0, scale=rms, size=1000)
spectrum += np.polyval(poly_coeffs, channel)
for H_center, H_fwhm, H_peak in zip(H_centers, H_fwhms, H_peaks):
    # H component
    H_sigma = H_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    spectrum += H_peak * np.exp(-0.5 * (channel - H_center)**2.0 / H_sigma**2.0)
    
    # He component
    He_peak = H_peak * yplus / He_H_fwhm_ratio
    He_sigma = H_sigma * He_H_fwhm_ratio
    He_center = H_center - He_offset
    spectrum += He_peak * np.exp(-0.5 * (channel - He_center)**2.0 / He_sigma**2.0)

# The H components are expected to be near channel ~ 0, so we normalize
# the channel range here. We also normalize the spectrum for good measure.
channel_range = np.max(channel) - np.min(channel)
norm_channel = channel / channel_range
norm_He_offset = He_offset / channel_range
spectral_range = np.max(spectrum) - np.min(spectrum)
norm_spectrum = spectrum / spectral_range
norm_rms = rms / spectral_range

plt.plot(norm_channel, norm_spectrum, 'k-')
plt.xlabel("Normalized Channel")
plt.ylabel("Normalized Brightness")
plt.tight_layout()
plt.show()
```

![Synthetic Data](https://raw.githubusercontent.com/tvwenger/mcmc_yplus/main/example/spectrum.png)

```python
# initialize the model
model = Model(
    2, # number of Gaussian components
    n_poly = 4, # number of baseline polynomial coefficients (i.e., order + 1)
    seed = 1234, # random seed
    verbose = True, # print information
)

# set data. data is a dictionary containing:
# data['channel'] :: 1-D array spectral axis definition
# data['obs_spectrum'] :: 1-D array brightness spectrum
# data['rms'] :: estimated spectral rms
# data['he_offset'] :: channel(H) - channel(He)
data = {
    'channel': norm_channel,
    'obs_spectrum': norm_spectrum,
    'rms': norm_rms,
    'he_offset': norm_He_offset,
}
model.set_data(data)

# Set prior distribution shapes. The prior distribution for each parameter
# is either a half-normal (for positive-definite parameters) or normal
# (otherwise) distribution. The values supplied here set the standard deviation of
# those posterior distributions.
model.set_priors(
    prior_coeffs = 0.1, # baseline coefficient(s)
    prior_amplitude = 0.5, # normalized Gaussian amplitude(s)
    prior_center = 0.05, # normalized Gaussian center(s)
    prior_fwhm = 0.1, # normalized Gaussian FWHM(s)
    prior_yplus = 0.1, # He/H abundance by number
    prior_He_H_fwhm_ratio = 1.0, # He/H FWHM ratio
)

# Add the likelihood to the model
model.add_likelihood()

# Generate prior predictive samples to test the prior distribution validity
prior_predictive = model.prior_predictive_check(
    samples=50, plot_fname="prior_predictive.png"
)

# Sample the posterior distribution with 4 chains and 4 CPUs
# using 1000 tuning iterations and then drawing 1000 samples
model.fit(init="adapt_diag", tune=500, draws=500, chains=4, cores=4)

# Plot the posterior sample chains
model.plot_traces("traces.png")

# Generate posterior predictive samples to check posterior inference
# thin = keep only every 50th posterior sample
posterior_predictive = model.posterior_predictive_check(
    thin=50, plot_fname="posterior_predictive.png"
)

# Plot the marginalized posterior samples. One plot is created
# per component (named corner_0.png, corner_1.png, etc. in this example),
# one plot is created for the component-combined posterior
# (named corner.png in this example), and one plot is created for the
# non-Gaussian parameters (baseline coefficients, yplus, and FWHM
# ratio; named corner_other.png in this example)
model.plot_corner("corner.png")

# Get the posterior point estimate mean, standard deviation,
# and 68% highest density interval
summary = model.point_estimate(stats=["mean", "std", "hdi"], hdi_prob=0.68)
print(summary['yplus'])
# {'mean': 0.09030119943455478, 'std': 0.009608247252565578, 'hdi': array([0.08118078, 0.09960042])}
```

![Prior Predictive](https://raw.githubusercontent.com/tvwenger/mcmc_yplus/main/example/prior_predictive.png)

![Posterior Predictive](https://raw.githubusercontent.com/tvwenger/mcmc_yplus/main/example/posterior_predictive.png)

![Corner](https://raw.githubusercontent.com/tvwenger/mcmc_yplus/main/example/corner_other.png)

### Inferring number of components

If the number of Gaussian features is not known, then `mcmc_yplus` can
iterate through different models and use various measures to determine
the optimal number of components. The `OptimizeModel` class creates
one model for each possible number of components. The syntax for
adding data, priors, and likelihoods to the optimizer is similar to
that of a single model.

```python
from mcmc_yplus.optimize import OptimizeModel

# initialize the model optimizer
optimizer = OptimizeModel(
    max_n_gauss = 5, # maximum number of Gaussian components
    n_poly = 4, # number of baseline polynomial coefficients (i.e., order + 1)
    seed = 1234, # random seed
    verbose = True, # print information
)

# add data to each model. data is a dictionary containing:
# data['channel'] :: 1-D array spectral axis definition
# data['obs_spectrum'] :: 1-D array brightness spectrum
optimizer.set_data(data)

# Set prior distribution shapes. The prior distribution for each parameter
# is either a half-normal (for positive-definite parameters) or normal
# (otherwise) distribution. The values supplied here set the standard deviation of
# those parameter posterior distributions.
optimizer.set_priors(
    prior_coeffs = 0.1, # baseline coefficient(s)
    prior_amplitude = 0.5, # normalized Gaussian amplitude(s)
    prior_center = 0.05, # normalized Gaussian center(s)
    prior_fwhm = 0.1, # normalized Gaussian FWHM(s)
    prior_yplus = 0.1, # He/H abundance by number
    prior_He_H_fwhm_ratio = 1.0, # He/H FWHM ratio
)

# Add the likelihood to the models
optimizer.add_likelihood()
```

Now we iterate over the models to sample the posterior distributions
starting with one Gaussian component. After each model, we compute the
BIC, check for divergences, and check that the chains and GMM clusters
appear converged.

```python
optimizer.fit_all(init="adapt_diag", tune=500, draws=500, chains=4, cores=4)
```

The "best" model -- the model with the lowest BIC before divergences
or other convergence checks fail -- is saved in `optimizer.best_model`.

```python
print(optimizer.best_model.n_gauss)
# 2
posterior_predictive = optimizer.best_model.posterior_predictive_check(
    thin=50, plot_fname="posterior_predictive.png"
)
optimizer.best_model.plot_corner("corner.png")
```

## Issues and Contributing

Anyone is welcome to submit issues or contribute to the development
of this software via [Github](https://github.com/tvwenger/mcmc_yplus).

## License and Copyright

Copyright (c) 2023 Trey Wenger

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
