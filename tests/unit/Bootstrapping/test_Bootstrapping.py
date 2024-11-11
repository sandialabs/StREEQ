import pytest
import yaml
import itertools as it
import numpy as np
from pathlib import Path
from scipy import stats

from tests.unit.Bootstrapping.datasets import *

import core.Bootstrapping as Bootstrapping

#=======================================================================================================================
std_error_tols_byB = {100: 0.5, 300: 0.2, 1000: 0.1}
rng = np.random.default_rng(496758112)
test_params = ([('Nonparametric', heterogeneous_dataset, B, rng) for B in std_error_tols_byB.keys()]
    + [('Parametric', heterogeneous_dataset, B, rng) for B in std_error_tols_byB.keys()]
    + [('Smoothed', heterogeneous_dataset, B, rng) for B in std_error_tols_byB.keys()]
)

#=======================================================================================================================
def get_reference_bootstraps(response_data, B, method, rng, rel_noise=0.1):
    M = len(response_data.Y[0])
    bootstraps = np.empty((M, B))
    if method == 'Nonparametric':
        for m, y in enumerate(response_data.Y[0]):
            for b in range(B):
                indices = rng.integers(y.size, size=(y.size,))
                bootstraps[m, b] = np.mean(y[indices])
    elif method == 'Parametric':
        for m, y in enumerate(response_data.Y[0]):
            for b in range(B):
                sample = np.mean(y)
                Nm = y.size
                s = np.std(y, ddof=1)
                sample += rng.standard_t(Nm-1, size=(1,))*(s/np.sqrt(Nm))
                bootstraps[m, b] = np.mean(sample)
    elif method == 'Smoothed':
        for m, y in enumerate(response_data.Y[0]):
            for b in range(B):
                indices = rng.integers(y.size, size=(y.size,))
                noise = rng.normal(loc=0, scale=rel_noise*np.std(y, ddof=1), size=y.size)
                bootstraps[m, b] = np.mean(y[indices] + noise)
    return bootstraps

#=======================================================================================================================
def get_code_bootstraps(input_params, response_data, subset, fit_model, initial_fits, B, method, rng):
    bootstrapper = getattr(Bootstrapping, method)(input_params, response_data, 1, subset, rng)
    code_bootstraps = np.empty((response_data.X[0].shape[0], B))
    for b in range(B):
        _, _, _, Y, _, _, _, _ = bootstrapper.get_bootstrap(fit_model, b+1, initial_fits)
        code_bootstraps[:, b] = Y
    return code_bootstraps

#=======================================================================================================================
@pytest.mark.parametrize("method, dataset, B, rng", test_params)
def test_Bootstrapping(method, dataset, B, rng):
    input_params, response_data, subset, fit_model, initial_fits = dataset(method, rng)
    code_bootstraps = get_code_bootstraps(input_params, response_data, subset, fit_model, initial_fits, B, method, rng)
    rel_noise = input_params['bootstrapping']['smoothed']['relative noise']
    reference_bootstraps = get_reference_bootstraps(response_data, B, method, rng, rel_noise=rel_noise)
    print(f"Bootstrapping Nonparametric test [method = '{method}', data set='{dataset.__name__}', B={B}]:")
    for m in range(len(response_data.Y[0])):
        print(f"  discretization index m={m+1}:")
        raw_y = response_data.Y[0][m]
        raw_std_error = np.std(raw_y, ddof=1) / np.sqrt(raw_y.size)
        code_std_error = np.std(code_bootstraps[m,:], ddof=1)
        ref_std_error = np.std(reference_bootstraps[m,:], ddof=1)
        rel_diff = np.abs(code_std_error - ref_std_error) / raw_std_error
        print(f"    standard error (raw, code, ref): ({raw_std_error}, {code_std_error}, {ref_std_error})")
        print(f"    standard error test relative difference: {rel_diff} [tol={std_error_tols_byB[B]}]")
        assert rel_diff <= std_error_tols_byB[B]
    print()

#=======================================================================================================================
@pytest.mark.skip(reason="unfinished")
def test_Bootstrapping_Residuals():
    pass

#=======================================================================================================================
@pytest.mark.skip(reason="unfinished")
def test_get_reduced_samples():
    pass
