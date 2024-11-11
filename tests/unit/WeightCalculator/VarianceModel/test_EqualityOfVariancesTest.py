import pytest
import numpy as np

from tests.unit.Util.statistical_distributions import *
from tests.unit.WeightCalculator.VarianceModel.ResponseData import ResponseData

from core.WeightCalculator.VarianceModel import VarianceModel

#=======================================================================================================================
pvalue_tol = 0.01
rng = np.random.default_rng(3104108404)
all_distributions = [normal, uniform, Laplace, gamma, beta, bimodal]

#=======================================================================================================================
@pytest.mark.parametrize("stat_test, distribution", [('Brown-Forsythe', _) for _ in all_distributions]
    + [('Levene', _) for _ in all_distributions] + [('Bartlett', _) for _ in [normal]]
)
def test_EqualityOfVariancesTest(stat_test, distribution):
    response_data = ResponseData(rng)
    response_data.sample_data(distribution)
    params = {'exponents': response_data.alpha,
        'equality of variance test': {'enable': True, 'test': stat_test, 'critical p-value': 0.0}
    }
    input_params = {'variance estimator': {'parametric model': params}}
    std, pvalue = VarianceModel(input_params, response_data, 1).get_standard_deviation()
    Nm = np.array([_.size for _ in response_data.Y[0]])
    Scalc = np.sqrt(Nm) * std
    print(f"Variance model equality-of-variance test: {stat_test}")
    print(f"Data sampled from {distribution.__name__} distribution")
    print(f"Calculated p-value: {pvalue} [tol={pvalue_tol}]")
    assert pvalue >= pvalue_tol
    print()
