import pytest
import numpy as np

from tests.unit.Util.statistical_distributions import *
from tests.unit.WeightCalculator.VarianceModel.ResponseData import ResponseData

from core.WeightCalculator.VarianceModel import VarianceModel

#=======================================================================================================================
tolerance = 0.003
rng = np.random.default_rng(2387136031)

#=======================================================================================================================
@pytest.mark.parametrize("distribution, tolerance", [(normal, tolerance), (uniform, tolerance), (Laplace, tolerance),
    (gamma, tolerance), (beta, tolerance), (bimodal, tolerance)]
)
def test_VarianceModel(distribution, tolerance):
    response_data = ResponseData(rng)
    response_data.sample_data(distribution)
    params = {'exponents': response_data.alpha,
        'equality of variance test': {'enable': True, 'test': 'Brown-Forsythe', 'critical p-value': 0.0}
    }
    input_params = {'variance estimator': {'parametric model': params}}
    std, pvalue = VarianceModel(input_params, response_data, 1).get_standard_deviation()
    Nm = np.array([_.size for _ in response_data.Y[0]])
    Scalc = std
    print(f"Variance model test with data sampled from {distribution.__name__} distribution")
    print(f"Exact standard deviation: {response_data.S}")
    print(f"Calculated standard deviation: {Scalc}")
    relative_error = np.max(np.abs(Scalc / response_data.S - 1))
    print(f"Maximum relative error in standard deviation: {relative_error} [tol={tolerance}]")
    assert relative_error <= tolerance
    print()
