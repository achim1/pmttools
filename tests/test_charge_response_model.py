import pytest
import numpy as np
import pmttools.charge_response_model as cr

from pyevsel.fitting import Model

# typical test parameters
xs = np.linspace(-1, 12, 10000)
mu_p = -.1
sigma_p = .02
p = .4
A = 3.
mu = 3
sigma = .2
testparams = [mu_p, sigma_p, p, A, mu, sigma]

def datafactory():
    return np.random.normal(mu, sigma, 10000)

def test_pedestal():
    assert len(cr.pedestal(xs, *testparams)) == len(xs)

def test_single_PE_response():
    assert len(cr.single_PE_response(xs, *testparams)) == len(xs)

def test_two_PE_resonse():
    assert len(cr.two_PE_response(xs, *testparams)) == len(xs)

def test_convolve_exponential_part():
    expo = cr.convolve_exponential_part(-1,12)
    assert len(expo(xs, *testparams)) == len(xs)

def test_simple_exponential_response():
    assert len(cr.simple_exponential_response(xs, *testparams)) == len(xs)

def test_multi_PE_response():
    responses = []
    for n in range(2,5):
        responses.append(cr.multi_PE_response(n))

    responses = [r(xs, *testparams) for r in responses]
    assert (responses[0] != responses[1]).all()

def test_construct_charge_response_model():
    data = datafactory()
    model = cr.construct_charge_response_model(data)
    assert isinstance(model, Model)

