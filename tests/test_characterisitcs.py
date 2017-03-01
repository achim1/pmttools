import pytest
import numpy as np

from pmttools import characteristics as c
from scipy.constants import elementary_charge as ELEMENTARY_CHARGE

from pyevsel.fitting import Model

def datafactory():
    return np.linspace(0,1000, 10000)

def randomdatafactory():
    return np.random.normal(3, .2, 10000)

def modelfactory():
    return Model(lambda x, p1, p2: x, startparams = (1,2))

def test_chi2_ndf_exponential_part():
    data = datafactory()
    result = c.chi2_ndf_exponential_part(data,data,data, 5,10)
    assert result == 0

def test_calculate_peak_to_valley_ratio():
    data = randomdatafactory()
    xs = datafactory()
    model = modelfactory()
    model.add_data(data, xs=xs)
    result = c.calculate_peak_to_valley_ratio(model, 1, 3, control_plot=True)
    assert result >= 0

def test_get_N_hit():
    data = randomdatafactory()
    hits, nohits = c.get_n_hit(data)
    assert nohits >= 0
    assert hits >= 0

def test_calculate_gain():
    gain = c.calculate_gain(0,1e12)
    gain *= ELEMENTARY_CHARGE
    assert gain == 1

def test_calculate_mu():
    mu = c.calculate_mu(randomdatafactory())
    assert mu >= 0
