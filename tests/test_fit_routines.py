import numpy as np
import os
import atexit
import tempfile

import pmttools.fit_routines as fit

from pyevsel.fitting import Model

TESTFILE = tempfile.NamedTemporaryFile(prefix="test", delete=False, suffix=".npy")


def randomdatafactory():
    return np.random.normal(3, .2, 1000)

def datafactory():
    return np.linspace(-1,12,1000)

def create_temporary_file():
    data = [randomdatafactory() for k in range(100)]
    xs = datafactory()
    x_incr = xs[1] - xs[0]
    head = {"xs": xs, "xincr": x_incr, "xunit": "pC", "yunit" : "counts"}
    np.save(TESTFILE, (head,data), allow_pickle=True)

def cleanup():
    try:
        os.remove(TESTFILE.name)
        os.remove(TESTFILE + ".npy")
    except:
        pass # nvm


#atexit.register(cleanup)
create_temporary_file()

def test_pedestal_fit():
    model = fit.pedestal_fit(TESTFILE.name, 20)
    os.remove(TESTFILE.name.replace(".npy",".pdf"))
    assert isinstance(model, Model)

def test_create_charge_response_from_file():
    model, fig = fit.create_charge_response_from_file(TESTFILE.name)
    assert isinstance(model, Model)
    assert isinstance(fig, p.Figure)

#cleanup()