from tempfile import NamedTemporaryFile

from numpy import ndarray
from scipy.io import savemat, loadmat


def convert_ndarray_to_matlab(input_array: ndarray):
    with NamedTemporaryFile() as f:
        savemat(f.name, {"input": input_array})
        return loadmat(f.name)["input"]
