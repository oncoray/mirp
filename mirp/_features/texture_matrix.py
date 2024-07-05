import numpy as np


class Matrix(object):
    def __init__(self):
        self.matrix: np.ndarray | None = None

    def is_empty(self):
        return self.matrix is None or len(self.matrix) == 0


class DirectionalMatrix(Matrix):

    def __init__(self):
        super().__init__()

