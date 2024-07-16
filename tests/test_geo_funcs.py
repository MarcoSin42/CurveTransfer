import pytest
import numpy as np

from CurveTransfer.geo_funcs import VectorIndexer, AngleIndexer, CurvatureIndexer
from numpy.linalg import norm

@pytest.fixture
def square():
    
    return sq

class TestGeoFunc:
    def test_vec_indexer(self):
        square = np.array([
            [1,1],
            [1,-1],
            [-1,-1],
            [-1,1]
        ])
        v_i = VectorIndexer(square)
        assert (np.array_equal(v_i(0),square[1] - square[0]))
        
    def test_ang_indexer(self):
        square = np.array([
            [1,1],
            [1,-1],
            [-1,-1],
            [-1,1]
        ])
        theta_i = AngleIndexer(square)
        tol = 0.001
        right_angle = np.pi / 2
        
        assert (abs(theta_i(0) - right_angle) <= tol)
        
    def test_curvature_indexer(self):
        square = np.array([
            [1,1],
            [1,-1],
            [-1,-1],
            [-1,1]
        ])
        k_i = CurvatureIndexer(square)
        v_i = VectorIndexer(square)
        tol = 0.001
        
        assert(k_i(0))

if __name__ == '__main__':
    print("Test")
    
    
    