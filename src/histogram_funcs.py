import numpy as np
from numpy.typing import ArrayLike
from numpy import histogram
from numpy.linalg import norm


from geo_funcs import VectorIndexer, AngleIndexer, CurvatureIndexer
 


def HistMatch(e: ArrayLike, g: ArrayLike, max_iter=30, w1:float = 0.025, w2:float = 0.0045) -> ArrayLike:
    """Given a exemplar curve, e, and a guide curve, g.  Return a curve 't', such that
    the histogram of curvature of t matches g and the overall shape resembles e.

    Args:
        e (ArrayLike): Exemplar curve, the overall shape of which 't' should resemble
        g (ArrayLike): Guide curve, the curvature characteristics which 't' should have

    Returns:
        t (ArrayLike): The histogram of curvature of 't' should closely resemble 'g' and the overall shape
        should resemble 'e' 
    """
    n_bins = 100
    H_d =  HoCS(e, 100)
    p_k = -w1*np.log(H_d)
    
    
    
    return np.array()


def E_guide(t: ArrayLike, g: ArrayLike) -> float:
    """ Energy minimization function to ensure that the overall shape of t and g remain the same
    An implementation of eq(4) in Edy Technical report in docs
    
    Args:
        t (ArrayLike): An N-length array of 2-D points representing the guide curve
        g (ArrayLike): An N-length array of 2-D points representing the target curve

    Returns:
        float: Euclidean distance square between all points of the target curve and the guide curve
    """
    
    if t.shape != g.shape:
        raise Exception("Target curve and guide curve are of different dimensions")
    
    diff = t - g
    norm = np.linalg.norm(diff, axis=1)
    sqr = np.square(norm)
    
    return np.sum(sqr)

def E_hist(t: ArrayLike, H_d: ArrayLike) -> float:
    """ Energy minimzation function to ensure that the curvature statististics of t match the desired curve
    An implemention of eq(5) in Edy Technical report in docs

    Args:
        t (ArrayLike): The target curve
        H_d (ArrayLike): The precomputed HoCS of the desired curve

    Returns:
        float: Overall energy discrepancy
    """
    H_t = HoCS(t)
    
    
    
    return -1.


def HoCS(shape: ArrayLike, n_bins: int=20) -> ArrayLike:
    """Return the curvature of histogram of a given closed shape represented by some polyline as described in the following paper
    Implementation of eq(1,2) in Edy Technical report in docs

    Args:
        shape (ArrayLike): A polyline curve.  A nx2 array of points where  each row is a 2-d point
        mask_size (int): The size of the mask
        n_bins (int): the number of bins needed
    Returns:
        ArrayLike: A 1-D array containing the histogram itself with 'n_bins'
    """
    N: int = shape.shape[0]
    k_i = CurvatureIndexer(shape)
    
    data = []
    for i in range(N - 2):
        data.append(k_i(i))
        
    try:
        return histogram(data, bins='auto', range=(0,2))
    except Exception as e:
        return histogram(data, bins=n_bins, range=(0,2))


if __name__ == "__main__":
    print("Test")
    
    from matplotlib.patches import CirclePolygon
    import matplotlib.pyplot as plt
    
    circ = CirclePolygon((0,0), radius=100, resolution=100)
    
    print(HoCS(circ.get_verts()))
    