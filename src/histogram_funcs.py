import numpy as np
from numpy.typing import ArrayLike
from numpy import histogram
from numpy.linalg import norm


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
    """An implementation of eq(4)
    
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
    


def HoCS(shape: ArrayLike, n_bins: int=20) -> ArrayLike:
    """Return the curvature of histogram of a given closed shape represented by some polyline as described in the following paper
    https://www.dgp.toronto.edu/~egarfink/Edy_Technical_report.pdf

    Args:
        shape (ArrayLike): A polyline curve.  A nx2 array of points where  each row is a 2-d point
        mask_size (int): The size of the mask
        n_bins (int): the number of bins needed
    Returns:
        ArrayLike: A 1-D array containing the histogram itself with 'n_bins'
    """
    
    N: int = shape.shape[0]
    
    # The i-th vector
    v_i = lambda i: shape[(i + 1) % N] - shape[i % N]
    
    # Computes the cosine angle between v_i and v_{i+1}
    theta_i = lambda i: np.arccos(
        v_i(i).dot(v_i(i+1))/(norm(v_i(i)*norm(v_i(i+1))))
    )
    # Curvature estimate at point i
    k_i = lambda i: theta_i(i) / ((norm(v_i(i)) + norm(v_i(i+1)))/2)
    
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
    