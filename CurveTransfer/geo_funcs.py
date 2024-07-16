""" 
Helpful functions for determining the geometry of a given curve
"""
import numpy as np
from numpy.typing import ArrayLike
from numpy.linalg import norm

def VectorIndexer(shape: ArrayLike):
    """Returns a vector indexing function for a given polyline
    Args:
        shape (ArrayLike): The polyline of a given shape

    Returns:
        (func): A vector indexer
    """
    N: int = shape.shape[0]
    
    def get_i_vector(i: int) -> ArrayLike:
        """ Returns the i-th vector

        Args:
            i (int): The i vector

        Returns:
            ArrayLike: A vector
        """
        return shape[(i + 1) % N] - shape[i % N]
    
    return get_i_vector

def AngleIndexer(shape: ArrayLike):
    """An indexer that returns the angle between the ith and i+1 vector

    Args:
        shape (ArrayLike): The polyline of a given shape

    Returns:
        (func): An Angle Indexer
    """
    v_i = VectorIndexer(shape) # Gets the i-th vector
    
    def get_i_angle(i: int) -> float:
        """Returns the angle between the i-th vector and the i+1 vector, i.e., returns angle at point i
        Args:
            i (int): The angle at point T_i
        Returns:
            float: The angle
        """
        return np.arccos( 
            v_i(i).dot(v_i(i+1))/(norm(v_i(i)*norm(v_i(i+1))))
        )
    
    return get_i_angle

def CurvatureIndexer(shape: ArrayLike):
    """Returns a curvature indexing function for a given shape

    Args:
        shape (ArrayLike): The polyline of the given shape

    Returns:
        (func): Curvature indexer
    """
    theta_i = AngleIndexer(shape)
    v_i = VectorIndexer(shape) # Gets the i-th vector
    def get_i_curvature(i: int) -> float:
        """Returns the curvature at the i-th point
        Args:
            i (int): I-th point on the shape

        Returns:
            float: The curvature at point i on the shape, t
        """
        return theta_i(i) / ((norm(v_i(i)) + norm(v_i(i+1)))/2)
    
    return get_i_curvature

if __name__ == '__main__':
    """
    shape = np.array([[0,1], [1,0], [99,99]])
    
    theta_i = AngleIndexer(shape)
    
    print(theta_i(0))
    """
    pass