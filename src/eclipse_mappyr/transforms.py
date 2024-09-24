import numpy as np

def spherical_to_cartesian(theta,phi,rad=None):
    """
        Convert spherical coordinates to cartesian coordinates.

        Parameters:
            theta: numpy array
                   colatitude (in rad)
            phi:   numpy array
                   azimuthal angle (in rad)
            rad:   numpy array; optional (default = None)
                   the radial coordinates

        Returns:
            x:     numpy array
                   the Cartesian coordinates on the x-axis
            y:     numpy array
                   the Cartesian coordinates on the y-axis
            z:     numpy array
                   the Cartesian coordinates on the z-axis
    """

    if(rad is None):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
    else:
        x = rad * np.sin(theta) * np.cos(phi)
        y = rad * np.sin(theta) * np.sin(phi)
        z = rad * np.cos(theta)

    return x,y,z
