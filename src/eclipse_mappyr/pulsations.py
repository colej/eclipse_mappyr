#!/usr/bin/env python3
#
# File: pulsation.py
# Author: Cole Johnston <cole.johnston@ru.nl>
# Coauthor: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Module describing the basic properties of a p-mode pulsation


import sys
import numpy as np
import astropy.units as au
import astropy.constants as ac

from scipy.special import sph_harm


def dsph_harm_dtheta(theta,phi,l=2,m=1):
    """
    Derivative of spherical harmonic wrt colatitude.
    Using Y_l^m(theta,phi).
    Equation::
        sin(theta)*dY/dtheta = (l*J_{l+1}^m * Y_{l+1}^m - (l+1)*J_l^m * Y_{l-1,m})
    E.g.: Phd thesis of Joris De Ridder
    """
    if abs(m)>=l:
        Y = 0.
    else:
        factor = 1./np.sin(theta)
        term1 = l     * norm_J(l+1,m) * sph_harm(theta,phi,l+1,m).real
        term2 = (l+1) * norm_J(l,m)   * sph_harm(theta,phi,l-1,m).real
        Y = factor * (term1 - term2)
    return Y/np.sin(theta)


def dsph_harm_dphi(theta,phi,l=2,m=1):
    """
    Derivative of spherical harmonic wrt longitude.
    Using Y_l^m(theta,phi).
    Equation::
        dY/dphi = i*m*Y
    """
    return 1j*m*sph_harm(theta,phi,l,m).real


def norm_J(l,m):
    """
    Normalisation factor
    """
    if abs(m)<l:
        J = np.sqrt( float((l**2-m**2))/(4.*l**2-1.))
    else:
        J = 0
    return J


class pmode_pulsation(object):

    def __init__(self, cfg, frot):
        """
            Loading in the TAR module from GYRE v5.x or v6.x, and reading in
            additional parameters describing the studied g-mode.

            Parameters:
                puls_freq:          astropy quantity
                                    (observed) cyclic pulsation frequency in the
                                    inertial reference frame
                puls_ampl:          float
                                    peak amplitude of the pulsation on the stellar
                                    surface (unit = mmag)
                frot:               astropy quantity
                                    cyclic rotation frequency of the studied star
                k:                  int
                                    latitudinal degree of the g-mode
                m:                  int
                                    azimuthal order of the g-mode
        """


        self._l = cfg['pulsation']['lval']
        self._m = cfg['pulsation']['mval']
        ## Assumes synchronised so frot = forb
        self._puls_freq_co = (np.abs(cfg['pulsation']['puls_freq'] - self._m*frot.value)) / au.day
        self._puls_freq = (self._puls_freq_co.value + self._m*frot.value) / au.day
        self._amplitude = cfg['pulsation']['puls_amp']

        self._mode_type = cfg['pulsation']['mode_type']

        self.Lr = 0.    # Radial component
        self.Lp = 0.    # Phi component
        self.Lt = 0.    # Theta component

        return


    @property
    def puls_freq(self):
        return self._puls_freq

    @property
    def puls_freq_co(self):
        return self._puls_freq_co

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def l(self):
        return self._l

    @property
    def m(self):
        return self._m

    @property
    def mode_type(self):
        return self._mode_type


    def calculate_puls_geometry(self, star):

        bn_div_an = (ac.G.to('Rsun3/(Msun day2)').value * star.Mstar)/(self._puls_freq_co**2 * star.Rstar **3)

        self.Lr = sph_harm(self._m, self._l, star.theta_incl, star.phi_incl).real
        self.Lt = bn_div_an * dsph_harm_dtheta(star.theta_incl,star.phi_incl,
                                               l=self._l,m=self._m)
        self.Lp = (bn_div_an/np.sin(star.theta)) * dsph_harm_dphi(star.theta_incl,star.phi_incl,
                                               l=self._l,m=self._m)


        return
