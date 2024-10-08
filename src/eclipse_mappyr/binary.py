#!/usr/bin/env python3
#
# File: binary.py
# Authors: Timothy Van Reeth <timothy.vanreeth@kuleuven.be> ; Cole Johnston <colej@mpa-garching.mpg.de>
# License: GPL-3+
# Description: Module that builds a basic model of a circular, synchronised
#              binary, based on a subset of basic input parameters.

import numpy as np
import astropy.units as au

from star import spherical_star as star
from transforms import spherical_to_cartesian

class circular_binary(object):

    def __init__(self, cfg):
        """
            Reading in the subset of basic parameters describing the circular, synchronised binary system.

            Parameters:
                P_orb:               astropy quantity
                                     the orbital period of the simulated system
                incl_deg:            float
                                     the inclination angle of the binary orbit (in degrees)
                light_contribution1: float; optional
                                     the relative light contribution of the primary component
                                     (default value = 0.5)
                light_contribution2: float; optional
                                     the relative light contribution of the secondary component
                                     (default value = 0.5)

                WARNING: light_contribution1 + light_contribution2 = 1!!
        """

        self._period = cfg['binary']['P_orb'] * au.day
        self._incl_deg = cfg['binary']['incl_deg']

        # setting default values for the component parameters
        self._primary = star(cfg['primary']['mass']*au.Msun,
                             cfg['primary']['radius']*au.Rsun,
                             self._incl_deg + cfg['primary']['obliquity'],
                             1./self._period, cfg['primary']['ld_coeff']
                             )

        self._secondary = star(cfg['secondary']['mass']*au.Msun,
                             cfg['secondary']['radius']*au.Rsun,
                             self._incl_deg + cfg['secondary']['obliquity'],
                             1./self._period, cfg['primary']['ld_coeff']
                             )

        self._light_contribution1 = cfg['primary']['light_contribution']
        self._light_contribution2 = cfg['secondary']['light_contribution']


        print('Binary Incl: {}'.format(self.incl_deg))
        print('Primary Incl: {}'.format(self.primary.incl_deg))
        print('Secondary Incl: {}'.format(self.secondary.incl_deg))
        return


    @property
    def period(self):
        return self._period.to(au.s)

    @property
    def freq_orb(self):
        return 1. / self._period.to(au.d)

    @property
    def incl_deg(self):
        return self._incl_deg

    @property
    def incl(self):
        return self._incl_deg * np.pi / 180.

    @property
    def light_contribution1(self):
        return self._light_contribution1

    @property
    def light_contribution2(self):
        return self._light_contribution2

    @property
    def semi_a(self):
        G_grav = 6.6743 * 10.**(-8.) * au.dyne * au.cm**2. / au.g**2.
        _semi_a = ( G_grav * (self._primary.Mstar + self._secondary.Mstar) * self._period**2. / (4.*np.pi**2.) )**(1./3.)
        return _semi_a.to(au.AU)

    @property
    def semi_a1(self):
        _a1 = self.semi_a * self._secondary.Mstar / (self._primary.Mstar + self._secondary.Mstar)
        return _a1.to(au.AU)

    @property
    def semi_a2(self):
        _a2 = self.semi_a * self._primary.Mstar / (self._primary.Mstar + self._secondary.Mstar)
        return _a2.to(au.AU)

    @property
    def coord1_co(self):
        return (-self.semi_a1, 0., 0.)

    @property
    def coord2_co(self):
        return (self.semi_a2, 0., 0.)

    @property
    def primary(self):
        return self._primary

    @property
    def secondary(self):
        return self._secondary

    @period.setter
    def period(self, Porb):
        self._period = Porb

    @incl_deg.setter
    def incl_deg(self, incl_degrees):
        self._incl_deg = incl_degrees

    @light_contribution1.setter
    def light_contribution1(self, lc1):
        self._light_contribution1 = lc1

    @light_contribution2.setter
    def light_contribution2(self, lc2):
        self._light_contribution2 = lc2


    def pot_U1(self, x, y, z, test_mass=1.*au.g):
        """
            Gravitational potential energy of the primary

            Parameters:
                self:        circular_binary object
                             the studied binary system
                x:           astropy quantity array
                             the x-coordinate of the evaluated point in the corotating frame
                y:           astropy quantity array
                             the y-coordinate of the evaluated point in the corotating frame
                z:           astropy quantity array
                             the z-coordinate of the evaluated point in the corotating frame
                test_mass:   astropy quantity; optional
                             the test mass used to calculate the potential energy
                             (default value = 1g)

            Returns:
                _U1:         astropy quantity
                             the potential enegry of the primary at the given coordinates.
        """

        G_grav = 6.6743 * 10.**(-8.) * au.dyne * au.cm**2. / au.g**2.
        _U1 = -G_grav * self._primary.Mstar * test_mass / np.sqrt((x - self.coord1_co[0])**2. + (y - self.coord1_co[1])**2. + (z - self.coord1_co[2])**2.)

        return _U1.to(au.erg)



    def pot_U2(self, x, y, z, test_mass=1.*au.g):
        """
            Gravitational potential energy of the secondary

            Parameters:
                self:        circular_binary object
                             the studied binary system
                x:           astropy quantity array
                             the x-coordinate of the evaluated point in the corotating frame
                y:           astropy quantity array
                             the y-coordinate of the evaluated point in the corotating frame
                z:           astropy quantity array
                             the z-coordinate of the evaluated point in the corotating frame
                test_mass:   astropy quantity; optional
                             the test mass used to calculate the potential energy
                             (default value = 1g)

            Returns:
                _U2:         astropy quantity
                             the potential enegry of the secondary at the given coordinates.
        """

        G_grav = 6.6743 * 10.**(-8.) * au.dyne * au.cm**2. / au.g**2.
        _U2 = -G_grav * self._secondary.Mstar * test_mass / np.sqrt((x - self.coord2_co[0])**2. + (y - self.coord2_co[1])**2. + (z - self.coord2_co[2])**2.)

        return _U2.to(au.erg)



    def kin_co(self, x, y, z, test_mass=1.*au.g):
        """
            kinetic energy in the corotating frame for a non-moving test_mass

            Parameters:
                self:        circular_binary object
                             the studied binary system
                x:           astropy quantity array
                             the x-coordinate of the evaluated point in the corotating frame
                y:           astropy quantity array
                             the y-coordinate of the evaluated point in the corotating frame
                z:           astropy quantity array
                             the z-coordinate of the evaluated point in the corotating frame
                test_mass:   astropy quantity; optional
                             the test mass used to calculate the kinetic energy
                             (default value = 1g)

            Returns:
                _T:          astropy quantity
                             the kinetic energy of the test mass
        """

        _T = 0.5 * test_mass * 4. * np.pi**2. * (x**2. + y**2.) / self._period**2.

        return _T.to(au.erg)



    def gen_potential(self, x, y, z, test_mass=1.*au.g):
        """
            Generalised potential in the corotating frame of the binary system

            Parameters:
                self:        circular_binary object
                             the studied binary system
                x:           astropy quantity array
                             the x-coordinate of the evaluated point in the corotating frame
                y:           astropy quantity array
                             the y-coordinate of the evaluated point in the corotating frame
                z:           astropy quantity array
                             the z-coordinate of the evaluated point in the corotating frame
                test_mass:   astropy quantity; optional
                             the test mass used to calculate the generalised potential
                             (default value = 1g)

            Returns:
                _E:          astropy quantity
                             the generalised potential
        """

        _E = -self.kin_co(x, y, z, test_mass) + self.pot_U1(x, y, z, test_mass) + self.pot_U2(x, y, z, test_mass)

        return _E.to(au.erg)



    def potential_at_surface(self, component=1, test_mass=1.*au.g):
        """
            Generalised potential at the surface of a selected component of the binary system

            Parameters:
                self:        circular_binary object
                             the studied binary system
                component:   int; optional
                             the component at whose surface we want to determine the potential
                             (default = 1 = primary; alternative = 2 = secondary).
                test_mass:   astropy quantity; optional
                             the test mass used to calculate the generalised potential
                             (default value = 1g)

            Returns:
                pot_at_surf: astropy quantity
                             the generalised potential at the surface of the selected stellar component
        """

        if(component == 1):
            x_co = (self.semi_a1**2. - self.R1**2.) / self.semi_a1
            y_co = np.sqrt(self.semi_a1**2. - self.R1**2.) * self.R1 / self.semi_a1
            pot_at_surf = self.gen_potential(x_co, y_co, 0.*au.AU, test_mass=test_mass)

        elif(component == 2):
            x_co = (self.semi_a2**2. - self.R2**2.) / self.semi_a2
            y_co = np.sqrt(self.semi_a2**2. - self.R2**2.) * self.R2 / self.semi_a2
            pot_at_surf = self.gen_potential(x_co, y_co, 0.*au.AU, test_mass=test_mass)

        return pot_at_surf



    def coord1(self, orb_phase):
        """
            Cartesian coordinates of the primary in the inertial reference
            frame of the observer, at a selected orbital phase.

            Parameters:
                self:        circular_binary object
                             the studied binary system
                orb_phase:   float
                             the selected orbital phase

            Returns:
                xcoord:      astropy quantity array
                             the x-coordinate of the primary
                ycoord:      astropy quantity array
                             the y-coordinate of the primary
                zcoord:      astropy quantity array
                             the z-coordinate of the primary
        """

        xcoord = -self.semi_a1.to(au.Rsun) * np.cos(2.*np.pi*orb_phase) * np.cos(self.incl)
        ycoord = -self.semi_a1.to(au.Rsun) * np.sin(2.*np.pi*orb_phase)
        zcoord = -self.semi_a1.to(au.Rsun) * np.cos(2.*np.pi*orb_phase) * np.sin(self.incl)

        return (xcoord,ycoord,zcoord)



    def coord2(self, orb_phase):
        """
            Cartesian coordinates of the secondary in the inertial reference
            frame of the observer, at a selected orbital phase.

            Parameters:
                self:        circular_binary object
                             the studied binary system
                orb_phase:   float
                             the selected orbital phase

            Returns:
                xcoord:      astropy quantity array
                             the x-coordinate of the secondary
                ycoord:      astropy quantity array
                             the y-coordinate of the secondary
                zcoord:      astropy quantity array
                             the z-coordinate of the secondary
        """

        xcoord = self.semi_a2.to(au.Rsun) * np.cos(2.*np.pi*orb_phase) * np.cos(self.incl)
        ycoord = self.semi_a2.to(au.Rsun) * np.sin(2.*np.pi*orb_phase)
        zcoord = self.semi_a2.to(au.Rsun) * np.cos(2.*np.pi*orb_phase) * np.sin(self.incl)

        return (xcoord,ycoord,zcoord)



    def eclipse_masks(self, orb_phase):
        """
            Determining which parts of the visible stellar sufaces are not eclipsed
            by the other binary component at the given orbital phase.

            Parameters:
                self:        circular_binary object
                             the binary system for which the eclipse masks are determined
                orb_phase:   float
                             the selected orbital phase

            Returns:
                out_of_ecl1: numpy array (dtype=bool)
                             mask indicating which mesh grid cells on the surface of
                             the primary are eclipsed (False) or observable (True).
                out_of_ecl2: numpy array (dtype=bool)
                             mask indicating which mesh grid cells on the surface of
                             the secondary are eclipsed (False) or observable (True).
        """

        # central coordinates of the two components
        coord1_in = self.coord1(orb_phase)
        coord2_in = self.coord2(orb_phase)

        # case 1: primary eclipse
        if(coord1_in[2] < coord2_in[2]):
            xsurf,ysurf,zsurf = spherical_to_cartesian(self._primary.theta_incl,self._primary.phi_incl,rad=self._primary.Rstar.to(au.Rsun))
            out_of_ecl1 = (xsurf + coord1_in[0] - coord2_in[0])**2. + (ysurf + coord1_in[1] - coord2_in[1])**2. > self._secondary.Rstar.to(au.Rsun)**2.
            out_of_ecl2 = np.array(np.ones(self._secondary.theta_incl.shape), dtype=bool)

        else:
            xsurf,ysurf,zsurf = spherical_to_cartesian(self._secondary.theta_incl,self._secondary.phi_incl,rad=self._secondary.Rstar.to(au.Rsun))
            out_of_ecl2 = (xsurf + coord2_in[0] - coord1_in[0])**2. + (ysurf + coord2_in[1] - coord1_in[1])**2. > self._primary.Rstar.to(au.Rsun)**2.
            out_of_ecl1 = np.array(np.ones(self._primary.theta_incl.shape), dtype=bool)

        return out_of_ecl1, out_of_ecl2
