#!/usr/bin/env python3
#
# File: tidal_perturbation_in_circular_binary.py
# Authors: Timothy Van Reeth <timothy.vanreeth@kuleuven.be> ; Cole Johnston <colej@mpa-garching.mpg.de>
# License: GPL-3+
# Description: Calculating the observed flux variations of a tidally
#              distorted g-mode pulsation in a circular, synchronised
#              binary system


import os
import sys
import glob
import yaml
import numpy as np
import subprocess as sp
import astropy.units as au

from progress.bar import Bar
from .binary import circular_binary
from .pulsations import pulsation_mode





def calculate_flux(binary, pulsation, distortion_factor=1., puls_phase=0., orb_phase=0.):
    """
        Calculate the observed flux of a binary star with a perturbed g-mode pulsation at a
        given pulsation and orbital phase, and indicate if the data point was simulated
        during an eclipse.

        Parameters:
            binary:            circular_binary object
                               the studied circular, synchronised binary system
            pulsation:         gmode_pulsation object
                               the g-mode pulsation that will be perturbed and evaluated
            distortion_factor: float; optional
                               the scaling factor of the mode visibility
                                across the stellar surface (default = 1)
            puls_phase:         float; optional
                                current phase of the pulsation cycle (in the inertial frame)
                                (default = 0.)
            orb_phase:          float; optional
                                current phase of the orbital cycle
                                (default = 0.)

        Returns:
            bin_iflux:          float
                                the simulated flux caused by (solely) the binarity
            tot_iflux:          float
                                the total simulated flux at the provided pulsational and
                                orbital phases
            puls_iflux:         float
                                the simulated flux caused by (solely) the distorted pulsation
            ecl_iflag:          int
                                flag indicating if the calculated data point occurs during an
                                eclipse
    """

    ecl_maps = binary.eclipse_masks(orb_phase)
    primary_vissurf_bool = binary.primary.theta_incl < np.pi/2.
    secondary_vissurf_bool = binary.secondary.theta_incl < np.pi/2.

    primary_mask = np.array(ecl_maps[0] & primary_vissurf_bool, dtype=float)
    secondary_mask = np.array(ecl_maps[1] & secondary_vissurf_bool, dtype=float)

    primary_vissurf = np.array(primary_vissurf_bool, dtype=float)
    secondary_vissurf = np.array(secondary_vissurf_bool, dtype=float)

    if(binary.primary.pulsating):
        norm_puls = calculate_normalised_pulsations(binary.primary, pulsation, puls_phase)
        # puls_scaling = distort_pulsation(binary.primary, orb_phase=orb_phase, distortion_factor=distortion_factor)
        puls_scaling = 1.
        primary_puls = 10.**(-0.0004 * pulsation.amplitude * norm_puls * puls_scaling)
        secondary_puls = np.ones(binary.secondary.theta.shape)
    else:
        norm_puls = calculate_normalised_pulsations(binary.secondary, pulsation, puls_phase)
        # puls_scaling = distort_pulsation(binary.secondary, orb_phase=orb_phase, distortion_factor=distortion_factor)
        puls_scaling = 1.
        secondary_puls = 10.**(-0.0004 * pulsation.amplitude * norm_puls * puls_scaling)
        primary_puls = np.ones(binary.primary.theta.shape)

    primary_totflux   = np.nansum(2. * np.cos(binary.primary.theta_incl)   * binary.primary.cell_weight   * binary.primary.limb_darkening()   * primary_puls   * primary_mask)
    secondary_totflux = np.nansum(2. * np.cos(binary.secondary.theta_incl) * binary.secondary.cell_weight * binary.secondary.limb_darkening() * secondary_puls * secondary_mask)
    primary_binflux   = np.nansum(2. * np.cos(binary.primary.theta_incl)   * binary.primary.cell_weight   * binary.primary.limb_darkening()   * primary_mask)
    secondary_binflux = np.nansum(2. * np.cos(binary.secondary.theta_incl) * binary.secondary.cell_weight * binary.secondary.limb_darkening() * secondary_mask)
    primary_refflux   = np.nansum(2. * np.cos(binary.primary.theta_incl)   * binary.primary.cell_weight   * binary.primary.limb_darkening()   * primary_vissurf)
    secondary_refflux = np.nansum(2. * np.cos(binary.secondary.theta_incl) * binary.secondary.cell_weight * binary.secondary.limb_darkening() * secondary_vissurf)

    tot_iflux = -2500.*np.log10( (binary.light_contribution1*primary_totflux/primary_refflux) + (binary.light_contribution2*secondary_totflux/secondary_refflux))
    bin_iflux = -2500.*np.log10( (binary.light_contribution1*primary_binflux/primary_refflux) + (binary.light_contribution2*secondary_binflux/secondary_refflux))
    puls_iflux = tot_iflux - bin_iflux

    if(ecl_maps[0].all() & ecl_maps[1].all()):
        ecl_iflag = 0
    else:
        ecl_iflag = 1

    return bin_iflux, tot_iflux, puls_iflux, ecl_iflag


def calculate_normalised_pulsations(star, pulsation, puls_phase):

    """
    Wrapper function to get a p mode or g mode pulsation normalised to the
    maximum amplitude at the stellar surface

    Parameters:
        star:           stellar_model object
                        the pulsating star  (primary or secondary)
        pulsation:      gmode_pulsation object
                        the simulated g-mode pulsation
        puls_phase:     float
                        the current phase of the studied pulsation (as seen
                        by the observer)

    Returns:
        norm_puls:      numpy array
                        normalised pulsation variability of the pulsation mode at the
                        stellar surface at the phase puls_phase

    # g modes currently not suppored
    """

    if pulsation.mode_type == 'p':
        return calculate_normalised_pmode_pulsations(star, pulsation, puls_phase)
    else:
        print('g modes are currently unsupported')
        sys.exit()



def calculate_normalised_pmode_pulsations(star, pulsation, puls_phase):
    """
        Converting the geomety of the calculated g-mode pulsation to temperature
        variations, Lagrangian displacements and the associated velocity field

        Parameters:
            star:           stellar_model object
                            the pulsating star
            pulsation:      p mode_pulsation object
                            the simulated p mode pulsation
            puls_phase:     float
                            the current phase of the studied pulsation (as seen
                            by the observer)

        Returns:
            norm_puls:      numpy array
                            normalised pulsation variability of the g-mode at the
                            stellar surface at the phase puls_phase
    """

    if(pulsation.m < 0.):
        sign = 1.
    else:
        sign = -1.

    norm_puls = (pulsation.Lr * np.cos(pulsation.m*star.phi + 2.*np.pi*sign*puls_phase)) / np.nanmax(pulsation.Lr * np.cos(pulsation.m*star.phi + 2.*np.pi*sign*puls_phase))

    return norm_puls







if __name__ == "__main__":

    # Reading the input parameters / variables
    maindir, binary, pulsation, distortion_factor, \
    N_forb_cycles, Nsample_per_cycle = read_inlist(sys.argv[1])
    print('Inlist read')

    # Setting the output directory and copying the used inlist
    mass1 = binary.primary.Mstar.to(au.Msun).value
    mass2 = binary.secondary.Mstar.to(au.Msun).value
    freq_orb = binary.freq_orb.to(1./au.d).value
    outdir = f'{maindir}binary_M{int(round(100.*mass1))}M{int(round(100.*mass2))}_forb{int(round(1000.*freq_orb))}_i{int(binary.incl_deg)}/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print('Directory created')
    print('Directory set')

    computation_nr = f"{len(glob.glob(f'{outdir}inlist*.yaml')) + 1}".zfill(3)
    sp.call(f'cp ./inlist.yaml {outdir}inlist{computation_nr}.yaml', shell=True)   # warning: do not forget to adapt this line if the inlist filename changes!

    # setting the time domain for tidal perturbation simulations
    time = np.linspace(0., float(N_forb_cycles), N_forb_cycles*Nsample_per_cycle+1) * binary.period.to(au.d)
    orb_phases = np.linspace(0.,float(N_forb_cycles),N_forb_cycles*Nsample_per_cycle+1) % 1.
    puls_phases = np.array(pulsation.puls_freq.to(1/au.d)*time, dtype=float) % 1.

    # time = np.loadtxt('tess_sector01_times.dat').T
    # time *= au.day
    # orb_phases = np.array( time * freq_orb, dtype=float) % 1.
    # puls_phases = np.array(pulsation.puls_freq.to(1/au.d)*time, dtype=float) % 1.


    print('Phase arrays constructed.')
    # Calculating the mode visibilities and kinetic energy
    binary_flux = []
    total_flux = []
    pulsation_flux = []
    eclipse_flags = []

    pbar = Bar('Calculating...', max=len(puls_phases))
    for iph,puls_phase,orb_phase in zip(np.arange(len(puls_phases)), puls_phases, orb_phases):

        # calculating the observed fluxes for (1) the binary + pulsation, (2) the binary, and (3) the pulsation, and provide (4) eclipse flags
        bin_iflux, tot_iflux, puls_iflux, ecl_iflag = calculate_flux(binary, pulsation, puls_phase=puls_phase, orb_phase=orb_phase, distortion_factor=distortion_factor)
        binary_flux.append(bin_iflux)
        total_flux.append(tot_iflux)
        pulsation_flux.append(puls_iflux)
        eclipse_flags.append(int(ecl_iflag))
        pbar.next()
    pbar.finish()

    binary_flux = np.array(binary_flux)
    total_flux = np.array(total_flux)
    pulsation_flux = np.array(pulsation_flux)
    eclipse_flags = np.array(eclipse_flags, dtype=int)

    # Saving the results
    save_results(f'{outdir}pmode_f{int(np.round(pulsation.puls_freq.value*1000000))}_perturbed-visibilities_{computation_nr}.dat', time, orb_phases, puls_phases, total_flux, binary_flux, pulsation_flux, eclipse_flags)
