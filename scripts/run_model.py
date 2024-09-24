#!/usr/bin/env python3
#
# File: tidal_perturbation_in_circular_binary.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
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
from binary import circular_binary
from pulsations import pmode_pulsation



def distort_pulsation(star, orb_phase=0., distortion_factor=1.):
    """
        Routine to scale the g-mode visibility as a function of the location on
        the stellar surface

        Parameters:
            star:               stellar_model object
                                the star in which the pulsation is distorted
            orb_phase:          float; optional
                                current phase of the orbital cycle
                                (default = 0.)
            distortion_factor:  float; optional
                                the scaling factor of the mode visibility
                                across the stellar surface (default = 1)

        Returns:
            puls_scaling:       numpy array

    """

    basefun = 0.5 * (3.*(np.sin(star.theta)*np.cos(star.phi - 2.*np.pi*orb_phase))**2. - 1.)
    puls_scaling = ( 2. * np.ones(star.theta.shape) / ( distortion_factor + 1. ) ) + ( (basefun - np.nanmin(basefun)) / (np.nanmax(basefun) - np.nanmin(basefun)) * 2. * ( (distortion_factor - 1.) / (distortion_factor + 1.) ) )

    return puls_scaling



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
            pulsation:      gmode_pulsation object
                            the simulated g-mode pulsation
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




def read_inlist(inlist_filename='./inlist.yaml'):
    """
        Read in the required variables to calculate r-mode visibilities,
        following the methodology from Saio et al. (2018).

        Parameters:
            inlist_filename:      string; optional (default: ./inlist.dat)
                                  the path to the inlist

        Returns:
            maindir:              string
                                  main work directory
            binary:               circular binary object
                                  the circular synchronised binary system that will be simulated
            pulsation:            gmode_pulsation object
                                  the g-mode pulsation that will be simulated
            distortion_factor:    float
                                  distortion factor of the simulated pulsation
            N_forb_cycles:        int
                                  number of orbital cycles to be simulated
            Nsample_per_cycle:    int
                                  number of simulated data points per orbital cycle
    """

    with open(inlist_filename,'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    f.close()

    # collecting the given parameter values within the appropriate objects
    binary = circular_binary(cfg)


    if cfg['pulsation']['mode_type'] == 'p':
        pulsation = pmode_pulsation(cfg, binary.freq_orb)
    elif cfg['pulsation']['mode_type'] == 'g':
        print('g modes are currently not supported')
        sys.exit()
    else:
        print('Mode type {} not recognised. Exiting.'.format(cfg['pulsation']['mode_type']))
        sys.exit()

    if(cfg['pulsation']['pulsating_star'] == 'primary'):
        binary.primary.pulsating = True
        pulsation.calculate_puls_geometry(binary.primary)
    elif(cfg['pulsation']['pulsating_star'] == 'secondary'):
        binary.secondary.pulsating = True
        pulsation.calculate_puls_geometry(binary.secondary)

    return cfg['general']['main_dir'], binary, pulsation, \
           cfg['pulsation']['distortion_factor'], \
           cfg['simulation']['N_forb_cycles'], \
           cfg['simulation']['Nsample_per_cycle']



def save_results(result_filename, time, orb_phases, puls_phases, total_flux, binary_flux, pulsation_flux, eclipse_flags):
    """
        Saving the calculated visibilities of the distorted g-mode pulsations

        Parameters:
            result_filename:    string
                                absolute path to the results output filename
            time:               astropy quantity array
                                the time stamps of the simulate data points
            orb_phases:         numpy array
                                orbital phases corresponding to the different time stamps
            puls_phases:        numpy array
                                pulsation phases at the different time stamps
            total_flux:         numpy array
                                the total observed flux variations at the different time stamps (unit: mmag)
            binary_flux:        numpy array
                                the flux variations from the binary motion at the different time stamps (unit: mmag)
            pulsation_flux:     numpy array
                                the flux variations from the pulsations at the different time stamps (unit: mmag)
            eclipse_flags:      numpy array of integers
                                flags indicating if the data point was taken during an eclipse (yes = 1; no = 0)

    """

    file = open(result_filename, 'w')

    headline = ' '*16 + 'time' + ' '*11 + 'orb_phase' + ' '*10 + 'puls_phase' + ' '*10 + 'total_flux' + ' '*14 + 'binary' + ' '*11 + 'pulsation' + ' '*13 + 'eclipse'
    file.write(f'{headline}\n')

    for itime, iorbphase, ipulsphase, iflux, ibin, ipuls, iflag in zip(time.to(au.d).value, orb_phases, puls_phases, total_flux, binary_flux, pulsation_flux, eclipse_flags):
        data = f'  {itime:18e}  {iorbphase:18e}  {ipulsphase:18e}  {iflux:18e}  {ibin:18e}  {ipuls:18e}                   {iflag}\n'
        file.write(data)
    file.close()

    return



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
