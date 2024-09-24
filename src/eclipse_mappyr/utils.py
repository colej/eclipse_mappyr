import os
import sys
import glob
import yaml


from .binary import circular_binary
from .pulsations import pulsation_mode


def build_binary(inlist_filename='./inlist.yaml'):
    """
        Read in the required variables to build a simple binary model.

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
