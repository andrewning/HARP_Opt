#!/usr/bin/env python
# encoding: utf-8
"""
harpopt_example.py

Created by Andrew Ning on 2014-03-18.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np

from harpopt import HARPOptCCBlade
from rotorse.rotoraerodefaults import CSMDrivetrain, RayleighCDF, WeibullCDF


# use_snopt = False
use_snopt = True
rotor = HARPOptCCBlade(use_snopt)

# -------- start inputs -------------

# --- rotor geometry ---
rotor.r_max_chord = 0.23577
rotor.chord_sub = [3.2612, 4.5709, 3.3178, 1.4621]
rotor.theta_sub = [13.2783, 7.46036, 2.89317, -0.0878099]
rotor.Rhub = 1.5
rotor.Rtip = 63.0
rotor.precone = 2.5
rotor.tilt = -5.0
rotor.yaw = 0.0
rotor.B = 3

# --- airfoils ---
basepath = '/Users/Andrew/Dropbox/NREL/5MW_files/5MW_AFFiles/'

# load all airfoils
airfoil_types = [0]*8
airfoil_types[0] = basepath + 'Cylinder1.dat'
airfoil_types[1] = basepath + 'Cylinder2.dat'
airfoil_types[2] = basepath + 'DU40_A17.dat'
airfoil_types[3] = basepath + 'DU35_A17.dat'
airfoil_types[4] = basepath + 'DU30_A17.dat'
airfoil_types[5] = basepath + 'DU25_A17.dat'
airfoil_types[6] = basepath + 'DU21_A17.dat'
airfoil_types[7] = basepath + 'NACA64_A17.dat'

# place at appropriate radial stations
af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

n = len(af_idx)
af = [0]*n
for i in range(n):
    af[i] = airfoil_types[af_idx[i]]

rotor.airfoil_files = af
rotor.r_af = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667, 0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724])
rotor.idx_cylinder = 3

# --- site characteristics ---
rotor.rho = 1.225
rotor.mu = 1.81206e-5
rotor.shearExp = 0.2
rotor.hubHt = 80.0

cdf_type = 'rayleigh'

if cdf_type == 'rayleigh':
    rotor.replace('cdf', RayleighCDF())
    rotor.cdf.xbar = 6.0

elif cdf_type == 'weibull':
    rotor.replace('cdf', WeibullCDF())
    rotor.cdf.A = 6.0
    rotor.cdf.k = 2.0


# --- control settings ---
rotor.control.Vin = 3.0
rotor.control.Vout = 25.0
rotor.control.ratedPower = 5e6
rotor.control.minOmega = 0.0
rotor.control.maxOmega = 12.0
rotor.control.pitch = 0.0
rotor.control.tsr = 7.55


# --- drivetrain efficiency ---
rotor.replace('dt', CSMDrivetrain())
rotor.dt.missing_deriv_policy = 'assume_zero'  # TODO: openmdao bug remove later
rotor.dt.drivetrainType = 'geared'


# --- options ---
rotor.nSector = 4
rotor.npts_coarse_power_curve = 20
rotor.npts_spline_power_curve = 200
rotor.AEP_loss_factor = 1.0
# optimize_stations = True

# -------- end of inputs -------------


# # run once to get baseline for normalization
# rotor.driver.run_iteration()
# rotor.AEP0 = rotor.aep.AEP
# rotor.driver.run_iteration()

rotor.AEP0 = 9716743.98384


rotor.run()


