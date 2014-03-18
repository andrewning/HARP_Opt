#!/usr/bin/env python
# encoding: utf-8
"""
HARP_Opt.py

Created by Andrew Ning on 2014-01-13.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Assembly
from openmdao.main.datatypes.api import Int, Float, Array, List, Str, VarTree, Slot
try:
    from pyopt_driver.pyopt_driver import pyOptDriver
except Exception:
    pass
from openmdao.lib.drivers.api import SLSQPdriver
from openmdao.lib.casehandlers.api import DumpCaseRecorder

from rotorse.rotoraerodefaults import RotorAeroVSWithCCBlade
from rotorse.rotoraero import VarSpeedMachine, DrivetrainLossesBase, CDFBase, RatedConditions



class HARPOptCCBlade(Assembly):

    # geometry
    r_max_chord = Float(iotype='in')
    chord_sub = Array(iotype='in', units='m', desc='chord at control points')
    theta_sub = Array(iotype='in', units='deg', desc='twist at control points')
    Rhub = Float(iotype='in', units='m', desc='hub radius')
    Rtip = Float(iotype='in', units='m', desc='tip radius')
    precone = Float(0.0, iotype='in', desc='precone angle', units='deg')
    tilt = Float(0.0, iotype='in', desc='shaft tilt', units='deg')
    yaw = Float(0.0, iotype='in', desc='yaw error', units='deg')
    B = Int(3, iotype='in', desc='number of blades')

    # airfoil
    r_af = Array(iotype='in', units='m', desc='locations where airfoils are defined on unit radius')
    airfoil_files = List(Str, iotype='in', desc='names of airfoil file')
    idx_cylinder = Int(iotype='in', desc='location where cylinder section ends on unit radius')

    # site characteristics
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    mu = Float(1.81206e-5, iotype='in', units='kg/m/s', desc='dynamic viscosity of air')
    shearExp = Float(0.2, iotype='in', desc='shear exponent')
    hubHt = Float(iotype='in', units='m')

    # control
    control = VarTree(VarSpeedMachine(), iotype='in')

    # options
    nSector = Int(4, iotype='in', desc='number of sectors to divide rotor face into in computing thrust and power')
    npts_coarse_power_curve = Int(20, iotype='in', desc='number of points to evaluate aero analysis at')
    npts_spline_power_curve = Int(200, iotype='in', desc='number of points to use in fitting spline to power curve')
    AEP_loss_factor = Float(1.0, iotype='in', desc='availability and other losses (soiling, array, etc.)')

    # used for normalization of objective
    AEP0 = Float(1.0, iotype='in', desc='used for normalization')

    # slots (must replace)
    dt = Slot(DrivetrainLossesBase)
    cdf = Slot(CDFBase)

    # outputs
    AEP = Float(iotype='out', units='kW*h', desc='annual energy production')
    V = Array(iotype='out', units='m/s', desc='wind speeds (power curve)')
    P = Array(iotype='out', units='W', desc='power (power curve)')
    ratedConditions = VarTree(RatedConditions(), iotype='out')
    diameter = Float(iotype='out', units='m')


    def __init__(self, use_snopt=False):
        self.use_snopt = use_snopt
        super(HARPOptCCBlade, self).__init__()


    def configure(self):
        RotorAeroVSWithCCBlade.configure_assembly(self)

        if self.use_snopt:
            self.replace('driver', pyOptDriver())
            self.driver.optimizer = 'SNOPT'
            self.driver.options = {'Major feasibility tolerance': 1e-6,
                                   'Minor feasibility tolerance': 1e-6,
                                   'Major optimality tolerance': 1e-5,
                                   'Function precision': 1e-8,
                                   'Iterations limit': 500}
        else:
            self.replace('driver', SLSQPdriver())
            self.driver.accuracy = 1.0e-6
            self.driver.maxiter = 500

        # objective
        self.driver.add_objective('-aep.AEP/AEP0')  # maximize AEP

        # design variables
        self.driver.add_parameter('r_max_chord', low=0.1, high=0.5)
        self.driver.add_parameter('chord_sub', low=0.4, high=5.3)
        self.driver.add_parameter('theta_sub', low=-10.0, high=30.0)
        self.driver.add_parameter('control.tsr', low=3.0, high=14.0)
        # if optimize_stations:
        #     self.driver.add_parameter('r_af', low=0.01, high=0.99)


        # outfile = open('resultso.txt', 'w')
        # self.driver.recorders = [DumpCaseRecorder(outfile)]
        self.driver.recorders = [DumpCaseRecorder()]


        if self.use_snopt:  # pyopt has an oustanding bug for unconstrained problems, so adding inconsequential constraint
            self.driver.add_constraint('spline.r_max_chord > 0.0')

