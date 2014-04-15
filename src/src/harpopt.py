#!/usr/bin/env python
# encoding: utf-8
"""
harpopt.py

Created by Andrew Ning on 2014-01-13.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import Assembly
from openmdao.main.datatypes.api import Float
try:
    from pyopt_driver.pyopt_driver import pyOptDriver
except Exception:
    pass
from openmdao.lib.drivers.api import SLSQPdriver
from openmdao.lib.casehandlers.api import DumpCaseRecorder

from rotorse.rotoraerodefaults import common_io_with_ccblade, common_configure_with_ccblade



class HARPOptCCBlade(Assembly):

    def __init__(self, varspeed=True, varpitch=True, cdf_type='weibull', use_snopt=False):
        self.varspeed = varspeed
        self.varpitch = varpitch
        self.use_snopt = use_snopt
        self.cdf_type = cdf_type
        super(HARPOptCCBlade, self).__init__()


    def configure(self):
        common_io_with_ccblade(self, self.varspeed, self.varpitch, self.cdf_type)
        common_configure_with_ccblade(self, self.varspeed, self.varpitch, self.cdf_type)

        # used for normalization of objective
        self.add('AEP0', Float(1.0, iotype='in', desc='used for normalization'))

        if self.use_snopt:
            self.replace('driver', pyOptDriver())
            self.driver.optimizer = 'SNOPT'
            # self.driver.optimizer = 'PSQP'
            self.driver.options = {'Major feasibility tolerance': 1e-6,
                                   'Minor feasibility tolerance': 1e-6,
                                   'Major optimality tolerance': 1e-5,
                                   'Function precision': 1e-8,
                                   'Iterations limit': 500,
                                   'Print file': 'harpopt_snopt.out',
                                   'Summary file': 'harpopt_snopt_summary.out'}
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
        if self.varspeed:
            self.driver.add_parameter('control.tsr', low=3.0, high=14.0)
        # if optimize_stations:
            # self.driver.add_parameter('r_af[1:-1]', low=0.01, high=0.99)


        # outfile = open('results.txt', 'w')
        # self.driver.recorders = [DumpCaseRecorder(outfile)]
        self.driver.recorders = [DumpCaseRecorder()]


        if self.use_snopt:  # pyopt has an oustanding bug for unconstrained problems, so adding inconsequential constraint
            self.driver.add_constraint('spline.r_max_chord > 0.0')

