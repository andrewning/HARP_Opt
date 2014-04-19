# HARP_Opt

HARP_Opt (**H**\orizontal **A**xis **R**otor **P**erformance **Opt**imization) is a tool for rotor optimization.  Aerodynamic performance is computed using [CCBlade](http://wind.nrel.gov/designcodes/simulators/ccblade/), a blade element momentum code with guaranteed convergence.  Analytic gradients are provided for all components in the analysis so that gradient-based optimization can be used efficiently.  A number of open-source and commercial optimizers can be used as made available through the [OpenMDAO framework](http://openmdao.org/).  While only basic aerodynamic performance is included in this version, the framework allows HARP_Opt to be integrated and expanded with other [NREL System Engineering](http://www.nrel.gov/wind/systems_engineering.html) tools.  These tools incluee rotor structural analysis, nacelle/drivetrain analysis, tower aero/structural analysis, and various cost models.  These additional models are scheduled to be released later this year, and allow for more complete turbine optimization studies.

Author: [S. Andrew Ning](mailto:andrew.ning@nrel.gov)

<!-- ## User Information

If you came to this page directly without going through the NWTC Information Portal, **we would appreciate if you could [report your user information](http://wind.nrel.gov/designcodes/simulators/ccblade/downloaders/CCBlade_github_redirect.html) before cloning the repository**.  We use this information in order to allocate resources for supporting our software, and to notify users of critical updates.
 -->

## Prerequisites

Python, NumPy, SciPy, Fortran compiler

## Installation

t
Install *and activate* OpenMDAO as described [here](http://openmdao.org/docs/getting-started/install.html).  

Install PyOpt as an OpenMDAO plugin

    > plugin install --github pyopt_driver

Install HARP_Opt as an OpenMDAO plugin

    > python setup.py install

(or ``python setup.py develop`` if you plan to make changes with the underlying implementation)

Test the setup by running the example

    > python example/example.py

Create your own scripts by starting with a copy of example.py



<!-- ## Run Unit Tests

To check if installation was successful, run the unit tests

    $ python test/test_ccblade.py
    $ python test/test_gradients.py
 -->
## Detailed Documentation

Access the online version at <http://nrel-wisdem.github.io/CCBlade>


