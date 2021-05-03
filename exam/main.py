from deterministic_SIR import get_testSIR, flatten_the_curve, vaccination
from plots import *



def testSIR():
    result = get_testSIR()
    plotSIR(*result)
    plotI(*result)


def flatten():
    plot_maxI(*flatten_the_curve())


def vax():
    plot_vacc(*vaccination())



# testSIR()
# flatten()
vax()