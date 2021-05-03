from deterministic_SIR import get_testSIR, flatten_the_curve, vaccination
from stochastic_SIR import get_test_stoch, prob_disappear
from SEIIaR import get_test_SEIIAR
from plots import *



def testSIR():
    result = get_testSIR()
    plotSIR(*result)
    plotI(*result)


def flatten():
    plot_maxI(*flatten_the_curve())


def vax():
    plot_vacc(*vaccination())


def test_stoch():
    result0 = get_testSIR()
    result = get_test_stoch()
    assert result0[-1]==result[-1] # should compare same args
    plotSIRs(result0, result)
    plotIs( result)

def disappear():
    plot_prob_dis(*prob_disappear())


def testSEIIaR():
    plotSEIIaRs(get_testSIR(), get_test_SEIIAR())


# testSIR()
# flatten()
# vax()
# test_stoch()
# disappear()
testSEIIaR()