from deterministic_SIR import get_testSIR, flatten_the_curve, vaccination
from stochastic_SIR import get_test_stoch, prob_disappear
from SEIIaR import get_test_SEIIAR, stay_home
from SEIIaR_commute import get_test_SEIIaR_commute, get_two_towns, get_nine_towns
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
    result0 = get_testSIR()
    result = get_test_SEIIAR()
    # plotSEIIaRs(result0, result)
    plotEav(result)


def test_isolation():
    plotEsafrs(stay_home(), frac=1)


def testSEIIaR_commute():
    result0 = get_testSIR()
    result = get_test_SEIIaR_commute()
    plotSEIIaRs(result0, result)
    plotEav(result)


def two_towns():
    plot_two_towns(get_two_towns())

def nine_towns():
    plot_many_towns(get_nine_towns())



# testSIR()
# flatten()
# vax()
# test_stoch()
# disappear()
# testSEIIaR()
# test_isolation()
# testSEIIaR_commute()
two_towns()
# nine_towns()