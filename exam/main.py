from deterministic_SIR import get_testSIR, flatten_the_curve, vaccination
from stochastic_SIR import get_test_stoch, prob_disappear
from SEIIaR import get_test_SEIIAR, stay_home
from SEIIaR_commute import get_Norway, get_Norway_lockdown, get_pop_structure_lochkdown, get_test_SEIIaR_commute, get_two_towns, get_nine_towns, get_pop_structure
from plots import *



def testSIR():
    result = get_testSIR()
    plotSIR(*result, name="TestSIR", subdir="2A/")
    plotI(*result, name="TestI", subdir="2A/")


def flatten():
    plot_maxI(*flatten_the_curve(), name="flatten", subdir="2A/")


def vax():
    plot_vacc(*vaccination(), name="vax", subdir="2A/")


def test_stoch():
    result0 = get_testSIR()
    result = get_test_stoch()
    assert result0[-1]==result[-1] # should compare same args
    plotSIRs(result0, result, name="TestSIR_stoch", subdir="2B/")
    plotIs( result, name="TestI_stoch", subdir="2B/")


def disappear():
    plot_prob_dis(*prob_disappear(), name="disappear", subdir="2B/")


def testSEIIaR():
    result0 = get_testSIR()
    result = get_test_SEIIAR()
    plotSEIIaRs(result0, result, name="TestSEIIaR", subdir="2C/")


def test_isolation():
    plotEsafrs(stay_home(), frac=1, name="isolation", subdir="2C/")


def testSEIIaR_commute():
    result0 = get_testSIR()
    result = get_test_SEIIaR_commute()
    plotSEIIaRs(result0, result, name="TestSEIIaR_commute", subdir="2D/")
    plotEav(result)


def two_towns():
    plot_two_towns(get_two_towns(), name="two_towns", subdir="2D/")

def nine_towns():
    plot_many_towns(get_nine_towns(), fs=(32, 12), name="nine_towns", subdir="2D/")


def pop_struct():
    plot_pop_struct(get_pop_structure(), name="pop_struct", subdir="2D/")

def pop_struct_lockdown():
    plot_pop_struct(get_pop_structure_lochkdown(), name="pop_struct_lockdown", subdir="2D/")

def num_infected():
    result = get_Norway()
    plotOslo(result, name="Oslo", subdir="2D/")
    plot_sum_inf(result, name="num_infected", subdir="2D/")

def num_infected_lockdown():
    result = get_Norway_lockdown()
    plotOslo(result, name="Oslo_lockdown", subdir="2D/")
    plot_sum_inf(result, name="num_infected_lockdown", subdir="2D/")


# testSIR()
# flatten()
# vax()
# test_stoch()
# disappear()
# testSEIIaR()
# test_isolation()
# testSEIIaR_commute()
# two_towns()
# nine_towns()
# pop_struct()
# pop_struct_lockdown()
num_infected()
# num_infected_lockdown()