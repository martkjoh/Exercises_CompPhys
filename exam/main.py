from deterministic_SIR import *
from stochastic_SIR import *
from SEIIaR import *
from SEIIaR_commute import *
from plots import *
import pickle


def testSIR():
    result = get_testSIR()
    plotSIR(*result, name="TestSIR", subdir="2A/")
    plotI(*result, name="TestI", subdir="2A/")


def conv_det():
    plot_conv_det(*SIR_deterministic_convergence(), "conv", "2A/")    


def flatten():
    results = flatten_the_curve()
    plot_maxI(results, name="flatten", subdir="2A/")
    plot_flattening(results, name="flattenIs", subdir="2A/")


def vax():
    result = vaccination()
    plot_vacc(result, name="vax_R", subdir="2A/")
    plot_growth(result, name="vax", subdir="2A/")


def test_stoch():
    result0 = get_testSIR()
    result = get_test_stoch()
    assert result0[-1]==result[-1] # should compare same args
    plotSIRs(result0, result, name="TestSIR_stoch", subdir="2B/")
    plotIs( result, name="TestI_stoch", subdir="2B/")


def conv_stoch():
    plot_conv_stoch(*SIR_stochastic_convergence(), "conv", "2B/")    


def disappear(run=False):
    plot_prob_dis(*prob_disappear(run), name="disappear", subdir="2B/")


def testSEIIaR():
    result0 = get_testSIR()
    result = get_test_SEIIAR()
    plotSEIIaRs(result0, result, name="TestSEIIaR", subdir="2C/", fs=(8, 6))


def conv_SEIIaR():
    plot_conv_SEIIaR(*SEIIaR_convergence(), "conv", "2C/")


def test_isolation(run=False):
    result = stay_home(run)
    plotEsafrs(result, name="isolation", subdir="2C/", fs=(8, 6))
    plot_prob_gr(result, name="isolation2_prob", subdir="2C/")


def testSEIIaR_commute():
    result0 = get_testSIR()
    result = get_test_SEIIaR_commute()
    plotSEIIaRs(result0, result, name="TestSEIIaR_commute", subdir="2D/", fs=(8, 6))


def conv_SEIIaR_commute(run=False):
    plot_conv_SEIIaR(*SEIIaR_commute_convergence(run), "conv", "2D/", ci=0)


def two_towns():
    plot_two_towns(get_two_towns(), name="two_towns", subdir="2D/")


def two_towns2():
    plot_two_towns(get_two_towns2(), name="two_towns2", subdir="2D/")


def nine_towns():
    plot_many_towns(get_nine_towns(), fs=(32, 12), name="nine_towns", subdir="2D/")


def pop_struct():
    plot_pop_struct(get_pop_structure(), name="pop_struct", subdir="2D/")


def poplutaion():
    N = get_pop_structure()
    N = np.sum(N, axis=1)
    plot_towns(N, name="pops", subdir="2D/")


def pop_struct_lockdown():
    plot_pop_struct(get_pop_structure(
        lockdown=True), name="pop_struct_lockdown", subdir="2D/"
        )


def num_infected(lockdown=False, run=False):
    suffx = "lockdown" if lockdown else ""
    datapath = "data/norway"+suffx+".npy"
    result = get_Norway(datapath, lockdown, run)

    plot_sum_inf(result, name="num_infected"+suffx, subdir="2D/")
    xs = result[0]
    x = np.mean(xs, axis=0)
    result = (x, *result[1:])
    N = get_pop_structure()
    pop = np.sum(N, axis=1).astype(int)
    i2 = np.argmax(pop[1:]) + 1
    plot_town_i(result, 0, name="Oslo"+suffx, subdir="2D/")
    plot_town_i(result, i2, name="Bergen"+suffx, subdir="2D/")


# testSIR()
# conv_det()
# flatten()
# vax()

# test_stoch()
# conv_stoch()
# disappear()

# testSEIIaR()
# conv_SEIIaR()
test_isolation()

# testSEIIaR_commute()
# conv_SEIIaR_commute(False)
# two_towns2()
# two_towns()
## Named nine towns in honour of the fact that I can't count
# nine_towns()

# pop_struct()
# pop_struct_lockdown()
# poplutaion()
# num_infected()
# num_infected(lockdown=True)

