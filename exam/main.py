from plots import plotSIR
from deterministic_SIR import get_testSIR
from plots import plotSIR



def testSIR():
    plotSIR(*get_testSIR())

if __name__=="__main__":
    testSIR()
