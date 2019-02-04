import matplotlib.pyplot

matplotlib.pyplot.plotfile('err.csv', ('iteration','log_err','log_validation_err'))
matplotlib.pyplot.show()
