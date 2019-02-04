import matplotlib.pyplot

matplotlib.pyplot.plotfile('err.csv', ('iteration','err','validation_err'),subplots=False, 
                              plotfuncs={'err': 'semilogy','validation_err':'semilogy'})
matplotlib.pyplot.grid(axis='both')
matplotlib.pyplot.title('Error vs iteration')
matplotlib.pyplot.show()
