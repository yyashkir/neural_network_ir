import matplotlib.pyplot

matplotlib.pyplot.plotfile('err.csv', ('iteration','err','validation_err'),linestyle="",marker=".",subplots=False, 
                              plotfuncs={'err': 'semilogy','validation_err':'semilogy'})
matplotlib.pyplot.grid(which="both")
matplotlib.pyplot.title('Error vs iteration')
matplotlib.pyplot.savefig('err.png')
matplotlib.pyplot.show()

