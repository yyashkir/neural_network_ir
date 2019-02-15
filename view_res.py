import matplotlib.pyplot

matplotlib.pyplot.plotfile('err.csv', ('iteration','err','validation_err'),linestyle="-",marker="",subplots=False, 
                              plotfuncs={'err': 'loglog','validation_err':'loglog'}
							  )
matplotlib.pyplot.grid(which="both")
matplotlib.pyplot.title('Errors vs iteration')
matplotlib.pyplot.savefig('err.png')
matplotlib.pyplot.show()

matplotlib.pyplot.plotfile('a2y.csv', ('y','y','a'),linestyle="",marker=".",subplots=False)
matplotlib.pyplot.grid(which="both")
matplotlib.pyplot.title('yield: modelled (a) vs historical (y)')
matplotlib.pyplot.savefig('a2y.png')
matplotlib.pyplot.show()

matplotlib.pyplot.plotfile('a2y.csv', ('t','y','a'),linestyle="",marker=".",subplots=False)
matplotlib.pyplot.grid(which="both")
matplotlib.pyplot.title('Yields: modelled (a) and historical (y)')
matplotlib.pyplot.savefig('a2yt.png')
matplotlib.pyplot.show()

