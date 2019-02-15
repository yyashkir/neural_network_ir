import matplotlib.pyplot
matplotlib.pyplot.plotfile('a2y.csv', ('y','y','a'),linestyle="",marker=".",subplots=False)

matplotlib.pyplot.grid(which="both")
matplotlib.pyplot.title('yield: modelled vs hist')
matplotlib.pyplot.savefig('a2y.png')
matplotlib.pyplot.show()

matplotlib.pyplot.plotfile('a2y.csv', ('t','y','a'),linestyle="",marker=".",subplots=False)

matplotlib.pyplot.grid(which="both")
matplotlib.pyplot.title('Yields: modelled and hist')
matplotlib.pyplot.savefig('a2yt.png')
matplotlib.pyplot.show()