def plot_predictions(a,yd,xl=r"$\beta$",yl=r"$\rho$"):
    from matplotlib.pyplot import subplots,plot,xlabel,ylabel,subplots_adjust
    x=a[0].transpose()[0]
    y=a[1].transpose()[yd]

    plot(x,y,'-',alpha=1)
    subplots_adjust(left=0.18,bottom=0.18)

    xlabel(xl)
    ylabel(yl)
