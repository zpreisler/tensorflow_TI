class landscape:
    """
    Generate rugged landscape
    """
    def __init__(self,n=64,random_seed=1313):
        """
        n: number of gaussians
        """
        from numpy.random import rand,seed
        from numpy import sqrt
        seed(random_seed)
        self.n=n
        v=rand(3,n)
        v[0]=v[0]*50-40
        v[1]=v[1]*5-2
        v[2]=v[2]*2.0*sqrt(10.0)
        self.v=v.transpose()

    def y(self,x,s=0.0):  
        from numpy.random import uniform,normal
        y=0
        for h,mu,sigma in self.v:
            y+=h*self.gaussian(x,mu,sigma)
        noise=normal(0.0,s,len(y))
        return y/self.n+noise

    def gaussian(self,x,mu,sigma):
        from numpy import exp,sqrt,pi
        return 1.0/sqrt(2*pi*sigma**2)*exp(-(x-mu)**2/(2.0*sigma**2))


