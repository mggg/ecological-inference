"""
Port of R code for Noncentral Hypergeometric distribution
 from Martin AD, Quinn KM, Park JH (2011). “MCMCpack: 
 Markov Chain Monte Carlo in R.” Journal of Statistical 
 Software, 42(9), 22. doi: 10.18637/jss.v042.i09.

Used in Greiner-Quinn method Gibbs sampler
"""
import math

# The function r defined in Liao and Rosen 2001
def r_function(n1, n2, m1, psi, i):
    return (n1-i+1)*(m1-i+1) / (i*(n2-m1+i)) * psi


def sample_low_to_high(lower, ran, pi, shift, uu):
      for i in range(lower, uu + 1):
        if(ran <= pi[i+shift]):
            return(i)
        ran = ran - pi[i+shift]


def sample_high_to_low(upper, ran, pi, shift, ll): 
      for i in range(upper, ll - 1, -1):
        if(ran <= pi[i+shift]):
            return(i)
        ran = ran - pi[i+shift]

        
class NonCentralHyperGeometric:
    """Allows for sampling from noncentralhypergeometric distribution
    Following the methods of Liao and Rosen, 2001
    
    If
    y1 ~ Binom(n1, pi1)
    y2 ~ Binom(n2, pi2)
    psi = pi1 (1-pi2) / pi2 (1-pi1)
    
    this the nchg distribution governs with parameters n1, n2, pis, m1
    is the distribution for y1 | y1 + y2 = m1
    """
    
    def __init__(self, n1, n2, m1, psi):
        """
        n1 : int
            Num trials for the first binomial dist y1~binom(n1,pi1)
        n2 : int
            Num trials for the second binomial dist y2~binom(n2,pi12)
        m1: int
            NCHG gives distribution for y1 | y1 + y2 = m1
        psi : float
            (pi1 * (1-pi2)) / (pi2 * (1-p1)), where pi1, pi2 are the probs
            of success in the two binomial distributions
        """
        self.n1 = n1
        self.n2 = n2
        self.m1 = m1
        self.psi = psi
        
        self.ll = max(0, m1-n2)
        self.uu = min(n1, m1)
        
        self._mode = None
        self._density = None #vector of p_i in Liao and Rosen notation
    
    
    @property
    def mode(self):
        # calculate mode
        if self._mode is None:
            a = self.psi - 1
            b = -( (self.m1 + self.n1 + 2)* self.psi + self.n2 - self.m1 )
            c = self.psi * (self.n1 + 1) * (self.m1 + 1)
            q = -( b + np.sign(b) * np.sqrt(b*b-4*a*c) )/2
            mode = math.trunc(c / q)
            if (self.uu >= mode) and (self.ll <= mode):
                self._mode = mode
            else:
                self._mode = math.trunc(q / a) 
        return self._mode
    
    #def get_samples(self, num_samples=1):
        
     
    @property
    def density(self):
        if self._density is None:
            pi = np.ones(self.uu - self.ll + 1)

            if (self.mode < self.uu):
                r = r_function(self.n1, self.n2, self.m1, self.psi, np.arange(self.mode + 1, self.uu + 1))
                pi[(self.mode + 1 - self.ll):(self.uu - self.ll + 1)] = np.cumprod(r)

            if (self.mode > self.ll):
                r = 1/ r_function(self.n1, self.n2, self.m1, self.psi, np.flip(np.arange(self.ll + 1, self.mode + 1)))
                pi[0: (self.mode - self.ll)] = np.flip(np.cumprod(r))
            self._density = pi / sum(pi)
            
        return self._density
    
    def get_sample(self):
        ran = np.random.uniform()   
        pi = self.density
        
        if self.mode == self.ll:
            return sample_low_to_high(self.ll, ran, pi, -self.ll, self.uu)
        if self.mode == self.uu:
            return sample_high_to_low(self.uu, ran, pi, -self.ll, self.ll) 
        if(ran < pi[self.mode - self.ll]):
            return self.mode
        ran = ran - pi[self.mode - self.ll]
        lower = self.mode - 1
        upper = self.mode + 1

        while True:
            if(pi[upper - self.ll] >= pi[lower - self.ll]):
                if(ran < pi[upper - self.ll]):
                    return(upper)
                ran = ran - pi[upper - self.ll]
                if(upper == self.uu):
                    samp = sample_high_to_low(lower, ran, pi, -self.ll, self.ll)
                    return samp
                upper = upper + 1
            
            else:
                if(ran < pi[lower - self.ll]):
                    return lower
                ran = ran - pi[lower - self.ll]
                if(lower == self.ll):
                    samp = sample_low_to_high(upper, ran, pi, -self.ll, self.uu)
                    return samp
                lower = lower - 1
            
