import numpy as np

class BasisFunction():
    """
    Class to take care of basis function properties
    """
    @staticmethod
    def initialize(basis_type,K,N=None,**kwargs):
        """
        :type basis_type: string
        :param basis_type: choice of basis function: sine_taper
        :type K: int
        :param K: number of basis vectors to use
        """
        if basis_type == "sine_taper":
            return SineBasis(K,N)
        elif basis_type == "sinc_taper":
            return SincBasis(K,N,**kwargs)
        else:
            raise ValueError("Unknown type of basis function: "+basis_type)

    def basis_vector(self,k,N,**kwargs):
        """
        :type k: int
        :param k: returns the k'th basis vector
        :type N: int
        :param N: number of samples (basis vector length)
        """
        if k > self.K:
            raise ValueError('Basis has only {} dimensions.'.format(self.K))

        return(self.basis_func(k+1,N,**kwargs))

    def coeff(self,vector,**kwargs):
        """
        :type vector: numpy array
        :param vector: Some vector to expand in the basis
        """
        N = len(vector)

        if self.basis is not None:
            C = np.dot(self.basis,vector)

        else:
            C = []
            for k in range(self.K):
                v = self.basis_vector(k,N,**kwargs)
                C.append(np.dot(v,vector))
            C = np.array(C)
        return(C)

    def expand(self,C,N=None,**kwargs):

        if N is not None:
            if self.N is not None and self.N != N:
                raise ValueError("N set twice.")


        if self.basis is not None:
            new_vector = np.dot(C,self.basis)

        else:
            new_vector = np.zeros(N)
            for k in range(self.K):
                new_vector += self.basis_vector(k,N,**kwargs) * C[k] 

        return(new_vector)


    def project(self,vector,N=None,**kwargs):

        
        C = self.coeff(vector,**kwargs)
        
        new_vector = self.expand(C,N=len(vector))

        return(new_vector)




class SineBasis(BasisFunction):

    def __init__(self,K,N):

        self.K = K
        self.N = N
        self.basis_type='sine_taper'
        self.basis = None

        if type(N) == int:
            basis = np.array([
                self.basis_vector(k,self.N) for k in range(self.K)])
            self.basis = np.ascontiguousarray(basis,dtype=np.float32)
        

    def basis_func(self,k,n):
        """
        Return the sine taper (Riedel & Sidorenko, IEEE'95)
        :type k: int
        :param k: return the k'th taper
        :type N: int
        :param N: Number of samples
        """

        x = np.linspace(0,n+1,n) # make sure it goes to 0

        
        norm = np.sqrt(2.)/np.sqrt(float(n-1))
        y = norm * np.sin(np.pi*k*x/(n+1))
        y[0] = 0.0
        y[-1] = 0.0
       
        return(y)


class SincBasis(BasisFunction):


    def __init__(self,K,N,**kwargs):

        self.K = K
        self.N = N
        self.basis_type='sinc_taper'
        self.basis = None
        self.freq = kwargs['freq']
        self.fmin = kwargs['fmin']
        self.fmax = kwargs['fmax']
        

        if type(N) == int:
            basis = np.array([
                self.basis_vector(k,self.N) for k in range(self.K)])
            self.basis = np.ascontiguousarray(basis,dtype=np.float32)
        

    def basis_vector(self,k,n):

        """
        Return the sinc function (reference?)
        :type k: int
        :param k: return the k'th taper
        :type N: int
        :param N: Number of samples
        """

        try: 
            ix_fmin = np.argmin(np.abs(self.freq-self.fmin))
            ix_fmax = np.argmin(np.abs(self.freq-self.fmax))
            n_supp = ix_fmax-ix_fmin
        except:
            ix_fmin = 0
            n_supp = n

        #print("Sinc integer step each %g samples" %width)

        x = np.linspace(0.0,self.K*np.pi*(n-n_supp)/n_supp,n)
        x -= self.K*np.pi*ix_fmin/n_supp
        
        argu = (x/np.pi-k) # non-normalized sinc: divide argument by pi
        norm =  np.sqrt((x[-1]-x[0])/(n*np.pi))
        y = norm * np.sinc(argu) # normalize
       


        return(y)

# class BasisFunction(object):
#     """
#     Class to take care of basis function properties
#     """

#     def __init__(self,basis_type,K,N=None,**kwargs):

         
#         self.basis_type = basis_type
#         self.K = int(K)
#         self.basis_func = choose_basis_function(self.basis_type)
#         self.basis = None
#         self.N = N
        
        
        
#         if type(self.N) == int:
#             basis = np.array([
#                 self.basis_vector(k,self.N,**kwargs) for k in range(self.K)])
#             self.basis = np.ascontiguousarray(basis,dtype=np.float32)




