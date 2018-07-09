import numpy as np
from noisi.my_classes.basisfunctions import choose_basis_function

class BasisFunction(object):
    """
    Class to take care of basis function properties
    """

    def __init__(self,basis_type,K,N=None):
        """
        :type basis_type: string
        :param basis_type: choice of basis function: sine_taper
        :type K: int
        :param n: number of basis vectors to use
        """

        self.basis_type = basis_type
        self.K = int(K)
        self.basis_func = choose_basis_function(self.basis_type)
        self.basis = None
        self.N = N
        
        if type(self.N) == int:
            basis = np.array([
                self.basis_vector(k,self.N) for k in range(self.K)])
            self.basis = np.ascontiguousarray(np.transpose(basis)
                ,dtype=np.float32)


    def coeff(self,vector):
        """
        :type vector: numpy array
        :param vector: Some vector to expand in the basis
        """
        N = len(vector)

        if self.basis is not None:
            C = np.dot(np.transpose(self.basis),vector)

        else:
            C = []
            for k in range(self.K):
                C.append(np.dot(self.basis_vector(k,N),vector))
            C = np.array(C)
        return(C)

    def expand(self,C,N=None):

        if N is not None:
            if self.N is not None and self.N != N:
                raise ValueError("N set twice.")


        if self.basis is not None:
            new_vector = np.dot(self.basis,C)

        else:
            new_vector = np.zeros(N)
            for k in range(self.K):
                new_vector += self.basis_vector(k,N) * C[k]

        return(new_vector)


    def project(self,vector,N=None):

        
        C = self.coeff(vector)
        
        new_vector = self.expand(C,N=len(vector))

        return(new_vector)

    def basis_vector(self,k,N):
        """
        :type k: int
        :param k: returns the k'th basis vector
        :type N: int
        :param N: number of samples (basis vector length)
        """
        if k > self.K:
            raise ValueError('Basis has only {} dimensions.'.format(self.K))
        return(self.basis_func(k,N))


