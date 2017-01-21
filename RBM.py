# ----------------------------------------------------
# Training a Centered Deep Boltzmann Machine
# ----------------------------------------------------
#
# Copyright: Gregoire Montavon
#
# This code is released under the MIT licence:
# http://www.opensource.org/licenses/mit-license.html
#
# ----------------------------------------------------
#
# This code is based on the paper:
#
#   G. Montavon, K.-R. Mueller
#   Deep Boltzmann Machines and the Centering Trick
#   in Neural Networks: Tricks of the Trade, 2nd Edn
#   Springer LNCS, 2012
#
# ----------------------------------------------------
#
# This code is a basic implementation of the centered
# deep Boltzmann machines (without model averaging,
# minibatching and other optimization hacks). The code
# also requires the MNIST dataset that can be
# downloaded at http://yann.lecun.com/exdb/mnist/.
#
# ----------------------------------------------------

import numpy
from PIL import Image
DIR = '/Users/blythed/work/Data/thirdparty/classification/MNIST/'

# ====================================================
# Global parameters
# ====================================================
lr      = 0.005     # learning rate
rr      = 0.005     # reparameterization rate
mb      = 25        # minibatch size
hlayers = [400,100] # size of hidden layers
biases  = [-1,-1]   # initial biases on hidden layers

# ====================================================
# Helper functions
# ====================================================
def arcsigm(x): return numpy.arctanh(2*x-1)*2
def sigm(x):    return (numpy.tanh(x/2)+1)/2
def realize(x): return (x > numpy.random.uniform(0,1,x.shape))*1.0
def render(x,name):
    x = x - x.min() + 1e-9
    x = x / (x.max() + 1e-9)
    Image.fromarray((x*255).astype('byte'),'L').save(name)

# ====================================================
# Centered deep Boltzmann machine
# ----------------------------------------------------
# - self.W: list of weight matrices between layers
# - self.B: list of bias associated to each unit
# - self.O: list of offsets associated to each unit
# - self.X: free particles tracking model statistics
# ====================================================
class DBM:
    # --------------------------------------------
    # Initialize model parameters and particles
    # --------------------------------------------
    def __init__(self,M,B):
        self.W = [numpy.zeros([m,n]).astype('float32') for m,n in zip(M[:-1],M[1:])]
        self.B = [numpy.zeros([m]).astype('float32')+b for m,b in zip(M,B)]
        self.O = [sigm(b) for b in self.B]
        self.X = [numpy.zeros([mb,m]).astype('float32')+o for m,o in zip(M,self.O)]

    # --------------------------------------------
    # Gibbs activation of a layer
    # --------------------------------------------
    def gibbs(self,X,l):
        bu = numpy.dot(X[l-1]-self.O[l-1],self.W[l-1]) if l   > 0      else 0
        td = numpy.dot(X[l+1]-self.O[l+1],self.W[l].T) if l+1 < len(X) else 0
        X[l] = realize(sigm(bu+td+self.B[l]))

    # --------------------------------------------
    # Reparameterization
    # --------------------------------------------
    def reparamB(self,X,i):
        bu = numpy.dot((X[i-1]-self.O[i-1]),self.W[i-1]).mean(axis=0) if i   > 0      else 0
        td = numpy.dot((X[i+1]-self.O[i+1]),self.W[i].T).mean(axis=0) if i+1 < len(X) else 0
        self.B[i] = (1-rr)*self.B[i] + rr*(self.B[i] + bu + td)

    def reparamO(self,X,i):
        self.O[i] = (1-rr)*self.O[i] + rr*X[i].mean(axis=0)

    # --------------------------------------------
    # Learning step
    # --------------------------------------------
    def learn(self,Xd):

        # Initialize a data particle
        X = [realize(Xd)]+[self.X[l]*0+self.O[l] for l in range(1,len(self.X))]
        
        # Alternate gibbs sampler on data and free particles
        for l in (range(1,len(self.X),2)+range(2,len(self.X),2))*5: self.gibbs(X,l)
        for l in (range(1,len(self.X),2)+range(0,len(self.X),2))*1: self.gibbs(self.X,l)
        
        # Parameter update
        for i in range(0,len(self.W)):
            self.W[i] += lr*(numpy.dot((     X[i]-self.O[i]).T,     X[i+1]-self.O[i+1]) -
                             numpy.dot((self.X[i]-self.O[i]).T,self.X[i+1]-self.O[i+1]))/len(Xd)
        for i in range(0,len(self.B)):
            self.B[i] += lr*(X[i]-self.X[i]).mean(axis=0)
        
        # Reparameterization
        for l in range(0,len(self.B)): self.reparamB(X,l)
        for l in range(0,len(self.O)): self.reparamO(X,l)

# ====================================================
# Example of execution
# ====================================================

# Initialize MNIST dataset and centered DBM
X = (numpy.fromfile(open(DIR + 'train-images-idx3-ubyte','r'),dtype='ubyte',count=16+784*60000)[16:].reshape([60000,784])).astype('float32')/255.0
nn = DBM([784]+hlayers,[arcsigm(numpy.clip(X.mean(axis=0),0.01,0.99))]+biases)

for it in range(1000):

    # Perform some learning steps
    for _ in range(100): nn.learn(X[numpy.random.permutation(len(X))[:mb]])
    
    # Output some debugging information
    print(("%03d |" + " %.3f "*len(nn.W))%tuple([it]+[W.std() for W in nn.W]))
    W = 1
    for l in range(len(nn.W)):
        W = numpy.dot(W,nn.W[l])
        m = int(W.shape[1]**.5)
        render(W.reshape([28,28,m,m]).transpose([2,0,3,1]).reshape([28*m,28*m]),'W%d.jpg'%(l+1));
    render((nn.X[0]).reshape([mb,28,28]).transpose([1,0,2]).reshape([28,mb*28]),'X.jpg');
