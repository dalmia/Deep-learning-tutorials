import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from time import time
from foxhound import activations
from foxhound import updates
from foxhound import inits
from foxhound.theano_utils import floatX, sharedX
import theano
import theano.tensor as T
from scipy.stats import gaussian_kde
from scipy.misc import imsave, imread

# Define hyperparameters
leakyrectify = activations.LeakyRectify()
rectify = activations.Rectify()
tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy
batch_size = 128
nh = 2048 # Number of hidden layers
init_fn = inits.Normal(scale=0.02)

# Visualize Gaussian curves
def gaussian_likelihood(X, u=0, s=1):
    return (1./(s * np.sqrt(2 * np.pi))) * np.exp(-((X - u) ** 2) / (2 * (s ** 2)))

def scale_and_shift(X, g, b, e=1e-8):
    X = X * g + b
    return X

# Build generative and discriminative Networks
def g(X, w, g, b, w2, g2, b2, wo):
    h = leakyrectify(scale_and_shift(T.dot(X, w), g, b))
    h2 = leakyrectify(scale_and_shift(T.dot(h, w2), g2, b2))
    y = T.dot(h2, wo)
    return y

def d(X, w, g, b, w2, g2, b2, wo):
    h = rectify(scale_and_shift(T.dot(X, w), g, b))
    h2 = tanh(scale_and_shift(T.dot(h, w2), g2, b2))
    y = sigmoid(T.dot(h2, wo))
    return y

# Initialize both networks
gw = init_fn((1, nh))
gg = inits.Normal(1., 0.02)(nh)
gb = inits.Normal(0., 0.02)(nh)
gw2 = init_fn((nh, nh))
gg2 = inits.Normal(1., 0.02)(nh)
gb2 = inits.Normal(0., 0.02)(nh)
gy = init_fn((nh, 1))
dw = init_fn((1, nh))
dg = inits.Normal(1., 0.02)(nh)
db = inits.Normal(0., 0.02)(nh)
dw2 = init_fn((nh, nh))
dg2 = inits.Normal(1., 0.02)(nh)
db2 = inits.Normal(0., 0.02)(nh)
dy = init_fn((nh, 1))

# set the parameters
g_params = [gw, gg, gb, gw2, gg2, gb2, gy]
d_params = [dw, dg, db, dw2, dg2, db2, dy]

# get the matrix response
Z = T.matrix()
X = T.matrix()

# build the generator from the transpose
gen = g(Z, *g_params)

# get both the real and generated samples
p_real = d(X, *d_params)
p_gen = d(gen, *d_params)

# get the cost function from both real, generated and discriminator functions
d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

d_cost = d_cost_gen + d_cost_real
g_cost = g_cost_d

cost = [g_cost, d_cost, d_cost_real, d_cost_gen]

# update the weight using the Adam method
lr = 0.001
lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt)
g_updater = updates.Adam(lr=lrt)

d_updates = d_updater(d_params, d_cost)
g_updates = g_updater(g_params, g_cost)

updates = d_updates + g_updates

# get the final variables for both networks, cost and score
_train_g = theano.function([X, Z], cost, updates=g_updates)
_train_d = theano.function([X, Z], cost, updates=d_updates)
_train_both = theano.function([X, Z], cost, updates=updates)
_gen = theano.function([Z], gen)
_score = theano.function([X], p_real)
_cost = theano.function([X, Z], cost)

# create the graph figure
fig = plt.figure()

# plotting function - plot both curves (for G and D)
def vis(i):
    s = 1
    u = 0
    zs = np.linspace(-1, 1, 500).astype('float32')
    xs= np.linspace(-5, 5, 500).astype('float32')
    ps = gaussian_likelihood(xs, 1.)

    gs = _gen(zs.reshape(-1, 1)).flatten()
    p_real = _score(xs.reshape(-1, 1)).flatten()
    kde = gaussian_kde(gs)

    plt.clf()
    plt.plot(xs, ps, '--', lw=2)
    plt.plot(xs, kde(xs), lw=2)
    plt.plot(xs, p_real, lw=2)
    plt.xlim([-5, 5])
    plt.ylim([0., 1.])
    plt.ylabel('Prob')
    plt.xlabel('x')
    plt.legend(['P(data)','G(z)', 'D(x)'])
    plt.title('GAN learning gaussian')
    fig.canvas.draw()
    plt.show(block=False)
    show()

# Train both networks
for i in range(100):
    # get the uniform distribution of both networks
    zmb = np.random.uniform(-1, 1, size=(batch_size, 1)).astype('float32')
    xmb = np.random.normal(1., 1, size=(batch_size, 1)).astype('float32')
    if i % 10 == 0:
        _train_g(xmb, zmb)
    else:
        _train_d(xmb, zmb)

    if i % 10 == 0:
        print(i)
        vis(i)
    lrt.set_value(floatX(lrt.get_value() * 0.9999))
