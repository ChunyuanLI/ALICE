from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
import numpy as np
import tensorflow as tf
import scipy.ndimage.interpolation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from celeb_model import encoder1, encoder2, discriminator
# from model_fancyCelebA_utils import encoder1, encoder2, discriminator
import scipy.io as sio
import pdb
import h5py
import json
import time

""" parameters """
n_epochs = 16
mb_size = 64
# X_dim = ()
lr = 1e-4
Z_dim = 64
pa, pb = 0., 0.
lamb_list = [1e-6, 1e-4, 1e-2, 1e-1, 0.5, 50, 100, 1000]

#####################################

def log(x):
    return tf.log(x + 1e-8)

""" data pre-process """
hdf5_root = '/home/lqchen/work/pixel-cnn-3/data/CelebA/'
Images = h5py.File('%sceleba_64.hdf5' % hdf5_root)['features']

# Images_all = np.transpose(Images, [0, 2, 3, 1])
# Images = Images_all[:162770]
# Images_val = Images_all[162770: 182637]
# Images_test = Images_all[182637:]
# del Images_all

num_train = 192000

""" function utilities """
def sample_XY(X, Y, size):
    start_idx = np.random.randint(0, X.shape[0] - size)
    return X[start_idx:start_idx + size], Y[start_idx:start_idx + size]


def sample_X(X, size, num_train=192000):
    start_idx = np.random.randint(0, num_train - size)
    return X[start_idx:start_idx + size]


def sample_Y(Y, size):
    start_idx = np.random.randint(0, Y.shape[0] - size)
    return Y[start_idx:start_idx + size]

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

""" Networks """
def generative_Y2X(z, reuse=None):
    with tf.variable_scope("Y2X", reuse=reuse):
        h = encoder2(z)
    return h
def generative_X2Y(x, reuse=None):
    with tf.variable_scope("X2Y", reuse=reuse):
        h = encoder1(x)
    return h

def data_network(x, y, reuse=None):
    with tf.variable_scope('D', reuse=reuse):
        f, d = discriminator(x, y)
    return tf.squeeze(f, squeeze_dims=[1]), tf.squeeze(d, squeeze_dims=[1])

# def data_network_2(x, y, reuse=None):
#     """Approximate z log data density."""
#     with tf.variable_scope('D2', reuse=reuse) as scope:
#         d = discriminator(x, y)
#
#     return tf.squeeze(d, squeeze_dims=[1])

""" Construct model and training ops """
tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[mb_size, 64, 64, 3])
z = tf.placeholder(tf.float32, shape=[mb_size, Z_dim])
lamb = tf.placeholder(tf.float32)
# Generator
z_gen = generative_X2Y(X)
X_gen = generative_Y2X(z)
# Discriminator
fxz, Dxz = data_network(X, z_gen)
fzx, Dzx = data_network(X_gen, z, reuse=True)

# Discriminator loss

D_loss = -tf.reduce_mean(log(Dzx) + log(1 - Dxz))

# Generator loss
L_x = -tf.reduce_mean(Dxz)
L_z = tf.reduce_mean(Dzx)
G_loss = L_x + L_z

## reconstruct
X_rec = generative_Y2X(z_gen, reuse=True)
z_rec = generative_X2Y(X_gen, reuse=True)
Xrec_loss = tf.reduce_mean(tf.abs(X - X_rec))
Zrec_loss = tf.reduce_mean(tf.abs(z - z_rec))

""" Solvers """
gvar1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Y2X")
gvar2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "X2Y")
dvars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "D")

opt = tf.train.AdamOptimizer(lr, beta1=0.5)

# G_loss = G_loss + pa * Xrec_loss + pb * Zrec_loss
G_loss = G_loss + lamb * Xrec_loss + lamb * Zrec_loss
# G_loss = G_loss + lamb * X_ce_loss + lamb * Z_ce_loss
D_solver = opt.minimize(D_loss, var_list = dvars1)
G_solver = opt.minimize(G_loss, var_list = gvar1 + gvar2)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

""" Training """
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)

# Load pretrained Model
# try:
#     saver.restore(sess=sess, save_path="../model/celeba_model_R.ckpt")
#     print("\n--------model restored--------\n")
# except:
#     print("\n--------model Not restored--------\n")
#     pass

disc_steps = 2
gen_steps = 1
for num_test in range(len(lamb_list)):
    init = tf.global_variables_initializer()
    sess.run(init)
    for it in range(n_epochs):

        # TODO: dynamic control of the steps
        # if it >= 4:
        #     disc_steps = 1
        #     gen_steps = 1
        for idx in range(0, num_train // mb_size):

            # _x = Images[idx * mb_size: (idx + 1) * mb_size]
            _x = sample_X(Images, mb_size)
            _x = np.transpose(_x, [0,2,3,1]) / 127.5 - 1
            z_sample = sample_Z(mb_size, Z_dim)
            for k in range(disc_steps):
                _, D_loss_curr = sess.run([D_solver, D_loss],
                                          feed_dict={X: _x, z: z_sample, lamb: lamb_list[num_test]})
            for j in range(gen_steps):
                _, G_loss_curr = sess.run([G_solver, G_loss],
                                          feed_dict={X: _x, z: z_sample, lamb: lamb_list[num_test]})

            if idx % 200 == 0:
                saver.save(sess, './model/celeba_model_ali_%d.ckpt' % num_test)
                print(num_test)
                print('epoch: {}; iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(
                    it, idx, D_loss_curr, G_loss_curr))
