import sys
from settings import DCGAN_ROOT

sys.path.append(DCGAN_ROOT)

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.theano_utils import floatX, sharedX, intX

from sklearn.externals import joblib
import glob

sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()


model_path = DCGAN_ROOT + '/models/imagenet_gan_pretrain_128f_relu_lrelu_7l_3x3_256z/'
discrim_params = [sharedX(p) for p in joblib.load(model_path + '30_discrim_params.jl')]


def discrim(X, w, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, w6, g6, b6, wy):
    h = lrelu(dnn_conv(X, w, subsample=(1, 1), border_mode=(1, 1)))
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(1, 1)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(1, 1), border_mode=(1, 1)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(2, 2), border_mode=(1, 1)), g=g4, b=b4))
    h5 = lrelu(batchnorm(dnn_conv(h4, w5, subsample=(1, 1), border_mode=(1, 1)), g=g5, b=b5))
    h6 = lrelu(batchnorm(dnn_conv(h5, w6, subsample=(2, 2), border_mode=(1, 1)), g=g6, b=b6))
    h6 = T.flatten(h6, 2)
    y = sigmoid(T.dot(h6, wy))
    return y


def inverse_transform(X):
    X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1) + 1.) / 2.
    return X


X = T.tensor4()

dX = discrim(X, *discrim_params)
_discrim = theano.function([X], dX)


def getScore(img_paths):
    from scipy.misc import imread, imresize
    imgs = [imresize(imread(i), (32, 32)).T for i in img_paths]
    samples = np.array(imgs)
    return _discrim(samples)


def folderScore(folder):
    img_paths = glob.glob(folder + "/*.jpg")
    return getScore(img_paths)


def average_folder(folder_list):
    print "Starting...."
    for f in folder_list:
        print "Folder %s"%f
        scores = folderScore(f)
        print scores
        print "Score mean :%f" % (np.mean(scores))
        print "---"

def main():
    return average_folder(sys.argv[1:])


if __name__ == '__main__':
    main()
