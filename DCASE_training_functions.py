import keras
from keras import backend as K
import numpy as np
import threading, random
import tensorflow as tf

class ckpt(keras.callbacks.Callback):
    
    def __init__(self,filepath,ckpts):
        self.ckpts = ckpts
        self.filepath = filepath
 
    def on_epoch_end(self, epoch, logs={}):
        if epoch+1 in self.ckpts:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            self.model.save(filepath, overwrite=True)
            print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
        return

class LR_WarmRestart(keras.callbacks.Callback):
    
    def __init__(self,nbatch,initial_lr,min_lr,epochs_restart,Tmult,T0):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.epochs_restart = epochs_restart
        self.nbatch = nbatch
        self.currentEP=0
        self.startEP=0
        self.Tmult=Tmult
        self.T0 =T0
        
    def on_epoch_begin(self, epoch, logs={}):
        if epoch+1<self.epochs_restart[0]:
            self.currentEP = epoch
        else:
            self.currentEP = epoch+1
            
        if np.isin(self.currentEP,self.epochs_restart):
            self.startEP=self.currentEP
            self.T0=self.T0*self.Tmult
        
    def on_epoch_end(self, epochs, logs={}):
        lr = K.get_value(self.model.optimizer.lr)
        print ('\nLearningRate:{:.6f}'.format(lr))
    
    def on_batch_begin(self, batch, logs={}):
        pts = self.currentEP + (batch+1)/(self.nbatch+1) - self.startEP
        decay = 1+np.cos(pts/self.T0*np.pi)
        lr = self.min_lr+0.5*(self.initial_lr-self.min_lr)*decay
        K.set_value(self.model.optimizer.lr,lr)

        
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    # python 3
#     def __next__(self):
#         with self.lock:
#             return self.it.__next__()
        
    # python 2
    def next(self):
        with self.lock:
            return self.it.next()
        
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class MixupGenerator():
    '''
    Reference: https://github.com/yu4u/mixup-generator
    '''
    
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, crop_length=400, y_train_2=None, datagen=None): #datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_2 = y_train_2
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.datagen = datagen
        self.sample_num = len(X_train)
        self.lock = threading.Lock()
        self.NewLength = crop_length
        self.swap_inds = [1,0,3,2,5,4]
        
    def __iter__(self):
        return self
    
    @threadsafe_generator
    def __call__(self):
        with self.lock:
            while True:
                indexes = self.__get_exploration_order()
                itr_num = int(len(indexes) // (self.batch_size * 2))

                for i in range(itr_num):
                    batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                   # X, y = self.__data_generation(batch_ids)

                    #yield X, y
                    yield self.__data_generation(batch_ids)

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        
        if self.NewLength > 0:
            for j in range(X1.shape[0]):
                StartLoc1 = np.random.randint(0,X1.shape[2]-self.NewLength)
                StartLoc2 = np.random.randint(0,X2.shape[2]-self.NewLength)

                X1[j,:,0:self.NewLength,:] = X1[j,:,StartLoc1:StartLoc1+self.NewLength,:]
                X2[j,:,0:self.NewLength,:] = X2[j,:,StartLoc2:StartLoc2+self.NewLength,:]

                if X1.shape[-1]==6:
                    #randomly swap left and right channels 
                    if np.random.randint(2) == 1:
                        X1[j,:,:,:] = X1[j:j+1,:,:,self.swap_inds]
                    if np.random.randint(2) == 1:
                        X2[j,:,:,:] = X2[j:j+1,:,:,self.swap_inds]


            X1 = X1[:,:,0:self.NewLength,:]
            X2 = X2[:,:,0:self.NewLength,:]
        
        X = X1 * X_l + X2 * (1.0 - X_l)
        
        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1.0 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1.0 - y_l)
            

        if self.y_train_2 is None:
            return X, y
        else:
            return X,[y, self.y_train_2[batch_ids[:self.batch_size]]]

        
def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Reference: https://github.com/umbertogriffo/focal-loss-keras
    
    Softmax version of focal loss.
            m
      FL = SUM -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
           c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(loss, axis=1)

    return categorical_focal_loss_fixed