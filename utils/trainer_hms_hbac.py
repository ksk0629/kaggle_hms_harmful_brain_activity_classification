# %% [code]
import math

import keras_cv
import keras
from keras import ops
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config_hms_hbac import Config


class Trainer():
    def __init__(self, config):
        self.loss = keras.losses.KLDivergence()
        self.config = config
        
        keras.utils.set_random_seed(self.config.seed)
    
    def compile_model(self):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=self.loss)
    
    def set_checkpoint(self, path: str):
        self.checkpoint = keras.callbacks.ModelCheckpoint(path,
                                                          monitor='val_loss',
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='min')
    
    def set_lr_callback(self, plot=False):
        lr_start, lr_max, lr_min = 5e-5, 6e-6 * self.config.batch_size, 1e-5
        lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

        def lrfn():  # Learning rate update function
            if self.config.epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            elif self.config.epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
            elif self.config.lr_mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(self.config.epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            elif self.config.lr_mode == 'step': lr = lr_max * lr_decay**((self.config.epoch - lr_ramp_ep - lr_sus_ep) // 2)
            elif self.config.lr_mode == 'cos':
                decay_total_epochs, decay_epoch_index = self.config.epochs - lr_ramp_ep - lr_sus_ep + 3, self.config.epoch - lr_ramp_ep - lr_sus_ep
                phase = math.pi * decay_epoch_index / decay_total_epochs
                lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
            return lr

        if plot:  # Plot lr curve if plot is True
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(self.config.epochs), [lrfn(epoch) for epoch in np.arange(self.config.epochs)], marker='o')
            plt.xlabel('epoch'); plt.ylabel('lr')
            plt.title('LR Scheduler')
            plt.show()

        self.lr_callback = keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback
    
    def set_model(self):
        self.model = keras_cv.models.ImageClassifier.from_preset(
            self.config.pretrained_model, num_classes=self.config.num_classes
        )
        
    def show_model_summary(self):
        self.model.summary()
    
    def train(self, train_ds, valid_ds):
        self.history = self.model.fit(
            train_ds, 
            epochs=self.config.epochs,
            callbacks=[self.lr_callback, self.checkpoint], 
            steps_per_epoch=len(train_df)//self.config.batch_size,
            validation_data=valid_ds, 
            verbose=self.config.verbose
        )