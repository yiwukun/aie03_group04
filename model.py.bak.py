import numpy as np
import keras ## broken for keras >= 2.0, use 1.2.2
from keras.models import Sequential,Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers.core import Flatten, Dense, Activation, Reshape
from keras.layers import ConvLSTM2D, Input,BatchNormalization, Conv2D
import os
from config import Config
from keras.callbacks import History 
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras.models import load_model
from keras.losses.mean_squared_error

#KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))
#import keras.backend.tensorflow_backend as KTF
#import tensorflow as tf

# 指定第一块GPU可用 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
K.set_session(sess)

class SRadModel():
    def ConvLSTM2D_model(self, input_shape):
        keras.backend.set_image_dim_ordering('tf')
        # This returns a tensor        

        if os.path.exists('model.h5'):
            # 如果模型已经存在，则加载模型，继续训练。
            print('模型已经存在，加载模型，继续训练。')
            model = load_model('model.h5')
            model.summary()  
        else:
            print('模型不存在，重新训练。')
            k = 4    
            go_backwards = False
            model = Sequential()
            model.add(ConvLSTM2D(filters=32, kernel_size=(k, k),
                                input_shape=input_shape, padding='same', return_sequences=True, 
                                activation='tanh', recurrent_activation='hard_sigmoid',
                                kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                                dropout=0.3, recurrent_dropout=0.3, go_backwards=go_backwards ))
            model.add(BatchNormalization())
            
            
            model.add(ConvLSTM2D(filters=32, kernel_size=(k, k), padding='same', return_sequences=True, 
                                activation='tanh', recurrent_activation='hard_sigmoid', 
                                kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                                dropout=0.4, recurrent_dropout=0.3, go_backwards=go_backwards ))
            model.add(BatchNormalization())

            model.add(ConvLSTM2D(filters=32, kernel_size=(k, k), padding='same', return_sequences=True, 
                                activation='tanh', recurrent_activation='hard_sigmoid', 
                                kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                                dropout=0.4, recurrent_dropout=0.3, go_backwards=go_backwards ))
            model.add(BatchNormalization())

            model.add(ConvLSTM2D(filters=32, kernel_size=(k, k), padding='same', return_sequences=True, 
                                activation='tanh', recurrent_activation='hard_sigmoid', 
                                kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                                dropout=0.4, recurrent_dropout=0.3, go_backwards=go_backwards ))
            model.add(BatchNormalization())
            
            
            model.add(ConvLSTM2D(filters=32, kernel_size=(k, k), padding='same', return_sequences=False, 
                                activation='tanh', recurrent_activation='hard_sigmoid', 
                                kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                                dropout=0.4, recurrent_dropout=0.3, go_backwards=go_backwards ))
            model.add(BatchNormalization())
            
            model.add(Conv2D(filters=3, kernel_size=(1, 1),
                        activation='sigmoid',
                        padding='same', data_format='channels_last'))         
            model.compile(optimizer='rmsprop', loss="fn_keras_rmse2")
            model.summary()                    
        self.model = model
        
    #def fn_keras_rmse(y_true, y_pred):
        #return K.sqrt(K.mean(K.square((y_pred*y_std) - (y_true*y_std))))

    def construct_model(self):
        # th input (channels, height, width)
        # tf input (height, width, channels)
        keras.backend.set_image_dim_ordering('tf')
        model = Sequential()

        #Encode:
        model.add(Convolution3D(16,(2,3,3), activation='relu', input_shape=(config.frames_per_input, 501, 501, 3),border_mode='same'))        
        model.add(MaxPooling3D(pool_size=(2,2,2)))
        model.add(Convolution3D(16,(2,3,3), activation='relu', border_mode='same'))
        model.add(Convolution3D(16,(2,3,3), activation='relu', border_mode='same'))
        model.add(MaxPooling3D(pool_size=(2,2,2)))
        model.add(Convolution3D(8,(1,3,3), activation='relu', border_mode='same'))
        #model.add(Convolution3D(8,(1,3,3), activation='relu', border_mode='same'))
        #model.add(MaxPooling3D(pool_size=(1,2,2)))        
        #model.add(Convolution3D(8,(1,3,3), activation='relu', border_mode='same'))
        #model.add(Convolution3D(8,(1,3,3), activation='relu', border_mode='same'))
        #Decode:
        model.add(Convolution3D(16,(2,3,3), activation='relu', border_mode='same'))
        model.add(UpSampling3D((1,2,2)))
        model.add(Convolution3D(16,(2,3,3), activation='relu', border_mode='same'))
        model.add(UpSampling3D((1,2,2)))
        model.add(Convolution3D(3,(2,3,3), activation='relu', border_mode='same'))
        # model.add(Convolution3D(16,(2,3,3), activation='relu', border_mode='same'))
        # model.add(UpSampling3D((1,1,1)))
        # model.add(Convolution3D(3,(2,3,3), activation='relu', border_mode='same'))
        # model.add(UpSampling3D((1,1,1)))
        # model.add(Flatten())

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.compile(loss=y_std, optimizer='adam')
        model.compile(loss='binary_crossentropy', optimizer='adam') # doesn't reset weights
        
        model.summary()        
        self.model = model

    def get_train_batch(self, train_samples):                        
        #每次读取batch_size个样本,每个样本里面61个图片    
        for sample_file in train_samples:
            data = np.load(sample_file) #单个样本包含61张图片
            data = data/255             #转换成0~1之间的数字             
            '''
            # 开始组织训练样本的input, output.
            @input: batch_size * 30 frames * height (501) * width (501) * channel (3)
            @output: batch_size * height (501) * width (501) * channel (3)
            '''            
            i = 0
            while i < config.picture_per_batch - config.frames_per_input - config.batch_size:
                batchX = []
                batchY = []
                for j in range(config.batch_size):
                    i += 1
                    batchX.append(data[i:i+config.frames_per_input , : , : , :])
                    batchY.append(data[i+config.frames_per_input, : , : , :])                
                yield np.array(batchX) , np.array(batchY)

    def get_validate_batch(self, val_samples):
        #每次读取batch_size个样本,每个样本里面61个图片            
        for sample_file in val_samples:            
            data = np.load(sample_file) #单个样本包含61张图片
            data = data/255             #转换成0~1之间的数字            
            '''
            # 开始组织训练样本的input, output.
            @input: batch_size * 30 frames * height (501) * width (501) * channel (3)
            @output: batch_size * height (501) * width (501) * channel (3)
            '''            
            i = 0
            while i < config.picture_per_batch - config.frames_per_input - config.batch_size:
                batchX = []
                batchY = []
                for j in range(config.batch_size):
                    i += 1
                    batchX.append(data[i:i+config.frames_per_input , : , : , :])
                    batchY.append(data[i+config.frames_per_input, : , : , :])                
                yield np.array(batchX) , np.array(batchY)

    def train(self):             
        # 获取所有的样本的文件名:
        bg = time.clock()
        self.get_all_samples()        
        np.random.seed(10)
        total_samples = len(self.all_samples_filenames)
        perm = np.random.permutation(total_samples)
        train_size = int(np.floor(total_samples*0.8))        
        train_samples = self.all_samples_filenames[perm[:train_size]]
        #val_samples = self.all_samples_filenames[perm[train_size:]]
        batchX, batchY  = next(self.get_train_batch(train_samples))
        print(np.shape(batchX) , np.shape(batchY))
        history = History()            
        history = self.model.fit_generator(self.get_train_batch(train_samples), max_q_size=10, steps_per_epoch=15000, epochs=1 ,verbose=1)
        end = time.clock()
        print('总训练总耗时间:%0.4f' % (end-bg))
        # 保存模型及参数：
        self.model.save('model.h5')        

    def predict(self, num_samples, num_frames):        
        if not os.path.exists('model.h5'):
            print('别着急亲，模型还没训练呢.')
            return None
        #模型已存在，则开始预测：
        for i in range(num_samples):
            X = next(self.get_test_batch())
        predict_images = self.model.predict(X)        
        #plt.imshow(pred_img[0])
        for img in predict_images:
            cv2.imwrite("predict/filename", img)        

if __name__ == '__main__':
    config = Config()
    SRadModel = SRadModel()
    #SRadModel.construct_model()
    input_shape = (None, 501, 501, 3)
    SRadModel.ConvLSTM2D_model(input_shape)
    #SRadModel.train()
    SRadModel.predict(30, 30)