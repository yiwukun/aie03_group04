import numpy as np
import keras ## broken for keras >= 2.0, use 1.2.2
from keras.models import Sequential,Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers.core import Flatten, Dense, Activation, Reshape
from keras.layers import ConvLSTM2D, Input,BatchNormalization, Conv2D
import os,sys
from config import Config
from keras.callbacks import History 
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras.models import load_model

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
            model.add(ConvLSTM2D(filters=16, kernel_size=(k, k),
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
            """
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
            """
            
            model.add(ConvLSTM2D(filters=16, kernel_size=(k, k), padding='same', return_sequences=False, 
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


    def get_train_batch(self, train_files):
        folder = config.root_path + '/train_processed/' + train_files           
        files = np.load(folder + '.npy')        
        for sample_file in files:
            sample_file = folder + '/' + sample_file
            #print(sample_file)
            data = np.load(sample_file) #单个样本包含61张图片
            data = (data-255)/255             #转换成 -1~0 之间的数字             
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

    def train(self, sample_file):             
        # 获取所有的样本的文件名:
        bg = time.clock()        
        batchX, batchY  = next(self.get_train_batch(sample_file))
        print(np.shape(batchX) , np.shape(batchY))
        history = History()            
        history = self.model.fit_generator(self.get_train_batch(sample_file), max_q_size=10, steps_per_epoch=config.steps_per_epoch, epochs=config.epochs ,verbose=1)
        end = time.clock()
        print('总训练总耗时间:%0.4f' % (end-bg))
        # 保存模型及参数：
        self.model.save('model.h5')        

    def predict(self, test_file):        
        import matplotlib.pyplot as plt
        import cv2
        import shutil        

        if not os.path.exists('model.h5'):
            print('别着急亲，模型还没训练呢.')
            #return None
        #模型已存在，则开始预测：        
        folder = config.root_path + '/test_processed/' + test_file
        files = np.load(folder + '.npy')
        for sample_file in files:                    
            sample_file = folder + '/' + sample_file                
            data = np.load(sample_file) #单个样本包含31张图片                     
            if len(data) < config.frames_per_test:
                continue             #如果图片不足frames_per_test,则跳过改样本
            

            submission_num = 0
            for i in range(30):                       
                data = data[-config.frames_per_test: , : , : , :]  #取该样本的最后 frames_per_test 帧进行图像预测.                
                X = np.expand_dims(data , axis=0)                
                X = (X -255)/255             #转换成 -1~0 之间的数字                
                predict_images = self.model.predict(X)
                #predict_images = np.random.random((1, 501, 501, 3))
                img = predict_images[0]                
                img = img * 255 + 255  #还原成255。
                plt.imshow(img)
                
                # 预测图片生产
                #RAD_000000000001000_f001.png                            
                pred_img_path = sample_file.replace('.npy', '_%03d' % i  +'.png')                                
                # 生成所有的图片：
                pred_img_path = pred_img_path.replace('test_processed',  '/predict')
                if not os.path.exists(os.path.dirname(pred_img_path)):
                    os.makedirs(os.path.dirname(pred_img_path))
                cv2.imwrite(pred_img_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])                                    
                # 生成大赛要求提交的图片：
                submission_folder = config.root_path + '/submission'
                submission = [4, 9, 14, 19, 24, 29]
                if not os.path.exists(submission_folder):
                    os.makedirs(submission_folder)                    
                if i in submission:
                    submission_num += 1
                    submission_img_path = submission_folder + '/' + sample_file[-23:-4] + '_f' + '%04d' % submission_num  +'.png'
                    print(submission_folder)
                    shutil.copy(pred_img_path , submission_img_path)                
                # 把预测的图片合并到X,然后取frames_per_test个作为输入，进行下次预测。
                data = np.concatenate((data, predict_images),axis=0)

if __name__ == '__main__':
    if len(sys.argv) == 2: 
        task = sys.argv[1]
        number = 1
    elif len(sys.argv) == 3: 
        task = sys.argv[1]
        number = sys.argv[2]
    else:    
        task = 'train'
        number = 1

    if task == 'train':
        sample_file = 'SRAD2018_TRAIN_0' + '%02d' % int(number)
    else:
        sample_file = 'SRAD2018_Test_1'

    config = Config()
    SRadModel = SRadModel()
    input_shape = (None, 501, 501, 3)
    SRadModel.ConvLSTM2D_model(input_shape)    
    if task == 'train':
        SRadModel.train(sample_file)
    else:
        SRadModel.predict(sample_file)