import os
import cv2
import time
import imageio
import numpy as np
import sys
from config import Config

def convert_img_2_data(root_path, folder, sub_folder):        
    path = root_path + '/' + folder + '/' + sub_folder
    print('请确保文件已经解压缩在当前文件夹[%s]下' % path)
    i = 0    
    for root, dirs, img_files in os.walk(path):                
        i += 1
        begin = time.clock()
        imgdata = [cv2.imread( os.path.join(root, img)) for img in img_files]                    
        newfile = root.replace(folder, folder + '_processed')
        if os.path.exists(newfile + '.npy'):
            print('第[%i]个样本%s已经处理过'%(i , newfile))
            continue
        # 判断结果
        if not os.path.exists(os.path.dirname(newfile)):
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(os.path.dirname(newfile))
        np.save(newfile, imgdata)
        end = time.clock()
        print('生成第[%i]个样本:%s, 其中包含图片[%i]张,耗时:[%.2f]'%(i, newfile, len(imgdata), (end-begin)))


def save_sample_filenames(root_path, folder, sub_folder):        
    for root, dirs, sample_files in os.walk(root_path + '/' + folder + '/' + sub_folder):        
        sorted(sample_files)  #按照样本文件名的顺序排序.        
        np.save(root , sample_files) 
        print(sample_files[:3])
        print('[' + root + '.npy]保存成功,共计%s个样本' % len(sample_files))

if __name__ == '__main__':  
    config = Config()
    start = time.clock()
    print('开始处理图片:')
    if len(sys.argv) == 2: 
        folder = sys.argv[1]
        number = 1
    elif len(sys.argv) == 3: 
        folder = sys.argv[1]
        number = sys.argv[2]
    else:    
        folder = 'train'
        number = 1

    if folder == 'train':
        sub_folder = 'SRAD2018_TRAIN_0' + '%02d' % int(number)
    else:
        sub_folder = 'SRAD2018_Test_1'    
    
    convert_img_2_data(config.root_path , folder , sub_folder)
    save_sample_filenames(config.root_path , folder +'_processed', sub_folder)
    end = time.clock()
    print('处理时间为:%.2f' % (end-start))