class Config(object):
    # configuration variables
    rnn_size = 501*3 # 输入向量的维度
    time_step_size = 500 # 循环层长度

    batch_size = 2
    frames_per_input = 3
    picture_per_batch = 61

    test_batch_size = 1
    pictures_per_test = 31
    frames_per_test = 3

    root_path = 'data'
    epochs = 1
    steps_per_epoch = 10