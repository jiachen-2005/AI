if __name__ == '__main__':
    import scipy.io as sio
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from dp_refined import Network

    # 加载mat数据文件
    print('加载mat数据文件...')
    train_data = sio.loadmat('MFCC/mat_data_augmented/train.mat')
    test_data = sio.loadmat('MFCC/mat_data_augmented/test.mat')

    # 提取数据
    train_samples = train_data['X']
    train_labels = train_data['y']
    test_samples = test_data['X']
    test_labels = test_data['y']

    print('原始数据形状:')
    print('Training set', train_samples.shape, train_labels.shape)
    print('    Test set', test_samples.shape, test_labels.shape)

    # 转换标签格式并进行one-hot编码
    # 确保标签是正确的形状
    if train_labels.ndim == 2 and train_labels.shape[0] == 1:
        train_labels = train_labels.T
    if test_labels.ndim == 2 and test_labels.shape[0] == 1:
        test_labels = test_labels.T
    
    print('转置后标签形状:')
    print('Training labels', train_labels.shape)
    print('    Test labels', test_labels.shape)

    encoder = OneHotEncoder(sparse_output=False)
    train_labels = encoder.fit_transform(train_labels)
    test_labels = encoder.transform(test_labels)

    print('One-hot编码后:')
    print('Training set', train_samples.shape, train_labels.shape)
    print('    Test set', test_samples.shape, test_labels.shape)

    # 获取数据维度
    image_size = train_samples.shape[1]  # 32
    num_labels = train_labels.shape[1]  # 7
    num_channels = 1

    print('数据维度信息:')
    print('Image size:', image_size)
    print('Number of labels:', num_labels)
    print('Number of channels:', num_channels)


    # def train_data_iterator(samples, labels, iteration_steps, chunkSize):
    #     """
    #     Iterator/Generator: get a batch of data
    #     这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
    #     用于 for loop， just like range() function
    #     """
    #     if len(samples) != len(labels):
    #         raise Exception('Length of samples and labels must equal')
    #     stepStart = 0  # initial step
    #     i = 0
    #     while i < iteration_steps:
    #         stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
    #         yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
    #         i += 1
    def train_data_iterator(samples, labels, iteration_steps, chunkSize):
        """
        Iterator/Generator: get a batch of data
        这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
        用于 for loop， just like range() function
        """
        if len(samples) != len(labels):
            raise Exception('Length of samples and labels must equal')
        # ----------------------------------- CHANGED HERE -----------------------------------
        # reshuffle before each epoch to eliminate cyclic behavior
        from random import randint
        reshuffle = True
        stepStart = 0  # initial step
        lastStart = 0  # last step
        i = 0
        while i < iteration_steps:
            lastStart = stepStart
            stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
            if reshuffle and stepStart < lastStart:
                for n in range(0,len(samples)):
                    tmp = randint(n,len(samples)-1)
                    tmp_image = samples[n]
                    samples[n] = samples[tmp]
                    samples[tmp] = tmp_image
                    tmp_label = labels[n]
                    labels[n] = labels[tmp]
                    labels[tmp] = tmp_label
            yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
            i += 1


    def test_data_iterator(samples, labels, chunkSize):
        """
        Iterator/Generator: get a batch of data
        这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
        用于 for loop， just like range() function
        """
        if len(samples) != len(labels):
            raise Exception('Length of samples and labels must equal')
        stepStart = 0  # initial step
        i = 0
        while stepStart < len(samples):
            stepEnd = stepStart + chunkSize
            if stepEnd <= len(samples):
                yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
                i += 1
            stepStart = stepEnd


    net = Network(
        train_batch_size=64, test_batch_size=150, pooling_scale=2,
        dropout_rate=0.5,
        base_learning_rate=0.001, decay_rate=0.99)
    net.define_inputs(
        train_samples_shape=(64, image_size, image_size, num_channels),
        train_labels_shape=(64, num_labels),
        test_samples_shape=(150, image_size, image_size, num_channels),
    )
    #
    net.add_conv(patch_size=3, in_depth=num_channels, out_depth=16, activation='relu', pooling=True, name='conv1')
    net.add_conv(patch_size=3, in_depth=16, out_depth=32, activation='relu', pooling=True, name='conv2')
    """
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=False, name='conv3')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv4')
    """
    # 4 = 两次 pooling, 每一次缩小为 1/2
    # 32 = conv4 out_depth
    net.add_fc(in_num_nodes=(image_size // 4) * (image_size // 4) * 32, out_num_nodes=64, activation='relu',
               name='fc1')
    net.add_fc(in_num_nodes=64, out_num_nodes=num_labels, activation=None, name='fc2')

    net.define_model()
    # net.run(train_samples, train_labels, test_samples, test_labels, train_data_iterator=train_data_iterator,
    #         iteration_steps=3000, test_data_iterator=test_data_iterator)
    net.train(train_samples, train_labels, data_iterator=train_data_iterator, iteration_steps=2000)
    net.test(test_samples, test_labels, data_iterator=test_data_iterator)

else:
    raise Exception('main.py: Should Not Be Imported!!! Must Run by "python main.py"')
