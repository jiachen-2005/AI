# encoding:utf-8
from scipy.io import loadmat as load
import numpy as np
import matplotlib.pyplot as plt

def reformat(samples, labels):
    #  改变数据的形状
    #（图片高，图片宽，通道数，图片数） -》 （图片数，图片宽，图片高，通道数）
    #（   0,    1,    2,      3）  -》 （   3，   0,    1,      2）
    new = np.transpose(samples,(3,0,1,2)).astype(np.float32)

    #labels 变成 one-hot encoding
    labels = np.array([x[0] for x in labels])
    one_hot_labels = []

    for num in labels:
        one_hot = [0.0]*10
        if num ==10:
            one_hot[0]=1.0
        else:
            one_hot[num]=1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels)
    return new, labels


def normalize(samples):
    # (R+G+B)/3
    # 0~255 -> -1.0~1.0
    a = np.add.reduce(samples, keepdims=True, axis=3)
    a = a/3.0
    return a/128.0-1.0


def distribution(labels, name):
    labels_int = np.argmax(labels, axis=1) #axis=1 意思是沿着行方向找最大值,并返回其index
    counts = np.bincount(labels_int, minlength=10) #统计不同数值的个数
    adjusted_counts = np.r_[counts[1:],counts[0]]
    """
    label_freq_dict = dict(enumerate(counts))
    print(label_freq_dict)
    """
    digits_labels =[str(i) for i in range(1,10)] +['10']
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(10), adjusted_counts, color='steelblue', alpha=0.8)
    plt.bar_label(bars, fmt='%d', padding=0.5)
    plt.title(f'{name} Distribution') 
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.xticks(range(10),digits_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def inspect(dataset, labels, i):
    print(labels[i])
    plt.imshow(dataset[i].squeeze())
    plt.show()


train = load('data/train_32x32.mat')
test = load('data/test_32x32.mat')


train_samples = train['X']
train_labels = train['y']

# print(type(train_samples))
#print(train_labels)

test_samples = test['X']
test_labels = test['y']

n_train_samples, _train_labels = reformat(train_samples,train_labels)
n_test_samples, _test_labels = reformat(test_samples, test_labels)

_train_samples = normalize(n_train_samples)
_test_samples = normalize(n_test_samples)



num_labels = 10
image_size = 32
num_channels = 1

if __name__ == '__main__':
    _train_samples = normalize(_train_samples)
    #print(_train_samples.shape)
    #inspect(_train_samples, _train_labels, 9641)
    distribution(_train_labels, 'Train Labels')

