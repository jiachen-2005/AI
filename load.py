import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# 这些是原始load.py中的变量名
_train_samples = None
_train_labels = None
_test_samples = None
_test_labels = None
image_size = None
num_labels = None
num_channels = None

def load_mfcc_data_improved(data_dir="processed_data_tf_shuffled", target_size=32):
    """
    加载MFCC数据并使用改进的方法下采样
    """
    global _train_samples, _train_labels, _test_samples, _test_labels
    global image_size, num_labels, num_channels
    
    print("=" * 60)
    print("加载MFCC数据（改进下采样）")
    print("=" * 60)
    
    # 1. 加载MFCC数据
    print(f"从 {data_dir} 加载数据...")
    
    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    label_encoder = joblib.load(f"{data_dir}/label_encoder.pkl")
    
    print(f"原始MFCC数据形状:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # 2. 改进的下采样方法
    print(f"\n使用改进方法下采样到{target_size}x{target_size}...")
    
    def improved_downsample(data, target_size=32):
        """
        使用插值下采样
        """
        batch_size, height, width, channels = data.shape
        
        if height == target_size and width == target_size:
            return data
        
        # 使用scipy的zoom进行下采样
        output = np.zeros((batch_size, target_size, target_size, channels), dtype=np.float32)
        
        for i in range(batch_size):
            # 计算缩放因子
            zoom_factors = (target_size/height, target_size/width, 1)
            
            # 应用缩放
            resized = zoom(data[i], zoom_factors, order=3)  # 三次样条插值
            
            # 确保形状正确
            if resized.shape != (target_size, target_size, channels):
                # 裁剪或填充
                resized = resized[:target_size, :target_size, :]
                if resized.shape[0] < target_size or resized.shape[1] < target_size:
                    padded = np.zeros((target_size, target_size, channels), dtype=np.float32)
                    padded[:resized.shape[0], :resized.shape[1], :] = resized
                    resized = padded
            
            output[i] = resized
        
        return output
    
    # 下采样
    X_train_resized = improved_downsample(X_train, target_size)
    X_test_resized = improved_downsample(X_test, target_size)
    
    print(f"下采样后数据形状:")
    print(f"  X_train: {X_train_resized.shape}")
    print(f"  X_test: {X_test_resized.shape}")
    
    def normalize_mfcc_improved(samples):
        """
        改进的归一化：保留更多信息
        """
      
        normalized = np.zeros_like(samples, dtype=np.float32)
        
        for i in range(len(samples)):
            sample = samples[i]
            mean = np.mean(sample)
            std = np.std(sample)
            std = max(std, 1e-8)
            
            normalized[i] = (sample - mean) / std
        
        return normalized
    
    print(f"\n归一化数据...")
    X_train_normalized = normalize_mfcc_improved(X_train_resized)
    X_test_normalized = normalize_mfcc_improved(X_test_resized)
    
    _train_samples = X_train_normalized
    _train_labels = y_train
    _test_samples = X_test_normalized
    _test_labels = y_test
    
    image_size = target_size
    num_labels = y_train.shape[1]
    num_channels = 1
    
    print(f"\n全局变量设置:")
    print(f"  image_size: {image_size}")
    print(f"  num_labels: {num_labels}")
    print(f"  num_channels: {num_channels}")
    
    return _train_samples, _train_labels, _test_samples, _test_labels, label_encoder

def get_global_variables():
    """获取全局变量"""
    return {
        '_train_samples': _train_samples,
        '_train_labels': _train_labels,
        '_test_samples': _test_samples,
        '_test_labels': _test_labels,
        'image_size': image_size,
        'num_labels': num_labels,
        'num_channels': num_channels
    }