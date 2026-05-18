import numpy as np
from python_speech_features import mfcc, delta
import librosa
import soundfile as sf
from pydub import AudioSegment
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


def trans_mp3_to_wav(newname,filepath):
    voice = AudioSegment.from_mp3(filepath)
    voice.export(newname, format='wav')


def get_ms_part_wav(main_mp3_path, start_time, end_time, part_wav_path):
    start_time = int(start_time)
    end_time = int(end_time)
    sound = AudioSegment.from_mp3(main_mp3_path)
    output = sound[start_time : end_time]
    output.export(part_wav_path, format="wav")


def detect_voice_activity(audio, sample_rate=8000):
    """
    简单的语音活动检测
    使用能量阈值法检测音频中的有效语音段
    
    参数:
        audio: 音频数据
        sample_rate: 采样率
    
    返回:
        有效段的开始和结束索引
    """
    # 计算能量
    energy = np.abs(audio)
    
    # 设置能量阈值（基于音频能量的平均值）
    threshold = np.mean(energy) * 0.3
    
    # 找到能量高于阈值的索引
    voice_indices = np.where(energy > threshold)[0]
    
    if len(voice_indices) == 0:
        return 0, len(audio)
    
    # 合并连续的语音帧
    start_idx = voice_indices[0]
    end_idx = voice_indices[-1]
    
    # 扩展一点边界，确保不丢失语音
    frame_size = int(sample_rate * 0.03)  # 30毫秒帧
    start_idx = max(0, start_idx - frame_size * 2)
    end_idx = min(len(audio), end_idx + frame_size * 2)
    
    return start_idx, end_idx


def get_mfcc(wav_path):
    fs, audio = wav.read(wav_path)
    # 语音有效段检测
    start_idx, end_idx = detect_voice_activity(audio, sample_rate=fs)
    audio = audio[start_idx:end_idx]
    # 提取MFCC特征
    mfcc_features = mfcc(audio, samplerate=8000, numcep=16)
    # 截补到统一形状(1024,16)
    target_length = 1024
    current_length = mfcc_features.shape[0]
    
    if current_length == target_length:
        return mfcc_features
    elif current_length > target_length:
        # 截取中间部分
        start_idx = (current_length - target_length) // 2
        return mfcc_features[start_idx:start_idx + target_length, :]
    else:
        # 填充静音（使用零值填充）
        padding = np.zeros((target_length - current_length, 16), dtype=mfcc_features.dtype)
        return np.concatenate((mfcc_features, padding), axis=0)


if __name__ == "__main__":
    get_ms_part_wav('VR.mp3', 1000, 6000, 'VR.wav')
    orig1 = get_mfcc('./audio_raw/processed/大白菜_000.wav')
    print(orig1.shape)
    plt.plot(orig1)
    plt.show()