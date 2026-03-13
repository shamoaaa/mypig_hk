# -- coding: utf-8 -*-
# import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sosfiltfilt, hilbert, find_peaks
from core.utils import get_amp, butter_bandpass, calculate_envelope, dual_lowpass_filter, row_normalize
from core.find_pulse import calc_breath
from core.find_pulse_new import calc_breath as new_calc_breath


import logging
logger = logging.getLogger('__main__')

class PigOestrusFeatureExtractor:
    def __init__(self, act_channel=4, sample_rate=20):
        self.act_channel = act_channel
        self.sample_rate = sample_rate

        # 创建滤波器
        self.generate_filter()


    def init_space(self, min_num):
        self.act = {}
        self.act['amp'] = np.zeros(2*min_num)
        self.act['env'] = np.zeros(2*min_num)
        self.act['normal'] = np.zeros(2*min_num)

        self.br = {}
        self.br['not_good_data'] = np.zeros(2*min_num)
        self.br['is_high'] = np.zeros(2*min_num)
        self.br['main_freq'] = np.zeros(2*min_num)
        self.br['new_main_freq'] = np.zeros(2*min_num)
        self.br['ideal_channel'] = np.zeros(2*min_num)

        self.hr = {}
        self.hr['main_freq'] = np.zeros(2*min_num)

        self.is_up = np.zeros(2*min_num)

    
    def process(self, merged_data):
        self.l = merged_data.shape[0]
        self.min_num = int(self.l /20 / 60)

        # 初始化空间
        self.init_space(self.min_num)

        # 以60秒为窗口、30秒为步长，提取特征
        for min_index in range(2 * self.min_num - 1):
            # 截取1分钟数据
            start_index = 600 * min_index
            end_index = min(self.l, start_index + 1200)
            frame_data = merged_data[start_index:end_index, :]  # 1200 * 23

            self.get_feature(frame_data, min_index)

        self.br['not_good_data'][-1] = self.br['not_good_data'][-2]
        self.br['is_high'][-1] = self.br['is_high'][-2]
        self.br['main_freq'][-1] = self.br['main_freq'][-2]
        self.br['new_main_freq'][-1] = self.br['new_main_freq'][-2]
        self.br['ideal_channel'][-1] = self.br['ideal_channel'][-2]

        self.hr['main_freq'][-1] = self.hr['main_freq'][-2]

        return self.act, self.br, self.hr, self.is_up


    def get_feature(self, frame_data, min_index):
        # 获取幅值
        m, n = frame_data.shape
        amp = np.zeros([m, n])
        normalize_amp = np.zeros([m, n])
        weight_data = None
        for i_frame in range(m):
            normalize_amp[i_frame:i_frame+1, :] = row_normalize(np.abs(frame_data[i_frame:i_frame+1, :]))
            amp[i_frame, :], weight_data = get_amp(frame_data[i_frame, :], i_frame, weight_data)

        # 获取最佳通道
        ideal_channel = np.argmax(np.mean(np.abs(amp), axis=0))

        # 获取相位
        phase = np.unwrap(np.angle(frame_data[:, ideal_channel]))

        # 提取活动量
        act_mean_amp, act_env = self.get_act(amp[:, self.act_channel])

        # 提取呼吸
        br_freq, new_br_freq, _ = self.get_br(phase, act_mean_amp, ideal_channel, min_index)

        # 提取心跳
        hr_freq, _ = self.get_hr(phase, new_br_freq)

        # 获取起卧
        all_phase = np.unwrap(np.angle(frame_data[:, 0:5]))
        if min_index == 0:
            up_down = self.get_up_down(all_phase[0:600, :], amp[0:600, 0:5])
            self.is_up[min_index] = 1 if up_down else 0

            up_down = self.get_up_down(all_phase[600:1200, :], amp[600:1200, 0:5])
            self.is_up[min_index+1] = 1 if up_down else 0
        else:
            up_down = self.get_up_down(all_phase[600:1200, :], amp[600:1200, 0:5])
            self.is_up[min_index+1] = 1 if up_down else 0

        # 保存数据
        if min_index == 0:
            self.act['amp'][min_index] = np.mean(act_mean_amp[0:600])
            self.act['env'][min_index] = np.mean(act_env[0:600])
            self.act['normal'][min_index] = np.sum(np.square(np.diff(normalize_amp[0:600, self.act_channel])))

            self.act['amp'][min_index+1] = np.mean(act_mean_amp[600:1200])
            self.act['env'][min_index+1] = np.mean(act_env[600:1200])
            self.act['normal'][min_index+1] = np.sum(np.square(np.diff(normalize_amp[600:1200, self.act_channel])))
        else:
            self.act['amp'][min_index+1] = np.mean(act_mean_amp[600:1200])
            self.act['env'][min_index+1] = np.mean(act_env[600:1200])
            self.act['normal'][min_index+1] = np.sum(np.square(np.diff(normalize_amp[600:1200, self.act_channel])))

        self.br['main_freq'][min_index] = br_freq
        self.br['new_main_freq'][min_index] = new_br_freq
        self.br['ideal_channel'][min_index] = ideal_channel

        self.hr['main_freq'][min_index] = hr_freq


    def generate_filter(self):
        # 构建活动滤波器
        self.act_sos = butter_bandpass(0.1, 6, self.sample_rate, order=8)

        # 构建心率滤波器
        self.hr_sos = butter_bandpass(0.67, 2, self.sample_rate, order=8)

        # 构建起卧滤波器
        self.up_down_sos = butter_bandpass(0.1, 6, self.sample_rate, order=8)


    def get_act(self, act_amp):
        """
        计算活动水平

        :return: 活动水平的计算结果 (例如: 活动得分或活动标记)
        """
        act_amp = sosfiltfilt(self.act_sos, act_amp.T)

        # 滑动平均
        window_size = 1 * 20
        window = np.ones(window_size) / window_size
        act_mean_amp = np.convolve(act_amp, window, mode='same')

        # 获取包络线
        _, act_env = calculate_envelope(act_mean_amp, 100)

        return np.abs(act_mean_amp), np.abs(act_env)


    def get_br(self, phase, act_mean, ideal_channel, min_index):
        """
        计算呼吸率

        :return: 呼吸率 (breath per minute)
        """
        # 滤波
        filter_phase = dual_lowpass_filter(phase, 20, 0.1, 1.25, order=61)
        new_filter_phase = dual_lowpass_filter(phase, 20, 0.1, 1.95, order=61)
        
        _, _, _, _, _, main_freq = calc_breath(filter_phase, fs=20)
        _, _, _, _, _, new_main_freq = new_calc_breath(new_filter_phase, act_mean, ideal_channel, min_index, fs=20)

        try:
            return round(main_freq), round(new_main_freq), filter_phase
        except:
            return -1.0, -1.0, filter_phase


    def get_hr(self, phase, br_freq):
        """
        计算脉率

        :return: 脉率 (beats per minute)
        """
        # 滤波
        filter_phase = sosfiltfilt(self.hr_sos, phase)

        if br_freq == -1:
            return -1, filter_phase

        # 差分、归一化
        diff_phase = np.diff(filter_phase)
        analytic_signal = hilbert(diff_phase)
        hr_enhi = np.abs(analytic_signal)
        diff_phase = diff_phase / hr_enhi

        # 剔除异常值ֵ
        diff_phase[np.isinf(diff_phase)] = 1
        diff_phase[np.isnan(diff_phase)] = 1
        diff_phase[np.abs(diff_phase) < 1e-2] = 1
        diff_phase[np.abs(diff_phase) > 1e2] = 1

        # 获取主频
        locs, _ = find_peaks(-diff_phase, height=0.75, distance=10)
        rr_diff = np.diff(locs)
        main_freq = round(20.0 / np.median(rr_diff) * 60, 0)

        return main_freq, filter_phase


    def get_up_down(self, phase, amp):
        """
        计算起卧

        :return: True表示起，False表示卧
        """
        
        # 对相位进行滤波
        # 判断前5个通道的相位是否存在大于2
        is_over = False
        for i in range(phase.shape[1]):
            phase[:, i] = sosfiltfilt(self.up_down_sos,  phase[:, i])

            count = np.sum(phase[:, i] > 2)
            if count > 5:
                is_over = True

        if is_over:
            if max(amp[:, 1])>=1e4 and np.var(phase[:, 1], ddof=0)>=0.1 and max(amp[:, 2])>=1e4 and np.var(phase[:, 2], ddof=0)>=0.1:
                return True
            elif max(amp[:, 3])>=5e3 and np.var(phase[:, 3], ddof=0)>=0.5 and max(amp[:, 4])>=5e3 and np.var(phase[:, 4], ddof=0)>=0.5:
                return True
            else:
                return False
            
        return False
