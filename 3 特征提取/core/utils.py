import logging
import os
import json

import h5py
import numpy as np

import scipy.signal as signal
from scipy.signal import firwin, filtfilt
from scipy.signal import butter, zpk2sos, detrend, firwin
from datetime import datetime, timedelta


logger = logging.getLogger('__main__')


def generate_date_strings(data_limit):
    # 解析输入的字符串，将其转换为 datetime 对象
    start_date = datetime.strptime(data_limit[0], '%Y%m%d')
    end_date = datetime.strptime(data_limit[1], '%Y%m%d')

    # 初始化结果数组和当前处理的日期
    date_strings = []
    current_date = start_date

    # 循环直到当前日期超过结束日期
    while current_date <= end_date:
        # 将当前日期格式化为字符串并添加到结果数组
        date_strings.append(current_date.strftime('%Y%m%d'))
        # 在当前日期上加一天
        current_date += timedelta(days=1)
    
    return date_strings


def decimal_to_ascii(decimal_number):
    # 将十进制数转换为十六进制字符串，不带'0x'前缀
    hex_string = format(decimal_number, 'x')
    
    # 如果长度是奇数，在前面补一个零
    if len(hex_string) % 2 != 0:
        hex_string = '0' + hex_string

    # 初始化结果字符串
    ascii_string = ""

    # 以两个字符为一组遍历十六进制字符串
    for i in range(0, len(hex_string), 2):
        hex_char = hex_string[i:i+2]
        # 将每组十六进制字符转换为十进制，再转换为ASCII字符
        ascii_string += chr(int(hex_char, 16))

    return ascii_string


def get_msg(file_path):
    folder = os.path.dirname(file_path)
    full_filename = os.path.basename(file_path)
    filename, _ = os.path.splitext(full_filename)

    dev_id = str.split(filename, '-')[0]
    date = str.split(filename, '-')[1][0:8]
    hour = str.split(filename, '-')[1][8:]

    msg = {
        'folder': folder,
        'full_filename': full_filename,
        'filename': filename,
        'dev_id': dev_id,
        'date': date,
        'hour': hour
    }

    return msg


def get_merged_data_hdf5(hdf5_file):
    try:
        rows_minutes = 60
        n_cols = 23
        merged_data = np.full([rows_minutes * 1200, n_cols], np.nan).astype(np.complex64)

        with h5py.File(hdf5_file, 'r') as file:
            cur_data = file['data'][:]
            frame_ids = file['frame_ids'][:]

        start_id = frame_ids[0]
        start_index = 0
        frame_len = 5 * 20
        for x, id in enumerate(frame_ids):
            # print(x, id, start_index)
            if start_index >= 72000:
                break

            if x == 0:
                if not frame_ids[0]+1 == frame_ids[1]:
                    if frame_ids[1]+1 == frame_ids[2]:
                        start_id = frame_ids[1]
                        continue

            if id == start_id:
                merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

                start_id = id + 1
                start_index += frame_len

            elif id > start_id:

                if x+1 == len(frame_ids):
                    merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

                    start_id = id + 1
                    start_index += frame_len
                else:
                    if frame_ids[x+1] == start_id:
                        continue
                    else:
                        start_index += (id - start_id) * frame_len

                        merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

                        start_index += frame_len
                        start_id = id + 1
            
            elif id < start_id:
                if x+1 == len(frame_ids):
                    merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

                    start_id = id + 1
                    start_index += frame_len
                else:
                    if frame_ids[x+1] == start_id:
                        continue
                    else:
                        merged_data[start_index: start_index+frame_len, :] = cur_data[x*frame_len: (x+1)*frame_len, :]

                        start_id = id + 1
                        start_index += frame_len

        # 如果丢太多，就用老版本的代码
        lost_frame = np.sum(np.isnan(merged_data)) // 2300
        all_frame = len(frame_ids)
        if lost_frame / all_frame >= 0.5:
            return get_merged_data_hdf5_old(hdf5_file)
        else:
            # 使用前一个有效值填充缺失值
            merged_data = fill_missing_with_zero_vectorized(merged_data)
            return merged_data
    except:
        return get_merged_data_hdf5_old(hdf5_file)
    

def get_merged_data_hdf5_old(hdf5_file_path):

    rows_minutes = 60
    n_cols = 23
    merged_data = np.full([rows_minutes * 1200, n_cols], np.nan).astype(np.complex64)

    with h5py.File(hdf5_file_path, 'r') as file:
        cur_data = file['data'][:]

    logger.info(f"Get merged data from {hdf5_file_path}, the data have minutes: {len(cur_data)/1200}")
    if len(cur_data) > rows_minutes * 1200:
        merged_data[:, :] = cur_data[0:rows_minutes * 1200, :]
    else:
        merged_data[0:len(cur_data), :] = cur_data
    merged_data = fill_missing_with_zero_vectorized(merged_data)

    return merged_data


def fill_missing_with_zero_vectorized(arr):
    # 创建结果数组副本
    result = arr.copy()
    
    # 按列处理
    for col in range(arr.shape[1]):
        column_data = arr[:, col]
        mask = np.isnan(column_data)
        
        # 找到非NaN值的索引
        valid_indices = np.where(~mask)[0]
        
        if len(valid_indices) == 0:
            # 如果整列都是NaN，全部填充0
            result[:, col] = 0
            continue
        
        # 构建索引数组
        idx = np.arange(len(column_data))
        # 对于每个位置，找到前一个有效值的索引
        idx = np.maximum.accumulate(np.where(~mask, idx, 0))
        
        # 获取填充值
        filled_values = column_data[idx]
        
        # 处理开头的NaN（idx=0的位置且原始值为NaN）
        filled_values[(idx == 0) & mask] = 0 + 0j
        
        result[:, col] = filled_values
    
    return result


def get_angle(data):
    result = np.zeros_like(data, dtype=np.float32)
    for i, row in enumerate(data):
        for j, d in enumerate(row):
            result[i, j] = np.angle(d)

    return result


def get_amp(data, i_frame, weight_data):
    weight = 0.95

    if i_frame == 0:
        weight_data = data
        diff_data = np.ones_like(data)
    else:
        if i_frame == 1:
            diff_data = data - weight_data
            weight_data = 0.5 * data + 0.5 * weight_data
        else:
            diff_data = data - weight_data
            weight_data = (1 - weight) * data + weight * weight_data

    diff_data = np.abs(diff_data)

    return diff_data, weight_data


def butter_bandpass(lowcut, highcut, fs=20, order=8):
    z, p, k = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='zpk')
    sos = zpk2sos(z, p, k)
    return butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')



def butter_lowpass(lowcut, fs=20, order=8):
    z, p, k = butter(order, lowcut, btype='lowpass', fs=fs, output='zpk')
    sos = zpk2sos(z, p, k)
    return sos


def fir_bandpass_filter(data, sample_rate=20, lowcut=0.1, highcut=1.25, order=61):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b = signal.firwin(order, [low, high], pass_zero=False)
    filtered_data = signal.lfilter(b, 1.0, data)
    return filtered_data


def fir_lowpass_diff(signal_input, fs=20, cutoff=2.0, numtaps=21, diff_order=1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    fir_coeff = signal.firwin(numtaps, normal_cutoff, window='hamming')
    filtered_signal = signal.lfilter(fir_coeff, 1.0, signal_input)
    if diff_order == 1:
        diff_signal = np.diff(filtered_signal)
    elif diff_order > 1:
        diff_signal = np.diff(filtered_signal, n=diff_order)
    else:
        diff_signal = filtered_signal.copy()
    return filtered_signal, diff_signal


def dual_lowpass_filter(signal, fs, low_cutoff_1, low_cutoff_2, order=61):
    nyq = 0.5 * fs  # 奈奎斯特频率
    
    # 设计第一个低通滤波器 (0.1Hz)
    coeff_1 = firwin(
        order + 1, 
        low_cutoff_1 / nyq,  # 归一化截止频率
        window='hamming',
        pass_zero='lowpass'
    )
    
    # 设计第二个低通滤波器 (1.25Hz)
    coeff_2 = firwin(
        order + 1,
        low_cutoff_2 / nyq,  # 归一化截止频率
        window='hamming',
        pass_zero='lowpass'
    )
    
    # 零相位滤波
    filtered_1 = filtfilt(coeff_1, 1.0, signal)  # 0.1Hz低通结果
    filtered_2 = filtfilt(coeff_2, 1.0, signal)  # 1.25Hz低通结果
    
    # 逐点相减：1.25Hz结果 - 0.1Hz结果
    return filtered_2 - filtered_1


def calc_br_by_peaks(peaks):
    len_peaks = len(peaks)
    if len(peaks) < 2:
        return None, None
    RR = peaks[1] - peaks[0]
    cur_seq = [peaks[0]]
    final_seqs = []
    for i in range(1, len_peaks):
        new_RR = peaks[i] - peaks[i-1]
        if new_RR > 0.7 * RR and new_RR < 1.3 * RR:
            cur_seq.append(peaks[i])
            RR = new_RR
            if i == len_peaks - 1:
                final_seqs.append(cur_seq)
        else:
            final_seqs.append(cur_seq)
            cur_seq = [peaks[i]]
            RR = new_RR

    if len_peaks > 30:
        num_thr = 0.4
    elif len_peaks > 20:
        num_thr = 0.5
    elif len_peaks > 15:
        num_thr = 0.6
    elif len_peaks > 10:
        num_thr = 0.6
    else:
        num_thr = 0.6

    # print(final_seqs)
    sub_max = None
    max_len = -1
    for item in final_seqs:
        item_len = len(item)
        if item_len > len_peaks * num_thr:
            if item_len > max_len:
                sub_max = item
                max_len = item_len

    if sub_max is not None:
        # print(sub_max)
        secs = (sub_max[-1] - sub_max[0])/20
        periods = max_len - 1 
        br = periods * 60 / secs 
    else:
        br = None
        secs = None
    return br, secs


def ampd(x, scale=None, debug=False):
    x = detrend(x)
    N = len(x)
    L = N // 2
    if scale:
        L = min(scale, L)

    # create LSM matix
    LSM = np.ones((L, N), dtype=bool)
    for k in np.arange(1, L + 1):
        LSM[k - 1, 0:N - k] &= (x[0:N - k] > x[k:N]
                                )  # compare to right neighbours
        LSM[k - 1, k:N] &= (x[k:N] > x[0:N - k])  # compare to left neighbours

    # Find scale with most maxima
    G = LSM.sum(axis=1)
    G = G * np.arange(
        N // 2, N // 2 - L, -1
    )  # normalize to adjust for new edge regions
    l_scale = np.argmax(G)

    if l_scale == 0:
        return np.array([-1])

    # find peaks that persist on all scales up to l
    pks_logical = np.min(LSM[0:l_scale, :], axis=0)
    pks = np.flatnonzero(pks_logical)
    if debug:
        return pks, LSM, G, l_scale
    return pks


def row_normalize(matrix):
    """行归一化：每行的值缩放到0-1之间"""
    row_mins = matrix.min(axis=1, keepdims=True)
    row_maxs = matrix.max(axis=1, keepdims=True)
    return (matrix - row_mins) / (row_maxs - row_mins + 1e-10)  # 加小量防止除以


def calculate_envelope(signal, window_size):
    envelope_low = np.zeros_like(signal)
    envelope_high = np.zeros_like(signal)
    half_window = window_size // 2

    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window)
        envelope_low[i] = np.min(signal[start:end])
        envelope_high[i] = np.max(signal[start:end])

    return envelope_low, envelope_high


def load_dict(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
    except json.JSONDecodeError:
        logger.error("Error decoding JSON from the file.")
    except IOError as e:
        logger.error(f"An error occurred while reading the file: {e.strerror}")

    return data


def save_dict(file_path, data):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
            logger.info(f"Data has been written to {file_path}")
    except IOError as e:
        logger.error(f"An error occurred while writing to the file: {e.strerror}")


def get_freq_by_pks(pks):
    rr_list = np.diff(pks)
    return 60*20/np.median(rr_list)
