import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore") # 忽略所有警告

import traceback
import numpy as np
import pandas as pd

from core.utils import generate_date_strings, get_merged_data_hdf5
import common_value


hdf5_dir = r'/data2/wens_data/dataset/faqing_peizhong/train/'  # 训练集/测试集
out_dir = r'/data1/chenming40/old_feature/'
out_features = os.path.join(out_dir, 'train_feature')  # 特征存储路径
out_fig = os.path.join(out_dir, 'fig')  # 图像存储路径
cpu_count = 64

# hdf5_dir = r'/data/chenming40/data/dataset/faqing_peizhong/test/'  # 训练集/测试集
# out_dir = r'/data1/chenming40/old_feature/'
# out_features = os.path.join(out_dir, 'test_feature')  # 特征存储路径
# out_fig = os.path.join(out_dir, 'fig')  # 图像存储路径
# cpu_count = 48

select_all_flag = True  # 是否选择特定日期和场地
select_data = ['20241113', '20241120']
select_changdi = 'shuitai'

date_list = generate_date_strings(select_data)

# 选择相关所有文件列表
all_file_lists = []
for path1 in os.listdir(hdf5_dir):
    cur_exp_path = os.path.join(hdf5_dir, path1)
    if (os.path.isdir(cur_exp_path) == False):
        continue

    for pid_id in os.listdir(cur_exp_path):
        pig_id_path = os.path.join(cur_exp_path, pid_id)
        if (os.path.isdir(pig_id_path) == False):
            continue
        pig_id_date_list = os.listdir(pig_id_path)
        pig_id_date_list_path = [os.path.join(pig_id_path, item) for item in pig_id_date_list]
        all_file_lists.extend(pig_id_date_list_path)

# 新的符合条件的路径列表
filtered_file_lists = []

if (select_all_flag):
    # 如果选择了所有数据，则直接使用所有文件列表
    filtered_file_lists = all_file_lists
else:
    # 遍历每个文件路径进行筛选
    for file_path in all_file_lists:
        dir_levels = file_path.split(os.sep)

        cur_changdi = dir_levels[-3]
        cur_changdi = cur_changdi.split('_')[-2]  # 获取场地名称
        cur_date = dir_levels[-1]
        if (cur_changdi in select_changdi and cur_date in date_list) and 'shuitai_12' in file_path:
            filtered_file_lists.append(file_path)


para = []
for date_path in filtered_file_lists:

    if os.path.exists(date_path):
        file_list = [os.path.join(date_path, name) for name in os.listdir(date_path) if os.path.isfile(os.path.join(date_path, name))]
    else:
        continue

    for file in file_list:

        # 创建结果输出目录
        out_path = ''
        if date_path.startswith(hdf5_dir):
            sufix_path = date_path[len(hdf5_dir):]
            out_path = os.path.join(out_features, sufix_path[:])
        else:
            print('error, dataset struct error: %s' % file)

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        para.append((file, out_path))


from core.pig_oestrus_features import PigOestrusFeatureExtractor
import multiprocessing


def run(file, out_path):

    try:

        filename, _ = os.path.splitext(os.path.basename(file))
        out_file = os.path.join(out_path, f"{filename}-features.csv")
        
        if not os.path.exists(out_file):
        
            # 创建PigOestrusAnalyzer实例
            if 'houbei' in file:
                analyzer = PigOestrusFeatureExtractor(6, 20)
            else:
                analyzer = PigOestrusFeatureExtractor(4, 20)

            # 获取hdf5数据
            merged_data = get_merged_data_hdf5(file)  # 72000 * 23

            # 获取活动水平、呼吸、心率
            act_level, br_rate, hr_rate, is_up = analyzer.process(merged_data)

            array_2d = np.column_stack((act_level['amp'], br_rate['main_freq'], hr_rate['main_freq'], is_up, br_rate['ideal_channel'], act_level['normal'], br_rate['new_main_freq']))
            df = pd.DataFrame(array_2d, columns=['act', 'br', 'hr', 'is_up', 'ideal_channel', 'normalize_act', 'new_br'])
            
            df.to_csv(out_file, index=False, na_rep='-1')

        common_value.add_finish_num()
        progress = (common_value.get_finish_num() / common_value.get_total_num()) * 100
        print(f'\rOverall progress: {round(progress, 3)}%', end='', flush=True)
    
    except Exception as e:
        if os.stat(file).st_size == 0:
            os.remove(file)
        else:
            print(file, os.stat(file), e)
        # traceback.format_exc()


total_num = len(para)
print(total_num)
common_value.set_total_num(total_num)

p = multiprocessing.Pool(cpu_count)
p.starmap(run, para)

p.close()
p.join()
