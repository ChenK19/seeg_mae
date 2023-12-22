import mne.filter
import numpy as np

from utils import *
from configs import *
from mat73 import loadmat
from scipy import signal
import tqdm

def zscore(a, axis):
    #from https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_py.py#L2662-L2730
    mn = a.mean(axis=axis, keepdims=True)
    std = a.std(axis=axis, ddof=0, keepdims=True)

    std[(std==0)] = 1.0 #this is a hack. I should eventually find where the bad data is
    z = (a - mn) / std
    return z

def get_stft(x, fs, show_fs=-1, nperseg=128, noverlap=96, normalizing=None, **kwargs):
    f, t, Zxx = signal.stft(x, fs, nperseg=nperseg, noverlap=noverlap, **kwargs)

    if "return_onesided" in kwargs and kwargs["return_onesided"] == True:
        Zxx = Zxx[:show_fs]
        f = f[:show_fs]
    else:
        pass #TODO
        #Zxx = np.concatenate([Zxx[:,:,:show_fs], Zxx[:,:,-show_fs:]], axis=-1)
        #f = np.concatenate([f[:show_fs], f[-show_fs:]], axis=-1)
    Zxx = np.abs(Zxx)

    if normalizing=="zscore":
        Zxx = zscore(Zxx, axis=-1)#TODO is this order correct? I put it this way to prevent input nans
        if (Zxx.std() == 0).any():
            Zxx = np.ones_like(Zxx)

        Zxx = Zxx[:,10:-10]

    elif normalizing=="db":
        Zxx = np.log(Zxx)

    if np.isnan(Zxx).any():
        Zxx = np.nan_to_num(Zxx, nan=0.0)

    # return f, t, torch.Tensor(np.transpose(Zxx))
    return f, t, Zxx

def preprocess_data(data_folder, elec_info_folder, data_save_folder, is_freq=None, nperseg=400, noverlap=390):
    file_list = getFilelist(folder_path=data_folder, ext='.mat')

    for patient_file_path in file_list:
        string_line = patient_file_path.split('/')
        recording_machine = string_line[-2]

        patient_name = string_line[-1][:-17]
        print(patient_name)

        if patient_name == 'Liminfeng' or patient_name == 'Wangdepan' or patient_name == 'Songqili' or patient_name == 'Shendanfeng':
            continue

        electrode_info_path = os.path.join(elec_info_folder, patient_name + '.xlsx')

        br, x, y, z = electrode_region(location_xlsx_file=electrode_info_path, patient_name=patient_name)

        if recording_machine == 'natus':
            cfg = cfg_natus
        elif recording_machine == 'nihon':
            cfg = cfg_nihon

        data = loadmat(patient_file_path)
        data = data['data']
        data = data.astype(np.float64)

        downsample_freq = cfg.downsample_freq

        sample_freq = cfg.sample_freq
        window_length = cfg.window_length

        data = mne.filter.resample(data, down=sample_freq/downsample_freq, n_jobs=-1)


        for ch in tqdm.tqdm(range(data.shape[0])):
            for t in range(int(data.shape[1] / (window_length * downsample_freq)) - 1):
                sample_save_path = '{}/{}_ch_{}_time_{}_{}s.pkl'.format(data_save_folder, patient_name, ch, t, window_length)
                sample = data[ch, t * window_length * downsample_freq : (t+1) * window_length * downsample_freq]
                if not is_freq:
                    sample_dict = {
                        'sample': sample,
                        'ch': ch,
                        'br': br[ch],
                        'time_start': t * window_length,
                        'time_end': (t + 1) * window_length,
                        'sample_freq': downsample_freq,
                        'x': x[ch],
                        'y': y[ch],
                        'z': z[ch]
                    }

                else:
                    f, t, sample = get_stft(sample, fs=downsample_freq, normalizing='db', nperseg=nperseg, noverlap=noverlap) #
                    sample_dict = {
                        'sample': sample,
                        'ch': ch,
                        'br': br[ch],
                        'time_start': t * window_length,
                        'time_end': (t + 1) * window_length,
                        'sample_freq': downsample_freq,
                        'x': x[ch],
                        'y': y[ch],
                        'z': z[ch],
                        't_resolution': t,
                        'f_resolution': f,
                        'nperseg': nperseg,
                        'noverlap': noverlap
                    }
                    # print('debug')

                dict2binary(binary_file_path=sample_save_path, data_dict=sample_dict)

        # print('debug')



def get_mean_std(data_folder, mean_std_save_folder, is_freq=False):

    file_list = getFilelist(folder_path=data_folder, ext='.pkl')
    sample_mean = 0
    sample_std = 0

    print('######### start computing mean ########')
    for f in file_list:
        sample_dict = binary2dict(f)
        sample = sample_dict['sample']
        sample_mean = sample_mean + sample
    sample_mean = np.mean(sample_mean / len(file_list))

    for f in file_list:
        sample_dict = binary2dict(f)
        sample = sample_dict['sample']
        sample_std = sample_std + (sample - sample_mean)**2
    sample_std = np.sqrt(np.mean(sample_std / len(file_list)))

    print('############# start computing std ############')
    mean_std = {
        'mean': sample_mean,
        'std': sample_std,
        'sample_numbers': len(file_list)
    }

    dict2binary(binary_file_path=os.path.join(mean_std_save_folder, 'mean_std_is_freq_{}.pkl'.format(is_freq)), data_dict=mean_std)

if __name__ == "__main__":
    data_folder = '/home/jz/hard_disk/data/20230516_ane_downsample_compare/20230516_ane_data/data'
    electrode_info_folder = '/home/jz/hard_disk/data/20230516_ane_downsample_compare/20230516_ane_data/Bipolar_coor/'
    data_save_folder = '/home/jz/hard_disk/data/20231213_anes_data/5s_segment_400Hz'

    mean_std_save_folder = '/home/jz/hard_disk/data/20231213_anes_data/'
    is_freq = False

    # preprocess_data(data_folder=data_folder, elec_info_folder=electrode_info_folder, data_save_folder=data_save_folder, is_freq=is_freq)
    get_mean_std(data_folder=data_save_folder, mean_std_save_folder=mean_std_save_folder, is_freq=is_freq)