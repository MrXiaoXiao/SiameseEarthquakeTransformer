# Will be cleaned and updated after 05.03.2021

import numpy as np
from pathlib import Path
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import obspy
import bisect
from pyproj import Geod
import sys
sys.path.append('../../S_EqT_codes/src/EqT_libs')
from src.S_EqT_model import S_EqT_Model_create
from src.S_EqT_model import S_EqT_Model_seprate
from src.S_EqT_model import S_EqT_RSRN_Model
from src.S_EqT_model import S_EqT_HED_Model
from src.S_EqT_concate_fix_corr import S_EqT_Concate_RSRN_Model
from src.misc import get_train_list
import keras.backend
import yaml
from random import shuffle
import os

# Simplified steps:
# 01 load config file
# 02 search all picks (aggressive mode or lasy mode)
# 03 save results

"""
def get_search_station_list(station, station_list, max_distance):
    search_list = list()
    for t_sta in station_list:
        # skip self
        if t_sta[0] == station[0]:
            continue
        # calculate distance
        t_dis = GEOD.inv(t_sta[3],t_sta[2],station[3],station[2])[2]/1000.0
        if t_dis > max_distance:
            continue
        search_list.append(t_sta)
    return search_list

station_list = list()
s_eqt_dict = dict()
station_list_file = open(sta_list_file_path,'r')
sta_id = 0
for line in station_list_file.readlines():
    if len(line) < 3:
        continue
    splits = line.split(' ')
    sta_name = splits[2]+'.'+splits[3]
    
    s_eqt_dict[sta_name] = dict()
    s_eqt_dict[sta_name]['P'] = list()
    s_eqt_dict[sta_name]['S'] = list()
    s_eqt_dict[sta_name]['P_Prob'] = list()
    s_eqt_dict[sta_name]['S_Prob'] = list()
    
    sta_lat = float(splits[1])
    sta_lon = float(splits[0])
    
    station_list.append( (sta_id, sta_name, sta_lat, sta_lon) )
    sta_id += 1

keras.backend.set_floatx('float32')
# load configuration files
cfgs = yaml.load(open('S_Test_example.yaml','r'),Loader=yaml.BaseLoader)
if_encoder_concate = int(cfgs['Model']['Encoder_concate'])
if if_encoder_concate == 1:
    encode_model, siamese_model, EqT_model = S_EqT_Concate_RSRN_Model(cfgs)
    encoder_encoded_list = cfgs['Model']['Encoder_concate_list']
    encoder_encoded_lengths = cfgs['Model']['Encoder_concate_lengths']
    encoder_encoded_channels = cfgs['Model']['Encoder_concate_channels']
        
elif int(cfgs['Model']['MODEL_RSRN'] == 0):
    encode_model, siamese_model, EqT_model = S_EqT_HED_Model(cfgs)
else:
    encode_model, siamese_model, EqT_model = S_EqT_RSRN_Model(cfgs)

if int(cfgs['Model']['LoadPretrainedModel']) == 1:
    siamese_model.load_weights(cfgs['Model']['PretrainModelPath'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='01_download_data_from_IRIS')
    parser.add_argument('--config-file', dest='config_file', 
                        type=str, help='Configuration file path',default='./default_pipline_config.yaml')
    args = parser.parse_args()
    cfgs = yaml.load(open(args.config_file,'r'),Loader=yaml.SafeLoader)
    siamese_model.load_weights('/home/Public/SiameseEQTransformer/SiameseEQT/models/S_Add_Noise_1023_UNI_NEW_model330000.hdf5')
    # For P or S
    # difference is velocity and trained model and encoded model
    # set up model
    base_time = obspy.UTCDateTime(2010,1,1)
    # search params
    # search distance in km
    max_search_distance = 30
    # template P threshold 0.5
    template_P_min_threshold = 0.5
    template_S_min_threshold = 0.5

    VpRefMin = 3.5
    VsRefMin = 2.5

    P_boundary_time = 1.0
    S_boundary_time = 1.0

    sta_list_file_path = '/mnt/BYEB/LA_JAN/LA_DATA/PostProcess/LAN_REAL_Use.dat'
    prev_file_str = '/mnt/BYEB/LA_JAN/LA_DATA/PostProcess/EQTRes/20100101/'
    search_hdf5_prefix =  '/mnt/BYEB/LA_JAN/LA_DATA/LA_2010_Jan_processed_hdfs/'
    detections_prefix = '/mnt/BYEB/LA_JAN/LA_DATA/LA_2010_Jan_detections/'
    GEOD = Geod(ellps='WGS84')
    # Rough meters/degree calculation
    M_PER_DEG = (GEOD.inv(0, 0, 0, 1)[2] + GEOD.inv(0, 0, 1, 0)[2]) / 2

    keep_t = None
    for sta_key in s_eqt_dict.keys():
        S_EqT_P_list = list()
        #print(sta_key)

        s_times = list()
        s_probs = list()
        
        prev_file = prev_file_str + '{}.S.txt'.format(sta_key)
        if os.path.exists(prev_file):
            f = open(prev_file,'r')
            for line in f.readlines():
                if len(line) > 3:
                    s_times.append(float(line.split(' ')[0]))
                    s_probs.append(float(line.split(' ')[1]))
            f.close()
        else:
            print(prev_file)
        #print(indexs)
        #s_times = s_times[indexs]
        #s_probs = s_probs[indexs]
        s_eqt_dict[sta_key]['S'] = s_times
        s_eqt_dict[sta_key]['S_Prob'] = s_probs

    RSRN_lengths = cfgs['Model']['RSRN_Encoded_lengths']
    RSRN_channels = cfgs['Model']['RSRN_Encoded_channels']


    plot_dx = 0
sta_search_dx = 0
plot_num = 0
for sta in station_list:
    sta_search_dx += 1
    print('On {} of {}'.format(sta_search_dx, len(station_list)))
    # select stations to search
    h5py_base_dir = search_hdf5_prefix
    search_list = get_search_station_list(sta, station_list, max_search_distance)
    seed_csv = detections_prefix + '{}_outputs/X_prediction_results.csv'.format(sta[1][3:])
    seed_csv_file = pd.read_csv(seed_csv)
    for e_time in seed_csv_file['file_name']:
        mask = (seed_csv_file['file_name'] == e_time)
        try:
            spt_t = obspy.UTCDateTime(seed_csv_file[mask]['p_arrival_time'].values[0]) - obspy.UTCDateTime(e_time[-27:])
            sst_t = obspy.UTCDateTime(seed_csv_file[mask]['s_arrival_time'].values[0]) - obspy.UTCDateTime(e_time[-27:])
            spt_prob = seed_csv_file[mask]['p_probability'].values[0]
            sst_prob = seed_csv_file[mask]['s_probability'].values[0]
        except:
            continue
        if sst_prob < 0.3:
            print('No GOOD SKIP')
            continue
        else:
            print('COOL Go ahead')
        spt = np.zeros([1,1])
        spt[0,0] = float(spt_t/60.0)
        sst = np.zeros([1,1])
        sst[0,0] = float(sst_t/60.0)
        coda_end = np.zeros([1,1])
        coda_end[0,0] = float(sst_t/60.0)
        print(e_time)
        ref_pick_time = obspy.UTCDateTime(e_time[-27:]) + spt[0,0] * 60 - base_time
        print('REF TIME:{}'.format(ref_pick_time))
        seed_h5file = h5py_base_dir + '{}.hdf5'.format(sta[1][3:])
        with h5py.File(seed_h5file,'r') as f:
            dataset = f.get('data/'+e_time)
            data_t = np.array(dataset)
            data_t -= np.mean(data_t, axis=0 ,keepdims=True)
            t_std = np.std(data_t, axis = 0, keepdims=True)
            t_std[t_std == 0] = 1.0
            data_t /= t_std
            data_t -= np.mean(data_t, axis=0 ,keepdims=True)
            data_t_in = np.zeros([1,6000,3])
            data_t_in[0,:,:] = data_t
            
        for search_sta in search_list:
            # check if pick exists 
            #print(search_sta)
            search_ref_picks = s_eqt_dict[search_sta[1]]['S'] 
            if len(search_ref_picks) > 1:
                ref_key_id = bisect.bisect_left(search_ref_picks,ref_pick_time)
                try:
                    ref_left = np.abs(search_ref_picks[ref_key_id-1] - ref_pick_time)
                    ref_middle = np.abs(search_ref_picks[ref_key_id] - ref_pick_time)
                    ref_right = np.abs(search_ref_picks[ref_key_id+1] - ref_pick_time)
                    if ref_left < 60.0 or ref_middle < 60.0 or ref_right < 60.0:
                        print('Skipping {}'.format(search_sta[1]))
                        continue
                except:
                    pass
            else:
                pass
            h5py_base_dir_search = search_hdf5_prefix
            search_csvfile = h5py_base_dir_search + '{}.csv'.format(search_sta[1][3:])
            search_csvfile = pd.read_csv(search_csvfile)            
            keys = list(search_csvfile['trace_name'])
            if len(keys) < 2:
                print('Keys Error')
                continue
            prefix = keys[0][:-27]
            keys = [key[-27:] for key in keys]
            key_id = bisect.bisect_left(keys,e_time[-27:])
            
            try:
                search_keys = [keys[key_id-2],keys[key_id-1],keys[key_id],keys[key_id+1],keys[key_id+2]]
            except:
                print('Search_keys Error')
                continue
            # calculate key search range

            search_h5file =  h5py_base_dir_search + '{}.hdf5'.format(search_sta[1][3:])
            t_update_list = list()
            t_update_list_prob = list()
            with h5py.File(search_h5file,'r') as f:
                max_pred_amp = 0
                for search_key in search_keys:

                    t_search_key = prefix + search_key
                    dataset = f.get('data/'+t_search_key)
                    data_s = np.array(dataset)
                    data_s -= np.mean(data_s, axis=0 ,keepdims=True)
                    t_std = np.std(data_s, axis = 0, keepdims=True)
                    t_std[t_std == 0] = 1.0
                    data_s /= t_std

                    data_s_in = np.zeros([1,6000,3])
                    data_s_in[0,:,:] = data_s

                    encoded_t = encode_model.predict(data_t_in)
                    encoded_s = encode_model.predict(data_s_in)

                    siamese_input_list = list()

                    for rdx in range(len(RSRN_lengths)):
                        temp_length = float(RSRN_lengths[rdx])
                        template_s = int(spt[0,0]*temp_length) - 1
                        template_e = int(coda_end[0,0]*temp_length) + 1
                        template_w = int(template_e - template_s)
                        encoded_t[rdx] = encoded_t[rdx][:,template_s:template_e,:]/float(template_w)
                        encoded_t[rdx] = encoded_t[rdx].reshape([1,template_w,1,int(RSRN_channels[rdx])])
                        encoded_s[rdx] = encoded_s[rdx].reshape([1,int(RSRN_lengths[rdx]),1,int(RSRN_channels[rdx])])

                        # channel normalization
                        for channel_dx in range(int(RSRN_channels[rdx])):
                            encoded_s[rdx][0,:,0,channel_dx] -= np.max(encoded_s[rdx][0,:,0,channel_dx])
                            half_window_len = int( 200.0*temp_length/6000.0   ) + 1
                            #window_mean = np.mean(encoded_s[rdx][0,half_window_len:-half_window_len,0,channel_dx])

                            encoded_s[rdx][0,:half_window_len,0,channel_dx] = encoded_s[rdx][0,half_window_len,0,channel_dx]
                            encoded_s[rdx][0,-half_window_len:,0,channel_dx] =  encoded_s[rdx][0,-half_window_len,0,channel_dx]

                            encoded_s[rdx][0,:,0,channel_dx] *= -1.0
                            encoded_s[rdx][0,:,0,channel_dx] -= np.mean(encoded_s[rdx][0,:,0,channel_dx])
                            t_max = np.max(np.abs(encoded_s[rdx][0,:,0,channel_dx]))
                            if t_max < 0.01:
                                t_max = 1
                            encoded_s[rdx][0,:,0,channel_dx] /= t_max

                            encoded_t[rdx][0,:,0,channel_dx] -= np.max(encoded_t[rdx][0,:,0,channel_dx])
                            encoded_t[rdx][0,:,0,channel_dx] *= -1.0
                            encoded_t[rdx][0,:,0,channel_dx] -= np.mean(encoded_t[rdx][0,:,0,channel_dx])
                            t_max = np.max(np.abs(encoded_t[rdx][0,:,0,channel_dx]))
                            if t_max < 0.01:
                                t_max = 1
                            encoded_t[rdx][0,:,0,channel_dx] /= t_max

                        siamese_input_list.append(encoded_t[rdx])
                        siamese_input_list.append(encoded_s[rdx])

                    #print('RSRN Channel_normal OK')
                    if if_encoder_concate == 1:
                        for rdx in range(len(RSRN_lengths), len(RSRN_lengths) + len(encoder_encoded_list)):
                            rdx_2 = rdx - len(RSRN_lengths) 
                            temp_length = float(encoder_encoded_lengths[rdx_2])
                            template_s = int(spt[0,0]*temp_length) - 1
                            template_e = int(coda_end[0,0]*temp_length) + 1
                            template_w = int(template_e - template_s)
                            #print('Concate 1 OK')
                            encoded_t[rdx] = encoded_t[rdx][:,template_s:template_e,:]/float(template_w)
                            encoded_t[rdx] = encoded_t[rdx].reshape([1,template_w,1,int(encoder_encoded_channels[rdx_2])])
                            encoded_s[rdx] = encoded_s[rdx].reshape([1,int(encoder_encoded_lengths[rdx_2]),1,int(encoder_encoded_channels[rdx_2])])
                            #print('Concate 2 OK')
                            # channel normalization
                            for channel_dx in range(int(encoder_encoded_channels[rdx_2])):
                                encoded_s[rdx][0,:,0,channel_dx] -= np.max(encoded_s[rdx][0,:,0,channel_dx])
                                half_window_len = int( 200.0*temp_length/6000.0   ) + 1
                                #window_mean = np.mean(encoded_s[rdx][0,half_window_len:-half_window_len,0,channel_dx])
                                #print('Concate 3 OK')
                                encoded_s[rdx][0,:half_window_len,0,channel_dx] = encoded_s[rdx][0,half_window_len,0,channel_dx]
                                encoded_s[rdx][0,-half_window_len:,0,channel_dx] =  encoded_s[rdx][0,-half_window_len,0,channel_dx]
                                #print('Concate 4 OK')
                                encoded_s[rdx][0,:,0,channel_dx] *= -1.0
                                encoded_s[rdx][0,:,0,channel_dx] -= np.mean(encoded_s[rdx][0,:,0,channel_dx])
                                t_max = np.max(np.abs(encoded_s[rdx][0,:,0,channel_dx]))
                                if t_max < 0.01:
                                    t_max = 1
                                encoded_s[rdx][0,:,0,channel_dx] /= t_max
                                #print('Concate 5 OK')
                                encoded_t[rdx][0,:,0,channel_dx] -= np.max(encoded_t[rdx][0,:,0,channel_dx])
                                encoded_t[rdx][0,:,0,channel_dx] *= -1.0
                                encoded_t[rdx][0,:,0,channel_dx] -= np.mean(encoded_t[rdx][0,:,0,channel_dx])
                                t_max = np.max(np.abs(encoded_t[rdx][0,:,0,channel_dx]))
                                if t_max < 0.01:
                                    t_max = 1
                                encoded_t[rdx][0,:,0,channel_dx] /= t_max
                                #print('Concate 6 OK')
                            siamese_input_list.append(encoded_t[rdx])
                            siamese_input_list.append(encoded_s[rdx])
                    pred_res = siamese_model.predict(siamese_input_list)
                    siamese_s = np.argmax(pred_res[-1][0,:,0,0])
                    pred_amp = pred_res[-1][0,siamese_s,0,0]
                    
                    if pred_amp > 0.1 and siamese_s > 800 and siamese_s < 5200:
                        siamese_s_time = obspy.UTCDateTime(search_key) + siamese_s * 0.01
                        t_update_time  = siamese_s_time - base_time
                        t_diff_time = siamese_s_time - (obspy.UTCDateTime(e_time[-27:]) + sst[0,0] * 60)
                        #r_snr = np.max(np.abs(data_s_in[0,siamese_s-50:siamese_s,0]))/np.max(np.abs(data_s_in[0,siamese_s:siamese_s+100,0]))
                        
                        if np.abs(t_diff_time) < 12:
                            pass
                        else:
                            #print(t_diff_time)
                            continue

                        max_pred_amp = pred_amp
                        s_pred_amp = pred_amp
                        siamese_s_time = obspy.UTCDateTime(search_key) + siamese_s * 0.01
                        t_update_time  = siamese_s_time - base_time
  
                        t_update_list.append(t_update_time)
                        t_update_list_prob.append(pred_amp)
                if len(t_update_list) > 0:
                    max_arg = np.argmax(t_update_list_prob)
                    t_update_time = t_update_list[max_arg]
                    bisect.insort(s_eqt_dict[search_sta[1]]['S'], t_update_time)
                    print('{}'.format(t_update_time))
    """
    plt.figure(figsize=(6,6))
    plt.scatter(sta[3],sta[2],color='r')
    for t_sta in search_list:
        plt.scatter(t_sta[3],t_sta[2],color='b')
    plt.xlim([-118.8,-118.0])
    plt.ylim([34.0,34.8])
    plt.show()
    plt.close()
    """
    # open search list csv and hdf5 file
    # for all high P
    #     for all search stations
    #        for all time within range
    #           calcualte P
    #           keep highest Prob
    #           check insert Point
    #           

    for sta_key in s_eqt_dict.keys():
    cur_file = '/mnt/BYEB/LA_JAN/LA_DATA/PostProcess/SEQTRes/20100101/{}.S.txt'.format(sta_key)
    f = open(cur_file,'w')
    arrival_dx = 0
    pre_arrival = -1
    for arrival in s_eqt_dict[sta_key]['S']:
        if arrival_dx == 0:
            pre_arrival = arrival
            arrival_dx += 1
        else:
            if np.abs(pre_arrival - arrival) < 1:
                continue
            else:
                pre_arrival = arrival
                
        S_res = '{:.3f} 0.80 0.00000000'.format(arrival)
        f.write(S_res+'\n')
    f.close()

    keep_t = None
for sta_key in s_eqt_dict.keys():
    S_EqT_S_list = list()
    print(sta_key)
    if len(s_eqt_dict[sta_key]['S']) > 0:
        pass
    else:
        continue
    s_times = s_eqt_dict[sta_key]['S']
    s_probs = s_eqt_dict[sta_key]['S_Prob']
    prev_file = '/mnt/BYEB/LA_JAN/LA_DATA/PostProcess/SEQTRes/20100101/{}.S.txt'.format(sta_key)
    if os.path.exists(prev_file):
        f = open(prev_file,'r')
        for line in f.readlines():
            if len(line) > 3:
                s_times.append(float(line.split(' ')[0]))
                s_probs.append(0.1)
        f.close()
    
for sta_key in s_eqt_dict.keys():
    s_times = s_eqt_dict[sta_key]['S']
    savefile = open('/mnt/BYEB/LA_JAN/LA_DATA/PostProcess/SEQTRes/20100101/{}.S.txt'.format(sta_key),'w')
    for s_t in s_times:
        savefile.write('{:.3f} 0.80000 0.00000000\n'.format(s_t))
    savefile.close()

    indexs = np.argsort(s_times)
    for idx in range(len(indexs)-1):
        t_id_0 =  indexs[idx]
        t_id_1 = indexs[idx+1]
        if s_times[t_id_1] - s_times[t_id_0] < 3.0:
            if s_probs[t_id_1] >= s_probs[t_id_0]:
                keep_t = s_times[t_id_1]
                keep_prob = s_probs[t_id_1]
            else:
                keep_t = s_times[t_id_0]
                keep_prob = s_probs[t_id_0]
        elif keep_t is None:
            S_EqT_S_list.append('{:.3f} {:.5f} 0.00000000'.format(s_times[t_id_0], s_probs[t_id_0]))
        elif idx == len(indexs)-2:
            S_EqT_S_list.append('{:.3f} {:.5f}  0.00000000'.format(s_times[t_id_0], s_probs[t_id_0]))
            S_EqT_S_list.append('{:.3f} {:.5f}  0.00000000'.format(s_times[t_id_1], s_probs[t_id_1]))
        else:
            S_EqT_S_list.append('{:.3f} {:.5f}  0.00000000'.format(keep_t, keep_prob))
            keep_t = None
    cur_file = '/root/Desktop/Public/SiameseEQTransformer/notebooks/EQT2REAL_PICK_SEQT/{}.P.txt'.format(sta_key)
    f = open(cur_file,'w')
    for S_res in S_EqT_S_list:
        f.write(S_res+'\n')
    f.close()
    #print(S_EqT_S_list)
    #break