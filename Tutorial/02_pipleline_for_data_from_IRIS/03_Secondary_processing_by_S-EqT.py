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
sys.path.append('../../S_EqT_codes/src')
sys.path.append('../../S_EqT_codes/src/EqT_libs')
from S_EqT_concate_fix_corr import S_EqT_Concate_RSRN_Model
from misc import get_train_list, get_search_station_list, get_closest_value
from data_preprocessing import build_phase_dict_from_EqT
import keras.backend as K
import keras
keras.backend.set_floatx('float32')
import yaml
from random import shuffle
import os
import argparse

# Simplified steps:
# 01 load config file
# 02 search all picks (aggressive mode or lasy mode)
# 03 save results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='03_run_S-EqT')
    parser.add_argument('--config-file', dest='config_file', 
                        type=str, help='Configuration file path',default='./default_pipline_config.yaml')
    args = parser.parse_args()
    cfgs = yaml.load(open(args.config_file,'r'),Loader=yaml.SafeLoader)
    
    # build phase dict using EqT results
    phase_dict, station_list = build_phase_dict_from_EqT(cfgs)
    
    # load s-eqt models -- P branch
    P_branch_cfgs = yaml.load(open(cfgs['S_EqT']['P_branch_config'],'r'),Loader=yaml.SafeLoader)
    encode_model, siamese_model, EqT_model = S_EqT_Concate_RSRN_Model(P_branch_cfgs)
    siamese_model.load_weights(cfgs['S_EqT']['P_branch_model'])
    RSRN_lengths = P_branch_cfgs['Model']['RSRN_Encoded_lengths']
    RSRN_channels = P_branch_cfgs['Model']['RSRN_Encoded_channels']
    encoder_encoded_list = P_branch_cfgs['Model']['Encoder_concate_list']
    encoder_encoded_lengths = P_branch_cfgs['Model']['Encoder_concate_lengths']
    encoder_encoded_channels = P_branch_cfgs['Model']['Encoder_concate_channels']
    sta_search_dx = 0

    search_hdf5_prefix = cfgs['EqT']['mseed_dir'] + '_processed_hdfs/'
    detections_prefix = cfgs['EqT']['det_res'] + '/'
    base_time = obspy.UTCDateTime(cfgs['REAL']['ref_time'])
    for sta in station_list:
        sta_search_dx += 1
        print('On {} {} of {} -- P_branch'.format(sta_search_dx,sta, len(station_list)))
        # select stations to search
        h5py_base_dir = search_hdf5_prefix
        search_list = get_search_station_list(sta, station_list, cfgs['S_EqT']['max_search_distance'])
        seed_csv = detections_prefix + '{}_outputs/X_prediction_results.csv'.format(sta[1][3:],sta[1][3:])
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
            if spt_prob < cfgs['S_EqT']['P_skip_threshold']:
                continue
            spt = np.zeros([1,1])
            spt[0,0] = float(spt_t/60.0)
            sst = np.zeros([1,1])
            sst[0,0] = float(sst_t/60.0)
            coda_end = np.zeros([1,1])
            coda_end[0,0] = float(sst_t/60.0)
            print(e_time)
            ref_pick_time = obspy.UTCDateTime(e_time[-27:]) + spt[0,0] * 60 - base_time
            print('REF TIME:{} {}'.format(ref_pick_time,sta))
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
                search_ref_picks = phase_dict[search_sta[1]]['P'] 
                if len(search_ref_picks) > 0:
                    closest_t = get_closest_value(search_ref_picks, ref_pick_time)
                    if np.abs(closest_t - ref_pick_time) < cfgs['S_EqT']['exist_range']:
                        continue                  
                else:
                    pass
                h5py_base_dir_search = search_hdf5_prefix
                search_csvfile = h5py_base_dir_search + '{}.csv'.format(search_sta[1][3:])
                search_csvfile = pd.read_csv(search_csvfile)
                keys = list(search_csvfile['trace_name'])
                if len(keys) < 2:
                    continue
                prefix = keys[0][:-27]
                keys = [key[-27:] for key in keys]
                key_id = bisect.bisect_left(keys,e_time[-27:])
                
                try:
                    if obspy.UTCDateTime(keys[key_id]) - obspy.UTCDateTime(e_time[-27:]) < 10:
                        search_keys = [keys[key_id-1],keys[key_id],keys[key_id+1]]
                    else:
                        search_keys = [keys[key_id-1],keys[key_id],keys[key_id-2]]
                except:
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
                        
                        if pred_amp > cfgs['S_EqT']['P_threshold'] and siamese_s > 200 and siamese_s < 5800:
                            siamese_s_time = obspy.UTCDateTime(search_key) + siamese_s * 0.01
                            t_update_time  = siamese_s_time - base_time
                            t_diff_time = siamese_s_time - (obspy.UTCDateTime(e_time[-27:]) + spt[0,0] * 60)
                            
                            if np.abs(t_diff_time) < 6:
                                pass
                            else:
                                continue
                            
                            t_update_list.append(t_update_time)
                            t_update_list_prob.append(pred_amp)
                            
                    if len(t_update_list) > 0:
                        max_arg = np.argmax(t_update_list_prob)
                        t_update_time = t_update_list[max_arg]
                        bisect.insort(phase_dict[search_sta[1]]['P'], t_update_time)
                        print('Retrieved time: {} {}'.format(t_update_time, search_sta))
    
    if os.path.exists(cfgs['S_EqT']['txt_folder']):
        pass
    else:
        os.makedirs(cfgs['S_EqT']['txt_folder'])

    for sta_key in phase_dict.keys():
        cur_file = cfgs['S_EqT']['txt_folder'] + '{}.P.txt'.format(sta_key)
        f = open(cur_file,'w')
        arrival_dx = 0
        for arrival in phase_dict[sta_key]['P']:
            """
            if arrival_dx == 0:
                pre_arrival = arrival
                arrival_dx += 1
            else:
                if np.abs(pre_arrival - arrival) < cfgs['S_EqT']['keep_time_range_P']:
                    continue
                else:
                    pre_arrival = arrival
            """
            S_res = '{:.3f} 1.00 0.00000000'.format(arrival)
            f.write(S_res+'\n')
        f.close()
    
    # load s-eqt models -- S branch
    K.clear_session()
    # build phase dict using EqT results
    phase_dict, station_list = build_phase_dict_from_EqT(cfgs,'S')
    
    # load s-eqt models -- S branch
    S_branch_cfgs = yaml.load(open(cfgs['S_EqT']['S_branch_config'],'r'),Loader=yaml.SafeLoader)
    encode_model, siamese_model, EqT_model = S_EqT_Concate_RSRN_Model(S_branch_cfgs)
    siamese_model.load_weights(cfgs['S_EqT']['S_branch_model'])
    RSRN_lengths = S_branch_cfgs['Model']['RSRN_Encoded_lengths']
    RSRN_channels = S_branch_cfgs['Model']['RSRN_Encoded_channels']
    encoder_encoded_list = S_branch_cfgs['Model']['Encoder_concate_list']
    encoder_encoded_lengths = S_branch_cfgs['Model']['Encoder_concate_lengths']
    encoder_encoded_channels = S_branch_cfgs['Model']['Encoder_concate_channels']
    sta_search_dx = 0
    
    search_hdf5_prefix = cfgs['EqT']['mseed_dir'] + '_processed_hdfs/'
    detections_prefix = cfgs['EqT']['det_res'] + '/'
    base_time = obspy.UTCDateTime(cfgs['REAL']['ref_time'])

    sta_search_dx = 0
    for sta in station_list:
        sta_search_dx += 1
        print('On {} {} of {} -- S_branch'.format(sta_search_dx,sta, len(station_list)))
        # select stations to search
        h5py_base_dir = search_hdf5_prefix
        search_list = get_search_station_list(sta, station_list)
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
            if sst_prob < cfgs['S_EqT']['S_skip_threshold']:
                #print('No GOOD SKIP')
                continue
            else:
                #print('COOL Go ahead')
                pass

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
                search_ref_picks = phase_dict[search_sta[1]]['S'] 
                
                if len(search_ref_picks) > 0:
                    ref_key_id = bisect.bisect_left(search_ref_picks,ref_pick_time)
                    closest_t = get_closest_value(search_ref_picks, ref_pick_time)
                    if np.abs(closest_t - ref_pick_time) < cfgs['S_EqT']['exist_range']:
                        print('Skipping {} {} {} {}'.format(closest_t, search_sta, ref_pick_time, sta))
                        continue
                else:
                    pass
                
                h5py_base_dir_search = search_hdf5_prefix
                search_csvfile = h5py_base_dir_search + '{}.csv'.format(search_sta[1][3:])
                search_csvfile = pd.read_csv(search_csvfile)            
                keys = list(search_csvfile['trace_name'])
                if len(keys) < 2:
                    #print('Keys Error')
                    continue
                prefix = keys[0][:-27]
                keys = [key[-27:] for key in keys]
                key_id = bisect.bisect_left(keys,e_time[-27:])
                
                try:
                    search_keys = [keys[key_id-2],keys[key_id-1],keys[key_id],keys[key_id+1],keys[key_id+2]]
                except:
                    #print('Search_keys Error')
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
                        
                        if pred_amp > cfgs['S_EqT']['S_threshold'] and siamese_s > 800 and siamese_s < 5200:
                            siamese_s_time = obspy.UTCDateTime(search_key) + siamese_s * 0.01
                            t_update_time  = siamese_s_time - base_time
                            t_diff_time = siamese_s_time - (obspy.UTCDateTime(e_time[-27:]) + sst[0,0] * 60)
                            #r_snr = np.max(np.abs(data_s_in[0,siamese_s-50:siamese_s,0]))/np.max(np.abs(data_s_in[0,siamese_s:siamese_s+100,0]))
                            
                            if np.abs(t_diff_time) < cfgs['S_EqT']['keep_time_range_S']:
                                pass
                            else:
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
                        bisect.insort(phase_dict[search_sta[1]]['S'], t_update_time)
                        print('Retrieved time: {} {}'.format(t_update_time, search_sta))
    
    for sta_key in phase_dict.keys():
        cur_file = cfgs['S_EqT']['txt_folder'] + '{}.S.txt'.format(sta_key)
        f = open(cur_file,'w')
        arrival_dx = 0
        for arrival in phase_dict[sta_key]['S']:
            """
            if arrival_dx == 0:
                pre_arrival = arrival
                arrival_dx += 1
            else:
                if np.abs(pre_arrival - arrival) < cfgs['S_EqT']['keep_time_range_S']:
                    continue
                else:
                    pre_arrival = arrival
            """        
            S_res = '{:.3f} 1.00 0.00000000'.format(arrival)
            
            f.write(S_res+'\n')
        f.close()
