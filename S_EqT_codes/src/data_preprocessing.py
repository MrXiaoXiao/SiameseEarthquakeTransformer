import numpy as np
import os

def build_phase_dict_from_EqT(cfgs, wavetype='P'):
    station_list = list()
    phase_dict = dict()
    station_list_file = open(cfgs['REAL']['save_sta'],'r')
    sta_id = 0

    for line in station_list_file.readlines():
        if len(line) < 3:
            continue
        splits = line.split(' ')
        sta_name = splits[2]+'.'+splits[3]
        
        phase_dict[sta_name] = dict()
        phase_dict[sta_name]['P'] = list()
        phase_dict[sta_name]['S'] = list()
        phase_dict[sta_name]['P_Prob'] = list()
        phase_dict[sta_name]['S_Prob'] = list()
        
        sta_lat = float(splits[1])
        sta_lon = float(splits[0])
        
        station_list.append( (sta_id, sta_name, sta_lat, sta_lon) )
        sta_id += 1
    
    for sta_key in phase_dict.keys():
        pick_times = list()
        pick_probs = list()
        
        prev_file = cfgs['EqT']['txt_folder'] + '{}.{}.txt'.format(sta_key,wavetype)
        if os.path.exists(prev_file):
            f = open(prev_file,'r')
            for line in f.readlines():
                if len(line) > 3:
                    pick_times.append(float(line.split(' ')[0]))
                    pick_probs.append(float(line.split(' ')[1]))
            f.close()
        else:
            print('Empty' + prev_file)

        phase_dict[sta_key]['{}'.format(wavetype)] = pick_times
        phase_dict[sta_key]['{}_Prob'.format(wavetype)] = pick_probs
    return phase_dict, station_list

def normalize_by_std(data_in):
    """
    std normalization
    """
    data_in -= np.mean(data_in, axis=0 ,keepdims=True)
    t_std = np.std(data_in, axis = 0, keepdims=True)
    t_std[t_std == 0] = 1.0
    data_in /= t_std

    return data_in

def get_response_list_for_vis(cfgs, spt_t_eqt, sst_t_eqt, encoded_t, encoded_s):

    RSRN_lengths = cfgs['Model']['RSRN_Encoded_lengths']
    RSRN_channels = cfgs['Model']['RSRN_Encoded_channels']
    #encoder_encoded_list = cfgs['Model']['Encoder_concate_list']
    #encoder_encoded_lengths = cfgs['Model']['Encoder_concate_lengths']
    #encoder_encoded_channels = cfgs['Model']['Encoder_concate_channels']

    ori_response_list_for_vis = list()
    enhanced_response_list_for_vis = list()

    t_spt_t = float(spt_t_eqt/6000.0)
    t_sst_t = float(sst_t_eqt/6000.0)

    for rdx in range(len(RSRN_lengths)):
        temp_length = float(RSRN_lengths[rdx])
        template_s = int(t_spt_t*temp_length) - 1
        template_e = int(t_sst_t*temp_length) + 1
        template_w = int(template_e - template_s)

        encoded_t[rdx] = encoded_t[rdx][:,template_s:template_e,:]/float(template_w)
        encoded_t[rdx] = encoded_t[rdx].reshape([1,template_w,1,int(RSRN_channels[rdx])])
        encoded_s[rdx] = encoded_s[rdx].reshape([1,int(RSRN_lengths[rdx]),1,int(RSRN_channels[rdx])])
        ori_response_list_for_vis.append(np.copy(encoded_t[rdx]))
        ori_response_list_for_vis.append(np.copy(encoded_s[rdx]))
        
        # channel-wise normalization
        for channel_dx in range(int(RSRN_channels[rdx])):
            encoded_s[rdx][0,:,0,channel_dx] -= np.max(encoded_s[rdx][0,:,0,channel_dx])
            half_window_len = int( 200.0*temp_length/6000.0   ) + 1
            
            encoded_s[rdx][0,:half_window_len,0,channel_dx] = encoded_s[rdx][0,half_window_len,0,channel_dx]
            encoded_s[rdx][0,-half_window_len:,0,channel_dx] =  encoded_s[rdx][0,-half_window_len,0,channel_dx]

            encoded_s[rdx][0,:,0,channel_dx] *= -1.0
            encoded_s[rdx][0,:,0,channel_dx] -= np.mean(encoded_s[rdx][0,:,0,channel_dx])
            t_max = np.max(np.abs(encoded_s[rdx][0,:,0,channel_dx]))
            if t_max < 0.001:
                t_max = 1
            encoded_s[rdx][0,:,0,channel_dx] /= t_max

            encoded_t[rdx][0,:,0,channel_dx] -= np.max(encoded_t[rdx][0,:,0,channel_dx])
            encoded_t[rdx][0,:,0,channel_dx] *= -1.0
            encoded_t[rdx][0,:,0,channel_dx] -= np.mean(encoded_t[rdx][0,:,0,channel_dx])
            t_max = np.max(np.abs(encoded_t[rdx][0,:,0,channel_dx]))
            if t_max < 0.001:
                t_max = 1
            encoded_t[rdx][0,:,0,channel_dx] /= t_max
            encoded_t[rdx][0,:,0,channel_dx] /= float(template_w)
        enhanced_response_list_for_vis.append(encoded_t[rdx])
        enhanced_response_list_for_vis.append(encoded_s[rdx])

    return ori_response_list_for_vis, enhanced_response_list_for_vis

def get_siamese_input_list(cfgs, spt_t_eqt, sst_t_eqt, encoded_t, encoded_s):

    RSRN_lengths = cfgs['Model']['RSRN_Encoded_lengths']
    RSRN_channels = cfgs['Model']['RSRN_Encoded_channels']
    encoder_encoded_list = cfgs['Model']['Encoder_concate_list']
    encoder_encoded_lengths = cfgs['Model']['Encoder_concate_lengths']
    encoder_encoded_channels = cfgs['Model']['Encoder_concate_channels']

    siamese_input_list = list()
    t_spt_t = float(spt_t_eqt/6000.0)
    t_sst_t = float(sst_t_eqt/6000.0)

    for rdx in range(len(RSRN_lengths)):
        temp_length = float(RSRN_lengths[rdx])
        template_s = int(t_spt_t*temp_length) - 1
        template_e = int(t_sst_t*temp_length) + 1
        template_w = int(template_e - template_s)

        encoded_t[rdx] = encoded_t[rdx][:,template_s:template_e,:]/float(template_w)
        encoded_t[rdx] = encoded_t[rdx].reshape([1,template_w,1,int(RSRN_channels[rdx])])
        encoded_s[rdx] = encoded_s[rdx].reshape([1,int(RSRN_lengths[rdx]),1,int(RSRN_channels[rdx])])

        # channel-wise normalization
        for channel_dx in range(int(RSRN_channels[rdx])):
            encoded_s[rdx][0,:,0,channel_dx] -= np.max(encoded_s[rdx][0,:,0,channel_dx])
            half_window_len = int( 200.0*temp_length/6000.0   ) + 1
            
            encoded_s[rdx][0,:half_window_len,0,channel_dx] = encoded_s[rdx][0,half_window_len,0,channel_dx]
            encoded_s[rdx][0,-half_window_len:,0,channel_dx] =  encoded_s[rdx][0,-half_window_len,0,channel_dx]

            encoded_s[rdx][0,:,0,channel_dx] *= -1.0
            encoded_s[rdx][0,:,0,channel_dx] -= np.mean(encoded_s[rdx][0,:,0,channel_dx])
            t_max = np.max(np.abs(encoded_s[rdx][0,:,0,channel_dx]))
            if t_max < 0.001:
                t_max = 1
            encoded_s[rdx][0,:,0,channel_dx] /= t_max

            encoded_t[rdx][0,:,0,channel_dx] -= np.max(encoded_t[rdx][0,:,0,channel_dx])
            encoded_t[rdx][0,:,0,channel_dx] *= -1.0
            encoded_t[rdx][0,:,0,channel_dx] -= np.mean(encoded_t[rdx][0,:,0,channel_dx])
            t_max = np.max(np.abs(encoded_t[rdx][0,:,0,channel_dx]))
            if t_max < 0.001:
                t_max = 1
            encoded_t[rdx][0,:,0,channel_dx] /= t_max
            encoded_t[rdx][0,:,0,channel_dx] /= float(template_w)
        siamese_input_list.append(encoded_t[rdx])
        siamese_input_list.append(encoded_s[rdx])

    for rdx in range(len(RSRN_lengths), len(RSRN_lengths) + len(encoder_encoded_list)):
        rdx_2 = rdx - len(RSRN_lengths) 
        temp_length = float(encoder_encoded_lengths[rdx_2])
        template_s = int(t_spt_t*temp_length) - 1
        template_e = int(t_sst_t*temp_length) + 1
        template_w = int(template_e - template_s)

        encoded_t[rdx] = encoded_t[rdx][:,template_s:template_e,:]/float(template_w)
        encoded_t[rdx] = encoded_t[rdx].reshape([1,template_w,1,int(encoder_encoded_channels[rdx_2])])
        encoded_s[rdx] = encoded_s[rdx].reshape([1,int(encoder_encoded_lengths[rdx_2]),1,int(encoder_encoded_channels[rdx_2])])

        # channel normalization
        for channel_dx in range(int(encoder_encoded_channels[rdx_2])):
            encoded_s[rdx][0,:,0,channel_dx] -= np.max(encoded_s[rdx][0,:,0,channel_dx])
            half_window_len = int( 200.0*temp_length/6000.0   ) + 1

            encoded_s[rdx][0,:half_window_len,0,channel_dx] = encoded_s[rdx][0,half_window_len,0,channel_dx]
            encoded_s[rdx][0,-half_window_len:,0,channel_dx] =  encoded_s[rdx][0,-half_window_len,0,channel_dx]

            encoded_s[rdx][0,:,0,channel_dx] *= -1.0
            encoded_s[rdx][0,:,0,channel_dx] -= np.mean(encoded_s[rdx][0,:,0,channel_dx])
            t_max = np.max(np.abs(encoded_s[rdx][0,:,0,channel_dx]))
            if t_max < 0.001:
                t_max = 1
            encoded_s[rdx][0,:,0,channel_dx] /= t_max

            encoded_t[rdx][0,:,0,channel_dx] -= np.max(encoded_t[rdx][0,:,0,channel_dx])
            encoded_t[rdx][0,:,0,channel_dx] *= -1.0
            encoded_t[rdx][0,:,0,channel_dx] -= np.mean(encoded_t[rdx][0,:,0,channel_dx])
            t_max = np.max(np.abs(encoded_t[rdx][0,:,0,channel_dx]))
            if t_max < 0.001:
                t_max = 1
            encoded_t[rdx][0,:,0,channel_dx] /= t_max
            encoded_t[rdx][0,:,0,channel_dx] /= float(template_w)

        siamese_input_list.append(encoded_t[rdx])
        siamese_input_list.append(encoded_s[rdx])
    
    return siamese_input_list