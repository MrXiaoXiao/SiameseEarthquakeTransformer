import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from random import shuffle
from obspy.core import UTCDateTime
import obspy
import os
from pathlib import Path
from pyproj import Geod

def get_search_station_list(station, station_list, max_distance=999999):
    GEOD = Geod(ellps='WGS84')
    # Rough meters/degree calculation
    M_PER_DEG = (GEOD.inv(0, 0, 0, 1)[2] + GEOD.inv(0, 0, 1, 0)[2]) / 2
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

def xml2REAL_sta(cfgs):
    sta_inv_folder = Path(cfgs['EqT']['STAXML'])
    save_f = open(cfgs['REAL']['save_sta'],'w')
    for sta_folder in sta_inv_folder.glob('*'):
        for sta_file in sta_folder.glob('*.xml'):
            sta_inv = obspy.read_inventory(str(sta_file))
            for net in sta_inv.networks:
                for sta in net.stations:
                    save_f.write('{:.4f} {:.4f} {:} {:} {:} {:.3f}\n'.format(
                                sta.longitude,sta.latitude,net.code,sta.code,'BHZ',sta.elevation/1000.0))

    save_f.close()

def convert2sec(t, t_ref):
    """
    convert UTCDatetime object to seconds
    Params:
    t       UTCDateTime     Time to be converted
    t_ref   UTCDateTime     Reference time
    """
    t_utc = UTCDateTime(t)
    t_ref_utc = UTCDateTime(t_ref)
    return t_utc - t_ref_utc

def convert_csv_to_real(picks_csv_path,cfgs):
    """
    conver 
    """
    file_path = picks_csv_path
    data = pd.read_csv(file_path)
    data = data.fillna(-999)
    nrows, ncols = data.shape
    if (nrows <= 0):
        return 0
    # get reference time
    reftime = cfgs['REAL']['ref_time']
    # obtain the name of net and station
    temp = data.loc[0, 'file_name'].split("_")
    nnet = temp[1]
    nsta = temp[0];save_file_name = nnet + '.' + nsta
    # create the format for the output.
    fmt = ''
    # create the list to save the results.
    itp_arr = []
    tp_prob_arr = []
    its_arr = []
    ts_prob_arr = []
    for index in range(0, nrows):
        # the time for P/S (s).
        itp = data.loc[index, 'p_arrival_time']
        if itp != -999:
            tp_prob = float(data.loc[index, 'p_probability'])
            tp_prob_arr.append(tp_prob)
            pick_time_P = convert2sec(itp,reftime)
            itp_arr.append(pick_time_P)
        its = data.loc[index, 's_arrival_time']
        if its != -999:
            ts_prob = float(data.loc[index, 's_probability'])
            ts_prob_arr.append(ts_prob)
            pick_time_S = convert2sec(its,reftime)
            its_arr.append(pick_time_S)
    # handle and save the results.
    itp_prob_arr = np.array([itp_arr, tp_prob_arr, np.zeros_like(itp_arr)])
    itp_prob_arr = itp_prob_arr.T
    itp_prob_arr = itp_prob_arr[np.argsort(itp_prob_arr[:,0])]
    its_prob_arr = np.array([its_arr, ts_prob_arr, np.zeros_like(its_arr)]) 
    its_prob_arr = its_prob_arr.T
    its_prob_arr = its_prob_arr[np.argsort(its_prob_arr[:,0])]
    length_P = itp_prob_arr.shape[0]
    length_S = its_prob_arr.shape[0]
    for i in range(length_P-1, 0, -1):
        diff = itp_prob_arr[i][0] - itp_prob_arr[i-1][0]
        if abs(diff) < 1:
            if (itp_prob_arr[i][1] < itp_prob_arr[i-1][1]):
                itp_prob_arr = np.delete(itp_prob_arr, i, axis=0)
            else:
                itp_prob_arr = np.delete(itp_prob_arr, i-1, axis=0)
    for i in range(length_S-1, 0, -1):
        diff = its_prob_arr[i][0] - its_prob_arr[i-1][0]
        if abs(diff) < 1:
            if (its_prob_arr[i][1] < its_prob_arr[i-1][1]):
                its_prob_arr = np.delete(its_prob_arr, i, axis=0)
            else:
                its_prob_arr = np.delete(its_prob_arr, i-1, axis=0)
    save_prefix = cfgs['EqT']['txt_folder']
    if os.path.exists(save_prefix):
        pass
    else:
        os.makedirs(save_prefix)
    np.savetxt(save_prefix+"%s.P.txt"%(save_file_name), itp_prob_arr, fmt='%.3f %.5f %.8f')
    np.savetxt(save_prefix+"%s.S.txt"%(save_file_name), its_prob_arr, fmt='%.3f %.5f %.8f')
    return 1 

def plot_P_branch_responses(cfgs):
    plot_t = np.arange(0,30.01,0.01)
    plt.figure(figsize=(16,12))
    ax1 = plt.subplot2grid((6,6),(0,4),colspan=2,rowspan=2)
    plt.title('Simplified EqT diagram',fontsize=14)
    img = mpimg.imread('EqT_Fig5_use.jpg')
    plt.imshow(img,aspect='auto')
    plt.xticks([])
    plt.yticks([])
    ax1.axis('off')

    ax_t1 = plt.subplot2grid((6,6),(0,0),rowspan=2,colspan=2)
    for idx in range(3):
        plt.plot(plot_t,data_t[0:3001,idx]/np.max(np.abs(data_t[0:3001,idx]))+idx*2 + 2,color='k')
        if idx == '0':
            plt.text(28, idx*2 + 2 + 0.1, 'E')
        if idx == '1':
            plt.text(28, idx*2 + 2 + 0.1, 'N')
        if idx == '2':
            plt.text(28, idx*2 + 2 + 0.1, 'Z')

    plt.plot([spt_t_eqt/100.0,spt_t_eqt/100.0],[1,8],color='b',label='EqT P')
    plt.plot([sst_t_eqt/100.0,sst_t_eqt/100.0],[1,8],color='r',label='EqT S')
    plt.title('Template seismogram from station CI.DEC\nStart Time: 2020-08-08T16:46:00',fontsize=14)
    plt.ylim([1,7.5])
    plt.legend(loc='upper right',prop= {'size':14})
    plt.yticks([])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim([0,30])
    plt.xlabel('time (s)',fontsize=14)

    ax_t2 = plt.subplot2grid((6,6),(2,0),colspan=2)
    rdx = 29
    t_len = int(len(encoded_t_plot[rdx][0,:,0])/2.0)
    for channel_dx in range(int(RSRN_channels[rdx])):
        plt.plot(encoded_t_plot[rdx][0,:t_len,channel_dx] + 0 * 2,color='k')
    plt.xlim([0,t_len])
    plt.title('OTR conv1d_35',fontsize=14)
    plt.plot([spt_t_eqt/4.0,spt_t_eqt/4.0],[-50,50],color='b',label='EqT P')
    plt.plot([sst_t_eqt/4.0,sst_t_eqt/4.0],[-50,50],color='r',label='EqT S')
    t_min_plot = np.min(encoded_t_plot[rdx])
    t_max_plot = np.max(encoded_t_plot[rdx])
    t_gain = (t_max_plot - t_min_plot)*0.15
    plt.ylim([t_min_plot - t_gain,t_max_plot + t_gain])
    t_ax=plt.gca();t_ax.spines['right'].set_color('b');t_ax.spines['top'].set_color('b');t_ax.spines['bottom'].set_color('b');t_ax.spines['left'].set_color('b');t_ax.spines['right'].set_linewidth(3);t_ax.spines['top'].set_linewidth(3);t_ax.spines['bottom'].set_linewidth(3);t_ax.spines['left'].set_linewidth(3)
    plt.text(-100,t_min_plot+t_gain*0.15,'Amplitude',fontsize=14,rotation=90)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax_t3 = plt.subplot2grid((6,6),(3,0),colspan=2)
    rdx = 29
    t_len = int(len(encoded_t_plot[rdx][0,:,0])/2.0)
    start_dx = int(spt_t_eqt*int(RSRN_lengths[rdx])/6000.0)
    end_dx = start_dx + len(encoded_t[rdx][0,:,0,0])
    temp_x_plot = np.arange(start_dx,end_dx,1)
    for channel_dx in range(int(RSRN_channels[rdx])):
        plt.plot(temp_x_plot, encoded_t[rdx][0,:,0,channel_dx] + 0 * 2,color='k')
    plt.xlim([0,t_len])
    plt.title('ETR conv1d_35',fontsize=14)
    plt.plot([spt_t_eqt/4.0,spt_t_eqt/4.0],[-50,50],color='b',label='EqT P')
    plt.plot([sst_t_eqt/4.0,sst_t_eqt/4.0],[-50,50],color='r',label='EqT S')
    t_min_plot = np.min(encoded_t[rdx])
    t_max_plot = np.max(encoded_t[rdx])
    t_gain = (t_max_plot - t_min_plot)*0.15
    plt.ylim([t_min_plot - t_gain,t_max_plot + t_gain])
    t_ax=plt.gca();t_ax.spines['right'].set_color('b');t_ax.spines['top'].set_color('b');t_ax.spines['bottom'].set_color('b');t_ax.spines['left'].set_color('b');t_ax.spines['right'].set_linewidth(3);t_ax.spines['top'].set_linewidth(3);t_ax.spines['bottom'].set_linewidth(3);t_ax.spines['left'].set_linewidth(3)
    plt.text(-140,t_min_plot+t_gain*0.15,'Amplitude',fontsize=14,rotation=90)
    plt.xlabel('Array Index (N)',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax_t4 = plt.subplot2grid((6,6),(2,4),colspan=2)
    rdx = 29
    t_len = int(len(encoded_t_plot[rdx][0,:,0])/2.0)
    for channel_dx in range(int(RSRN_channels[rdx])):
        plt.plot(no_normal_corr[rdx][0,:t_len,0,channel_dx]/57.0 + 0 * 2,color='k')
    plt.xlim([0,t_len])
    plt.title('OCC conv1d_35',fontsize=14)
    t_min_plot = np.min(no_normal_corr[rdx])/57.0
    t_max_plot = np.max(no_normal_corr[rdx])/57.0
    t_gain = (t_max_plot - t_min_plot)*0.15
    plt.ylim([t_min_plot - t_gain,t_max_plot + t_gain])
    plt.plot([SEqT_P/4.0,SEqT_P/4.0],[-50,50],color='b',linestyle='--',label='S-EqT P')
    #plt.plot([SEqT_S/4.0,SEqT_S/4.0],[-50,50],color='r',linestyle='--',label='S-EqT S')
    t_ax=plt.gca();t_ax.spines['right'].set_color('b');t_ax.spines['top'].set_color('b');t_ax.spines['bottom'].set_color('b');t_ax.spines['left'].set_color('b');t_ax.spines['right'].set_linewidth(3);t_ax.spines['top'].set_linewidth(3);t_ax.spines['bottom'].set_linewidth(3);t_ax.spines['left'].set_linewidth(3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax_s1 = plt.subplot2grid((6,6),(0,2),rowspan=2,colspan=2)
    for idx in range(3):
        plt.plot(plot_t,data_s[0:3001,idx]/np.max(np.abs(data_s[0:3001,idx]))+idx*2 + 2,color='k')
        if idx == '0':
            plt.text(28, idx*2 + 2 + 0.1, 'E')
        if idx == '1':
            plt.text(28, idx*2 + 2 + 0.1, 'N')
        if idx == '2':
            plt.text(28, idx*2 + 2 + 0.1, 'Z')

    plt.yticks([])
    plt.plot([SEqT_P/100.0,SEqT_P/100.0],[1,8],color='b',linestyle='--',label='S-EqT P')
    #plt.plot([SEqT_S/100.0,SEqT_S/100.0],[1,8],color='r',linestyle='--',label='S-EqT S')
    plt.xlim([0,30])
    plt.ylim([1,7.5])
    plt.legend(loc='upper right',prop= {'size':14})
    #plt.text(0,7.5,'(c)',fontsize=20)
    plt.title('Searching seismogram from station CI.RIN\nStart Time: 2020-08-08T16:46:00',fontsize=14)
    plt.xlabel('time (s)',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax_s2 = plt.subplot2grid((6,6),(2,2),colspan=2)
    rdx = 29
    t_len = int(len(siamese_input_list_no_normal[rdx*2+1][0,:,0,0])/2.0)
    for channel_dx in range(int(RSRN_channels[rdx])):
        plt.plot(siamese_input_list_no_normal[rdx*2+1][0,:t_len,0,channel_dx] + 0 * 2,color='k')    
    plt.xlim([0,t_len])
    plt.title('OSR conv1d_35',fontsize=14)
    t_min_plot = np.min(siamese_input_list_no_normal[rdx*2+1])
    t_max_plot = np.max(siamese_input_list_no_normal[rdx*2+1])
    t_gain = (t_max_plot - t_min_plot)*0.15
    plt.ylim([t_min_plot - t_gain,t_max_plot + t_gain])
    plt.plot([SEqT_P/4.0,SEqT_P/4.0],[-50,50],color='b',linestyle='--',label='S-EqT P')
    #plt.plot([SEqT_S/4.0,SEqT_S/4.0],[-50,50],color='r',linestyle='--',label='S-EqT S')
    t_ax=plt.gca();t_ax.spines['right'].set_color('b');t_ax.spines['top'].set_color('b');t_ax.spines['bottom'].set_color('b');t_ax.spines['left'].set_color('b');t_ax.spines['right'].set_linewidth(3);t_ax.spines['top'].set_linewidth(3);t_ax.spines['bottom'].set_linewidth(3);t_ax.spines['left'].set_linewidth(3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax_s3 = plt.subplot2grid((6,6),(3,2),colspan=2)
    rdx = 29
    t_len = int(len(encoded_s[rdx][0,:,0,0])/2.0)
    for channel_dx in range(int(RSRN_channels[rdx])):
        plt.plot(encoded_s[rdx][0,:t_len,0,channel_dx] + 0 * 2,color='k')
    plt.xlim([0,t_len])
    plt.title('ESR conv1d_35',fontsize=14)
    t_min_plot = np.min(encoded_s[rdx])
    t_max_plot = np.max(encoded_s[rdx])
    t_gain = (t_max_plot - t_min_plot)*0.15
    plt.ylim([t_min_plot - t_gain,t_max_plot + t_gain])
    plt.plot([SEqT_P/4.0,SEqT_P/4.0],[-50,50],color='b',linestyle='--',label='S-EqT P')
    #plt.plot([SEqT_S/4.0,SEqT_S/4.0],[-50,50],color='r',linestyle='--',label='S-EqT S')
    t_ax=plt.gca();t_ax.spines['right'].set_color('b');t_ax.spines['top'].set_color('b');t_ax.spines['bottom'].set_color('b');t_ax.spines['left'].set_color('b');t_ax.spines['right'].set_linewidth(3);t_ax.spines['top'].set_linewidth(3);t_ax.spines['bottom'].set_linewidth(3);t_ax.spines['left'].set_linewidth(3)
    plt.xlabel('Array Index (N)',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax_s4 = plt.subplot2grid((6,6),(3,4),colspan=2)
    rdx = 29
    t_len = int(len(encoded_s[rdx][0,:,0,0])/2.0)
    for channel_dx in range(int(RSRN_channels[rdx])):
        plt.plot(corr_res[rdx][0,:,0,channel_dx] + 0 * 2,color='k')
    plt.xlim([0,t_len])
    plt.title('ECC conv1d_35',fontsize=14)
    t_min_plot = np.min(corr_res[rdx])
    t_max_plot = np.max(corr_res[rdx])
    t_gain = (t_max_plot - t_min_plot)*0.15
    plt.ylim([t_min_plot - t_gain,t_max_plot + t_gain])
    plt.plot([SEqT_P/4.0,SEqT_P/4.0],[-50,50],color='b',linestyle='--',label='S-EqT P')
    #plt.plot([SEqT_S/4.0,SEqT_S/4.0],[-50,50],color='r',linestyle='--',label='S-EqT S')
    t_ax=plt.gca();t_ax.spines['right'].set_color('b');t_ax.spines['top'].set_color('b');t_ax.spines['bottom'].set_color('b');t_ax.spines['left'].set_color('b');t_ax.spines['right'].set_linewidth(3);t_ax.spines['top'].set_linewidth(3);t_ax.spines['bottom'].set_linewidth(3);t_ax.spines['left'].set_linewidth(3)
    plt.xlabel('Array Index (N)',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax_eqt_final_resp = plt.subplot2grid((6,6),(4,0),colspan=3)
    rdx = 31
    t_len = int(len(siamese_input_list_no_normal[rdx*2+1][0,:,0,0])/2.0)
    for channel_dx in range(int(RSRN_channels[rdx])):
        plt.plot(siamese_input_list_no_normal[rdx*2+1][0,:t_len,0,channel_dx] + 0 * 2,color='k')    
    plt.xlim([0,t_len])
    plt.title('OSR conv1d_35',fontsize=14)
    t_min_plot = np.min(siamese_input_list_no_normal[rdx*2+1])
    t_max_plot = np.max(siamese_input_list_no_normal[rdx*2+1])
    t_gain = (t_max_plot - t_min_plot)*0.15
    plt.ylim([t_min_plot - t_gain,t_max_plot + t_gain])
    plt.plot([SEqT_P,SEqT_P],[-50,50],color='b',linestyle='--',label='S-EqT P')
    #plt.plot([SEqT_S,SEqT_S],[-50,50],color='r',linestyle='--',label='S-EqT S')
    #t_ax=plt.gca();t_ax.spines['right'].set_color('b');t_ax.spines['top'].set_color('b');t_ax.spines['bottom'].set_color('b');t_ax.spines['left'].set_color('b');t_ax.spines['right'].set_linewidth(3);t_ax.spines['top'].set_linewidth(3);t_ax.spines['bottom'].set_linewidth(3);t_ax.spines['left'].set_linewidth(3)
    plt.xlabel('Array Index (N)',fontsize=14)
    plt.title('Response of the penultimate layer in P branch of EqT model',fontsize=14)
    plt.ylabel('Amplitude',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax_seqt_final_resp = plt.subplot2grid((6,6),(4,3),colspan=3)
    for idx in range(7):
        plt.plot(SEqT_final_response[0,:,0,idx],color='k')
    t_min_plot = np.min(SEqT_final_response)
    t_max_plot = np.max(SEqT_final_response)
    t_gain = (t_max_plot - t_min_plot)*0.15
    plt.ylim([-20,t_max_plot + t_gain])
    plt.plot([SEqT_P,SEqT_P],[-40,40],color='b',linestyle='--',label='S-EqT P')
    #plt.plot([SEqT_S,SEqT_S],[-40,40],color='r',linestyle='--',label='S-EqT S')
    plt.xlim([0,t_len])
    #plt.ylabel('Amplitude',fontsize=14)
    plt.xlabel('Array Index (N)',fontsize=14)
    plt.title('Response of the penultimate layer in P branch of S-EqT model',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax_eqt_final = plt.subplot2grid((6,6),(5,0),colspan=3)
    plt.plot(plot_t,res_search[2][0,0:3001,0],color='k')
    plt.ylim([-0.05,0.3])
    plt.xlim([0,30])
    plt.plot([SEqT_P/100.0,SEqT_P/100.0],[-1,1],color='b',linestyle='--',label='S-EqT P')
    #plt.plot([SEqT_S/100.0,SEqT_S/100.0],[-1,1],color='r',linestyle='--',label='S-EqT S')
    plt.xlabel('time (s)',fontsize=14)
    plt.ylabel('Probability',fontsize=14)
    plt.title('P phase probabilities by EqT model',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax_seqt_final = plt.subplot2grid((6,6),(5,3),colspan=3)
    plt.plot(plot_t,pred_res[-1][0,0:3001,0,0],color='k')
    plt.xlim([0,30])
    plt.ylim([-0.05,0.3])
    plt.plot([SEqT_P/100.0,SEqT_P/100.0],[-1,1],color='b',linestyle='--',label='S-EqT P')
    #plt.plot([SEqT_S/100.0,SEqT_S/100.0],[-1,1],color='r',linestyle='--',label='S-EqT S')
    #plt.xlabel('time (s)',fontsize=14)
    plt.title('P phase probabilities by S-EqT model',fontsize=14)
    plt.xlabel('time (s)',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close()

    return

def sort_STEAD_csv_by_earthquake(csv_file):
    """
    read STEAD csv file and save a new file sorted by earthquakes
    """

    df = pd.read_csv(csv_file,low_memory=True)
    
    df1 = df.drop_duplicates(subset='source_id',keep='first')

    df.dropna()

    e_list = []

    counter = 0

    for m in df1['source_id']:
        # get rid of empty id
        if len(m) < 2:
            continue
        
        if counter % 1000 == 0:
            print('On {}'.format(counter))

        counter += 1

        mask = (df['source_id'] == m)
    
        t_df = df[mask].dropna()

        if len(t_df) > 1:
            e_list.append(t_df)
        else:
            continue

    for dfx in range(len(e_list)):
        name = "file{}".format(dfx)
        e_list[dfx].to_csv('./e_sort/{}.csv'.format(name))
        #save_dict[name] = e_list[dfx]
    #np.save('data.npy',save_dict)

    return

def get_train_list(cfgs, h5file, e_name):
    """
    Get train batch
    """
    # csv file
    csv_f = pd.read_csv(e_name)

    data_list = list()
    label_list = list()

    for key in csv_f['trace_name']:
        # data list
        dataset = h5file.get('data/'+key)
        data = np.array(dataset)
        data_list.append(data)

        spt = int(dataset.attrs['p_arrival_sample'])
        sst = int(dataset.attrs['s_arrival_sample'])
        coda_end = int(dataset.attrs['coda_end_sample'])
        snr = dataset.attrs['snr_db']

        temp_label = [spt, sst, coda_end, snr]
        
        label_list.append(temp_label)

    return data_list, label_list

def get_train_list_v2(cfgs, h5file, e_name):
    """
    Get train batch
    """
    # csv file
    csv_f = pd.read_csv(e_name)
    t_keys = csv_f['trace_name'].values
    csv_f.set_index('trace_name', inplace=True)

    data_list = list()
    label_list = list()

    for key in t_keys:
        t_df = csv_f.loc[key]
        chunk_id = int(t_df['chunk_id'])
        # data list
        dataset = h5file[chunk_id].get('data/'+key)
        data = np.array(dataset)
        data_list.append(data)

        spt = int(dataset.attrs['p_arrival_sample'])
        sst = int(dataset.attrs['s_arrival_sample'])
        coda_end = int(dataset.attrs['coda_end_sample'])
        snr = dataset.attrs['snr_db']
        temp_label = [spt, sst, coda_end, snr]
        label_list.append(temp_label)

    return data_list, label_list

def get_test_list(cfgs, h5file, e_name, test_trace_name):
    """
    Get train batch
    """
    # csv file
    csv_f = pd.read_csv(e_name)
    t_keys = csv_f['trace_name'].values
    csv_f.set_index('trace_name', inplace=True)
    #print(t_keys[0])
    #shuffle(t_keys)
    
    data_list = list()
    label_list = list()

    # test search data
    test_key = test_trace_name
    t_df = csv_f.loc[test_key]
    chunk_id = int(t_df['chunk_id'])
    # data list
    dataset = h5file[chunk_id].get('data/'+test_key)
    
    #print(test_key)
    #print(chunk_id)

    data = np.array(dataset)
    data_list.append(data)

    spt = int(dataset.attrs['p_arrival_sample'])
    sst = int(dataset.attrs['s_arrival_sample'])
    coda_end = int(dataset.attrs['coda_end_sample'])
    snr = dataset.attrs['snr_db']
    temp_label = [spt, sst, coda_end, snr]
    label_list.append(temp_label)
    
    for key in t_keys:
        if key == test_trace_name:
            continue
        
        t_df = csv_f.loc[key]
        chunk_id = int(t_df['chunk_id'])

        # data list
        dataset = h5file[chunk_id].get('data/'+key)
        data = np.array(dataset)
        data_list.append(data)
        #print(test_key)
        #print(chunk_id)
        spt = int(dataset.attrs['p_arrival_sample'])
        sst = int(dataset.attrs['s_arrival_sample'])
        coda_end = int(dataset.attrs['coda_end_sample'])
        snr = dataset.attrs['snr_db']
        temp_label = [spt, sst, coda_end, snr]
        label_list.append(temp_label)
        
        break
    
    return data_list, label_list

def load_data_from_Siamese_STEAD(f_name, chunk_name):
    """
    Siamese STEAD
    """
    # hdf5 file
    f1 = h5py.File(chunk_name)
    for key in f1.keys():
        print(key)

    # read csv file
    csv_f = pd.read_csv(f_name)
    for key in csv_f['trace_name']:
        print(key)
        dataset = f1.get('data/'+key)
        data = np.array(dataset)
        print(np.shape(data))
        spt = int(dataset.attrs['p_arrival_sample'])
        sst = int(dataset.attrs['s_arrival_sample'])
        coda_end = int(dataset.attrs['coda_end_sample'])
        snr = dataset.attrs['snr_db']
        plt.figure(figsize=(16,4))
        
        plt.subplot(3,1,1)
        plt.title('P: {}  S: {}  CODA_END: {}  SNR:{}'.format(spt,sst,coda_end,snr))

        plt.plot(data[:,0],color='k')
        max_d = np.max(data[:,0])
        min_d = np.min(data[:,0])
        plt.plot([spt,spt],[min_d,max_d],color='b',label='P-arrival')
        max_d = np.max(data[:,0])
        min_d = np.min(data[:,0])
        plt.plot([sst,sst],[min_d,max_d],color='r',label='S-arrival')
        max_d = np.max(data[:,0])
        min_d = np.min(data[:,0])
        plt.plot([coda_end,coda_end],[min_d,max_d],color='g',label='Coda End')

        plt.subplot(3,1,2)
        plt.plot(data[:,1],color='k')
        max_d = np.max(data[:,1])
        min_d = np.min(data[:,1])
        plt.plot([spt,spt],[min_d,max_d],color='b',label='P-arrival')
        max_d = np.max(data[:,1])
        min_d = np.min(data[:,1])
        plt.plot([sst,sst],[min_d,max_d],color='r',label='S-arrival')
        max_d = np.max(data[:,1])
        min_d = np.min(data[:,1])
        plt.plot([coda_end,coda_end],[min_d,max_d],color='g',label='Coda End')

        plt.subplot(3,1,3)
        plt.plot(data[:,2],color='k')        
        max_d = np.max(data[:,2])
        min_d = np.min(data[:,2])
        plt.plot([spt,spt],[min_d,max_d],color='b',label='P-arrival')
        max_d = np.max(data[:,2])
        min_d = np.min(data[:,2])
        plt.plot([sst,sst],[min_d,max_d],color='r',label='S-arrival')
        max_d = np.max(data[:,2])
        min_d = np.min(data[:,2])
        plt.plot([coda_end,coda_end],[min_d,max_d],color='g',label='Coda End')
        
        plt.show()

    return

if __name__ == '__main__':
    # rand_id = np.random.randint(low=0,high=36246)
    # print(rand_id)
    # rand_id = 19531
    # choose 19531
    load_data_from_Siamese_STEAD('D:/SourceCode/SiameseEQT/src/e_sort/file{}'.format(rand_id),'E:/STEAD_DATA/chunk2/chunk2.hdf5')
    #sort_STEAD_csv_by_earthquake('E:/STEAD_DATA/chunk2.csv')