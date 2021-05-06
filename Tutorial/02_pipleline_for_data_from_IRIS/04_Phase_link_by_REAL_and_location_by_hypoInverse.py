import obspy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import obspy
from obspy import read, UTCDateTime
import os
import math
import yaml
from pathlib import Path
import shutil
import re

import pandas as pd
import numpy as np
import glob
import obspy
from obspy import UTCDateTime
import os
import math
import yaml
from pathlib import Path
import shutil
import re

def create_station_file(cfgs):
    real_sta_file_path = cfgs['CSV2REAL']['save_sta']
    real_sta_file = open(real_sta_file_path, 'r')
    
    hypo_sta_file_path = cfgs['REAL2HYPO']['save_sta']
    hypo_sta_file = open(hypo_sta_file_path, 'w')

    for line in real_sta_file.readlines():
        splits = line.split(' ')
        lat = '{:.5f}'.format(float(splits[1]))
        lon = '{:.5f}'.format(float(splits[0]))
        code = splits[2]+'.'+splits[3]
        ele = '{:.2f}'.format(float(splits[-1])*1000.0)
        pad = '-1'

        hypo_line = '\t'.join([code,lat,lon,ele,pad]) + '\n'

        hypo_sta_file.write(hypo_line)
    
    real_sta_file.close()
    hypo_sta_file.close()
    return

def create_pha_file(cfgs):
    real_event_dict_path =  cfgs['REAL2HYPO']['event_dict']
    real_event_dict = np.load(real_event_dict_path,allow_pickle=True )[()]
    
    hypo_pha_file_path = cfgs['REAL2HYPO']['save_pha']
    hypo_pha_file = open(hypo_pha_file_path,'w')
    
    for e_key in real_event_dict.keys():
        ot = str(real_event_dict[e_key]['REAL_TIME'])
        lat = '{:5f}'.format(real_event_dict[e_key]['REAL_LAT'])
        lon = '{:5f}'.format(real_event_dict[e_key]['REAL_LON'])
        dep = '{:5f}'.format(real_event_dict[e_key]['REAL_DEP'])
        mag = '1.0'
        # create event line
        event_line = ','.join([ot,lat,lon,dep,mag]) + '\n'
        hypo_pha_file.write(event_line)
        
        temp_pick_dict = dict()
        for pick_info in real_event_dict[e_key]['Picks']:
            code = pick_info[0]
            pick_type = pick_info[1]
            pick_time = pick_info[2]

            if code in temp_pick_dict.keys():
                temp_pick_dict[code][pick_type] = pick_time
            else:
                temp_pick_dict[code] = dict()
                temp_pick_dict[code]['P'] = -1
                temp_pick_dict[code]['S'] = -1
                temp_pick_dict[code][pick_type] = pick_time
        
        for pick_key in temp_pick_dict.keys():
            net = pick_key.split('.')[0]
            sta = pick_key.split('.')[1]
            tp = str(temp_pick_dict[pick_key]['P'])
            ts = str(temp_pick_dict[pick_key]['S'])
            pick_line =  ','.join([net,sta,tp,ts]) + ',-1,-1,-1\n'
            hypo_pha_file.write(pick_line)
    
    hypo_pha_file.close()
    return

def xml2REAL_sta(cfgs):
    sta_inv = obspy.read_inventory(cfgs['STAXML'])
    save_f = open(cfgs['CSV2REAL']['save_sta'],'w')

    for net in sta_inv.networks:
        for sta in net.stations:
            if sta.longitude > cfgs['MINLON'] and sta.longitude < cfgs['MAXLON'] and sta.latitude > cfgs['MINLAT'] and sta.latitude < cfgs['MAXLAT']:
                pass
            else:
                continue   

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

def convert_csv_to_real(picks_csv_path,cfgs,job_ID=0):
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
    reftime = cfgs['CSV2REAL']['ref_time_list'][job_ID]
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
    save_prefix = cfgs['CSV2REAL']['save_folder'] + cfgs['CSV2REAL']['save_prefix'][job_ID]
    if os.path.exists(save_prefix):
        pass
    else:
        os.makedirs(save_prefix)
    np.savetxt(save_prefix+"%s.P.txt"%(save_file_name), itp_prob_arr, fmt='%.3f %.5f %.8f')
    np.savetxt(save_prefix+"%s.S.txt"%(save_file_name), its_prob_arr, fmt='%.3f %.5f %.8f')
    return 1

def runREAL(cfgs):
    """
    Run REAL Scripts
    """
    for idx in range(len(cfgs['REALPARAMS']['year'])):
        # copy temp perl file
        f_perl = open('./REAL_scripts/runREAL.pl', 'r')
        f_perl_source = f_perl.read()
        f_perl.close()
        f_perl_source = f_perl_source.replace('YEAR_KEY', cfgs['REALPARAMS']['year'][idx])
        f_perl_source = f_perl_source.replace('MON_KEY', cfgs['REALPARAMS']['mon'][idx])
        f_perl_source = f_perl_source.replace('DAY_KEY', cfgs['REALPARAMS']['day'][idx])
        f_perl_source = f_perl_source.replace('DIR_KEY','\"'  + cfgs['REALPARAMS']['dir'] + cfgs['CSV2REAL']['save_prefix'][idx] + '\"')
        f_perl_source = f_perl_source.replace('STATION_KEY', cfgs['REALPARAMS']['station'])
        f_perl_source = f_perl_source.replace('TTIME_KEY', cfgs['REALPARAMS']['ttime'])
        f_perl_source = f_perl_source.replace('R_KEY', cfgs['REALPARAMS']['R'])
        f_perl_source = f_perl_source.replace('G_KEY', cfgs['REALPARAMS']['G'])
        f_perl_source = f_perl_source.replace('V_KEY', cfgs['REALPARAMS']['V'])
        f_perl_source = f_perl_source.replace('S_KEY', cfgs['REALPARAMS']['S'])
        f_perl_temp = open('./REAL_scripts/runREAL_temp.pl','w')
        f_perl_temp.write(f_perl_source)
        f_perl_temp.close()
        real_output = os.system('./REAL_scripts/runREAL_temp.pl')
        print('STATUS: {}'.format(real_output))
        
        os.rename('./catalog_sel.txt', './REAL_Res/{}_catalog_sel.txt'.format(cfgs['CSV2REAL']['save_prefix'][idx][:-1]))
        os.rename('./phase_sel.txt', './REAL_Res/{}_phase_sel.txt'.format(cfgs['CSV2REAL']['save_prefix'][idx][:-1]))
    return

def merge_phasesel(cfgs):
    """
    Merge CNV and phase sel files
    """
    for idx in range(len(cfgs['CSV2REAL']['save_prefix'])):
        e_dict = dict()
        base_time = obspy.UTCDateTime(cfgs['CSV2REAL']['ref_time_list'][idx])
        e_ID = None
        f_sel = open('./REAL_Res/{}_phase_sel.txt'.format(cfgs['CSV2REAL']['save_prefix'][idx][:-1]) ,'r')
        for line in f_sel.readlines():
            line_split = re.sub('\s{2,}',' ',line).split(' ')
            if len(line_split) > 11:
                e_ID = '{}'.format(int(line_split[1]))
                e_dict[e_ID] = dict()
                real_time = base_time + float(line_split[6])
                e_dict[e_ID]['REAL_TIME'] = real_time
                e_dict[e_ID]['REAL_LAT'] = float(line_split[8])
                e_dict[e_ID]['REAL_LON'] = float(line_split[9])
                e_dict[e_ID]['REAL_DEP'] = float(line_split[10])
                e_dict[e_ID]['Picks'] = list()
            else:
                sta_name = line_split[1] + '.' + line_split[2]
                pick_type = line_split[3]
                pick_time = base_time + float(line_split[4])
                if pick_time - e_dict[e_ID]['REAL_TIME'] < 0.01:
                    continue
                e_dict[e_ID]['Picks'].append([sta_name, pick_type, pick_time])
        f_sel.close()
        #print_dict(e_dict)
        np.save('./REAL_Res/{}_e_dict.npy'.format(cfgs['CSV2REAL']['save_prefix'][idx][:-1]),e_dict)
    return

def print_dict(e_dict):
    for key in e_dict.keys():
        print('E_ID: {} VELTime: {} LAT: {} LON: {} DEP: {}'.format(key,
                                                                e_dict[key]['VELEST_TIME'],
                                                                e_dict[key]['VELEST_LAT'],
                                                                e_dict[key]['VELEST_LON'],
                                                                e_dict[key]['VELEST_DEP']))

        print('REALTime: {} LAT: {} LON: {} DEP: {}'.format(e_dict[key]['REAL_TIME'],
                                                                e_dict[key]['REAL_LAT'],
                                                                e_dict[key]['REAL_LON'],
                                                                e_dict[key]['REAL_DEP']))
        for pick in e_dict[key]['Picks']:
            print(pick)
    return

def pad_empty_sta(cfgs):
    f = open(cfgs['CSV2REAL']['save_sta'],'r')
    lines = f.readlines()
    f.close()

    save_prefix_list = cfgs['CSV2REAL']['save_prefix']
    save_folder = cfgs['CSV2REAL']['save_folder']

    for line in lines:
        splits = line.split(' ')
        sta_name = splits[3]
        net_name = splits[2]
        for t_prefix in save_prefix_list:
            if os.path.exists(save_folder + t_prefix):
                pass
            else:
                continue
            t_P_name = save_folder + t_prefix + net_name + '.' +sta_name+'.P.txt'
            t_S_name = save_folder + t_prefix + net_name + '.' +sta_name+'.S.txt'
            if os.path.exists(t_P_name):
                pass
            else:
                t_f = open(t_P_name, 'w')
                t_f.close()
            if os.path.exists(t_S_name):
                pass
            else:
                t_f = open(t_S_name, 'w')
                t_f.close()
    return

if __name__ == '__main__':
    cfgs = yaml.load(open('iceSeisConfig.yaml','r'),Loader=yaml.Loader)  
    
    pad_empty_sta(cfgs)
    runREAL(cfgs)
    merge_phasesel(cfgs)
    
    create_station_file(cfgs)
    create_pha_file(cfgs)
   
    os.rename(cfgs['REAL2HYPO']['save_sta'], './hypoinverse/input/HYPO.sta')
    os.rename(cfgs['REAL2HYPO']['save_pha'], './hypoinverse/input/HYPO.pha')
    os.chdir('./hypoinverse')
    hypo_output = os.system('python run_hyp.py')
    print('STATUS: {}'.format(hypo_output))
    os.chdir('..')
    os.rename( './hypoinverse/output/example.ctlg', './HypoRes/{}.ctlg'.format(cfgs['REAL2HYPO']['HYPO_key']) )
    os.rename( './hypoinverse/output/example.pha','./HypoRes/{}.pha'.format(cfgs['REAL2HYPO']['HYPO_key']) )
    os.rename( './hypoinverse/output/example.sum','./HypoRes/{}.sum'.format(cfgs['REAL2HYPO']['HYPO_key']) )
    os.rename( './hypoinverse/output/example_good.csv','./HypoRes/{}.good'.format(cfgs['REAL2HYPO']['HYPO_key']) )
    os.rename( './hypoinverse/output/example_bad.csv','./HypoRes/{}.bad'.format(cfgs['REAL2HYPO']['HYPO_key']) )
