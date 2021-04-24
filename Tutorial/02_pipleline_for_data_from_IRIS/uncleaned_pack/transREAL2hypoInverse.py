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

if __name__ == '__main__':

    cfgs = dict()
    cfgs['CSV2REAL'] = dict()
    cfgs['CSV2REAL']['save_sta'] = './SA_sta_seqt_use.dat'
    cfgs['REAL2HYPO'] = dict()
    cfgs['REAL2HYPO']['save_sta'] = './revise_dataset/ori_seqt/LAN_HYPO_Use_seqt.dat'
    cfgs['REAL2HYPO']['event_dict'] = './revise_dataset/ori_seqt/20200801_e_dict_seqt_knife.npy'
    cfgs['REAL2HYPO']['save_pha'] = './revise_dataset/ori_seqt/LAN_HYPO_PHA_seqt.pha'
    create_station_file(cfgs)
    create_pha_file(cfgs)