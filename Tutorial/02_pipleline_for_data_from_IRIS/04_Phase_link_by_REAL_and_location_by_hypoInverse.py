import obspy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from obspy import read, UTCDateTime
import os
import math
import yaml
from pathlib import Path
import shutil
import re
import argparse

def create_station_file(cfgs):
    real_sta_file_path = cfgs['REAL']['save_sta']
    real_sta_file = open(real_sta_file_path, 'r')
    
    hypo_sta_file_path = cfgs['HypoInverse']['save_sta']
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
    real_event_dict_path =  cfgs['HypoInverse']['eqt_event_dict']
    real_event_dict = np.load(real_event_dict_path,allow_pickle=True )[()]
    
    hypo_pha_file_path = cfgs['HypoInverse']['save_pha_eqt']
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

    real_event_dict_path =  cfgs['HypoInverse']['seqt_event_dict']
    real_event_dict = np.load(real_event_dict_path,allow_pickle=True )[()]
    
    hypo_pha_file_path = cfgs['HypoInverse']['save_pha_seqt']
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

def runREAL(cfgs):
    """
    Run REAL Scripts
    """
    for idx in range(len(cfgs['REAL']['year'])):
        # copy temp perl file
        f_perl = open('./REAL_scripts/runREAL.pl', 'r')
        f_perl_source = f_perl.read()
        f_perl.close()
        f_perl_source = f_perl_source.replace('YEAR_KEY', cfgs['REAL']['year'][idx])
        f_perl_source = f_perl_source.replace('MON_KEY', cfgs['REAL']['mon'][idx])
        f_perl_source = f_perl_source.replace('DAY_KEY', cfgs['REAL']['day'][idx])
        f_perl_source = f_perl_source.replace('DIR_KEY','\"'  + cfgs['REAL']['eqt_dir']  + '\"')
        f_perl_source = f_perl_source.replace('STATION_KEY', cfgs['REAL']['station'])
        f_perl_source = f_perl_source.replace('TTIME_KEY', cfgs['REAL']['ttime'])
        f_perl_source = f_perl_source.replace('R_KEY', cfgs['REAL']['R'])
        f_perl_source = f_perl_source.replace('G_KEY', cfgs['REAL']['G'])
        f_perl_source = f_perl_source.replace('V_KEY', cfgs['REAL']['V'])
        f_perl_source = f_perl_source.replace('S_KEY', cfgs['REAL']['S'])
        f_perl_temp = open('./REAL_scripts/runREAL_temp.pl','w')
        f_perl_temp.write(f_perl_source)
        f_perl_temp.close()
        real_output = os.system('./REAL_scripts/runREAL_temp.pl')
        print('STATUS: {}'.format(real_output))
        
        os.rename('./catalog_sel.txt', './catalogs/eqt_real_catalog_sel.txt')
        os.rename('./phase_sel.txt', './catalogs/eqt_real_phase_sel.txt')

        # copy temp perl file
        f_perl = open('./REAL_scripts/runREAL.pl', 'r')
        f_perl_source = f_perl.read()
        f_perl.close()
        f_perl_source = f_perl_source.replace('YEAR_KEY', cfgs['REAL']['year'][idx])
        f_perl_source = f_perl_source.replace('MON_KEY', cfgs['REAL']['mon'][idx])
        f_perl_source = f_perl_source.replace('DAY_KEY', cfgs['REAL']['day'][idx])
        f_perl_source = f_perl_source.replace('DIR_KEY','\"'  + cfgs['REAL']['seqt_dir']  + '\"')
        f_perl_source = f_perl_source.replace('STATION_KEY', cfgs['REAL']['station'])
        f_perl_source = f_perl_source.replace('TTIME_KEY', cfgs['REAL']['ttime'])
        f_perl_source = f_perl_source.replace('R_KEY', cfgs['REAL']['R'])
        f_perl_source = f_perl_source.replace('G_KEY', cfgs['REAL']['G'])
        f_perl_source = f_perl_source.replace('V_KEY', cfgs['REAL']['V'])
        f_perl_source = f_perl_source.replace('S_KEY', cfgs['REAL']['S'])
        f_perl_temp = open('./REAL_scripts/runREAL_temp.pl','w')
        f_perl_temp.write(f_perl_source)
        f_perl_temp.close()
        real_output = os.system('./REAL_scripts/runREAL_temp.pl')
        print('STATUS: {}'.format(real_output))
        
        os.rename('./catalog_sel.txt', './catalogs/seqt_real_catalog_sel.txt')
        os.rename('./phase_sel.txt', './catalogs/seqt_real_phase_sel.txt')

    return

def merge_phasesel(cfgs):
    """
    Merge phase sel files
    """
    e_dict = dict()
    base_time = obspy.UTCDateTime(cfgs['REAL']['ref_time'])
    e_ID = None
    f_sel = open('./catalogs/eqt_real_phase_sel.txt','r')
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
    np.save('./catalogs/eqt_real_e_dict.npy',e_dict)

    e_dict = dict()
    base_time = obspy.UTCDateTime(cfgs['REAL']['ref_time'])
    e_ID = None
    f_sel = open('./catalogs/seqt_real_phase_sel.txt','r')
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
    
    np.save('./catalogs/seqt_real_e_dict.npy',e_dict)

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
    f = open(cfgs['REAL']['save_sta'],'r')
    lines = f.readlines()
    f.close()

    save_folder = cfgs['REAL']['eqt_dir']
    for line in lines:
        splits = line.split(' ')
        sta_name = splits[3]
        net_name = splits[2]

        t_P_name = save_folder + net_name + '.' +sta_name+'.P.txt'
        t_S_name = save_folder + net_name + '.' +sta_name+'.S.txt'
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

    f = open(cfgs['REAL']['save_sta'],'r')
    lines = f.readlines()
    f.close()    
    save_folder = cfgs['REAL']['seqt_dir']
    for line in lines:
        splits = line.split(' ')
        sta_name = splits[3]
        net_name = splits[2]

        t_P_name = save_folder + net_name + '.' +sta_name+'.P.txt'
        t_S_name = save_folder + net_name + '.' +sta_name+'.S.txt'
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
    parser = argparse.ArgumentParser(description='04_Phase_link_by_REAL_and_location_by_HypoInverse')
    parser.add_argument('--config-file', dest='config_file', 
                        type=str, help='Configuration file path',default='./default_pipline_config.yaml')
    args = parser.parse_args()
    cfgs = yaml.load(open(args.config_file,'r'),Loader=yaml.SafeLoader)
    
    pad_empty_sta(cfgs)
    runREAL(cfgs)
    merge_phasesel(cfgs)
    
    create_station_file(cfgs)
    create_pha_file(cfgs)
   
    os.rename(cfgs['HypoInverse']['save_sta'], './Hypoinverse_scripts/input/HYPO.sta')
    os.rename(cfgs['HypoInverse']['save_pha_eqt'], './Hypoinverse_scripts/input/HYPO.pha')
    os.chdir('./Hypoinverse_scripts')
    hypo_output = os.system('python run_hyp.py')
    print('STATUS: {}'.format(hypo_output))
    os.chdir('..')
    os.rename( './Hypoinverse_scripts/output/example.ctlg', './catalogs/eqt_hypoInverse.ctlg')
    os.rename( './Hypoinverse_scripts/output/example.pha','./catalogs/eqt_hypoInverse.pha')
    os.rename( './Hypoinverse_scripts/output/example.sum','./catalogs/eqt_hypoInverse.sum')
    os.rename( './Hypoinverse_scripts/output/example_good.csv','./catalogs/eqt_hypoInverse.good')
    os.rename( './Hypoinverse_scripts/output/example_bad.csv','./catalogs/eqt_hypoInverse.bad')

    os.rename(cfgs['HypoInverse']['save_pha_seqt'], './Hypoinverse_scripts/input/HYPO.pha')
    os.chdir('./Hypoinverse_scripts')
    hypo_output = os.system('python run_hyp.py')
    print('STATUS: {}'.format(hypo_output))
    os.chdir('..')
    os.rename( './Hypoinverse_scripts/output/example.ctlg', './catalogs/seqt_hypoInverse.ctlg')
    os.rename( './Hypoinverse_scripts/output/example.pha','./catalogs/seqt_hypoInverse.pha')
    os.rename( './Hypoinverse_scripts/output/example.sum','./catalogs/seqt_hypoInverse.sum')
    os.rename( './Hypoinverse_scripts/output/example_good.csv','./catalogs/seqt_hypoInverse.good')
    os.rename( './Hypoinverse_scripts/output/example_bad.csv','./catalogs/seqt_hypoInverse.bad')