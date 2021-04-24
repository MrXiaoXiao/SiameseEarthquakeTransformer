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

def generate_real_sta_dat(cfgs):
    f = open(cfgs['STACOORD'],'r')
    lines = f.readlines()
    f.close()
    save_f = open(cfgs['CSV2REAL']['save_sta'],'w')

    for line in lines:
        splits = line.split(' ')
        sta_name = splits[0]
        lat = float(splits[2])
        lon = float(splits[1])
        save_f.write('{:.4f} {:.4f} {:} {:} {:} {:.3f}\n'.format(
                        lon,lat,'BY',sta_name,'BHZ',0.0))
    save_f.close()

    return

def pad_empty_sta(cfgs):
    f = open(cfgs['STACOORD'],'r')
    lines = f.readlines()
    f.close()
    save_prefix_list = cfgs['CSV2REAL']['save_prefix']
    save_folder = cfgs['CSV2REAL']['save_folder']

    for line in lines:
        splits = line.split(' ')
        sta_name = splits[0]
        for t_prefix in save_prefix_list:
            if os.path.exists(save_folder + t_prefix):
                pass
            else:
                continue
            t_P_name = save_folder + t_prefix + 'BY.'+sta_name+'.P.txt'
            t_S_name = save_folder + t_prefix + 'BY.'+sta_name+'.S.txt'
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

def REAL_RUNDENCY_BUG_FIX(ori_name,target_name):
    f = open(ori_name,'r')
    f_target = open(target_name,'w')
    for line in f.readlines():
        line_split = re.sub('\s{2,}',' ',line)
        splits = line_split.split(' ')
        if len(splits) > 10:
            f_target.write(line)
            t_list = list()
        else:
            t_key = splits[2] + splits[3]
            if t_key in t_list:
                continue
            else:
                t_list.append(t_key)
                f_target.write(line)
    f.close()
    f_target.close()
    os.remove(ori_name)
    return

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
        REAL_RUNDENCY_BUG_FIX('./phase_sel.txt', './REAL_Res/{}_phase_sel.txt'.format(cfgs['CSV2REAL']['save_prefix'][idx][:-1]))
        #os.rename('./phase_sel.txt', './{}_phase_sel.txt'.format(cfgs['CSV2REAL']['save_prefix'][idx][:-1]))
    return

def runVELEST(cfgs):
    """
    Run VELEST Scripts
    """
    for idx in range(len(cfgs['CSV2REAL']['save_prefix'])):
        # merge together
        f_perl = open('./VELEST_scripts/mergetogether.pl', 'r')
        f_perl_source = f_perl.read()
        f_perl.close()
        f_perl_source = f_perl_source.replace('MERGE_DIR_KEY', cfgs['VELESTPARAMS']['merge_dir_key'])
        f_perl_source = f_perl_source.replace('MERGE_TOGETHER_KEY', cfgs['VELESTPARAMS']['merge_together_key'])
        f_perl_source = f_perl_source.replace('MERGE_TARGET_KEY', cfgs['VELESTPARAMS']['merge_target_key'].format(cfgs['CSV2REAL']['save_prefix'][idx][:-1]))
        
        f_perl_temp = open('./VELEST_scripts/mergetogether_temp.pl','w')
        f_perl_temp.write(f_perl_source)
        f_perl_temp.close()

        velest_merge_output = os.system('./VELEST_scripts/mergetogether_temp.pl')
        print('STATUS: {}'.format(velest_merge_output))
        
        # convert output
        f_perl = open('./VELEST_scripts/convertformat.pl', 'r')
        f_perl_source = f_perl.read()
        f_perl.close()
        f_perl_source = f_perl_source.replace('STATION_KEY', cfgs['VELESTPARAMS']['format_station_key'])
        f_perl_source = f_perl_source.replace('MERGE_TOGETHER_KEY', cfgs['VELESTPARAMS']['merge_together_key'])
        
        f_perl_temp = open('./VELEST_scripts/convertformat_temp.pl','w')
        f_perl_temp.write(f_perl_source)
        f_perl_temp.close()

        velest_convert_output = os.system('./VELEST_scripts/convertformat_temp.pl')
        print('STATUS: {}'.format(velest_convert_output))

        # velest relocate
        # cal neqs
        neqs = '{}'.format(len(open('./VELEST_scripts/new.cat','r').readlines()))
        f_perl = open('./VELEST_scripts/velest_template.cmn','r')
        f_perl_source = f_perl.read()
        f_perl.close()
        f_perl_source = f_perl_source.replace('OLAT_KEY', cfgs['VELESTPARAMS']['olat'])
        f_perl_source = f_perl_source.replace('OLON_KEY', cfgs['VELESTPARAMS']['olon'])
        f_perl_source = f_perl_source.replace('NEQS_KEY', neqs)
        f_perl_source = f_perl_source.replace('ITTMAX_KEY', cfgs['VELESTPARAMS']['ittmax'])
        f_perl_source = f_perl_source.replace('INV_RATIO_KEY', cfgs['VELESTPARAMS']['inv_ratio'])
        f_perl_source = f_perl_source.replace('DMAX_KEY', cfgs['VELESTPARAMS']['dmax'])

        f_perl_temp = open('./VELEST_scripts/velest.cmn','w')
        f_perl_temp.write(f_perl_source)
        f_perl_temp.close()

        os.chdir('./VELEST_scripts/')
        velest_output = os.system('./velest')
        print('STATUS: {}'.format(velest_output))
        os.chdir('../')
        os.rename('./VELEST_scripts/final.CNV', './VELESTRes/{}_final.CNV'.format(cfgs['CSV2REAL']['save_prefix'][idx][:-1]))

    return

def merge_CNV_phasesel(cfgs):
    """
    Merge CNV and phase sel files
    """
    for idx in range(len(cfgs['CSV2REAL']['save_prefix'])):
        
        e_dict = dict()
        e_counter = 1
        base_time = obspy.UTCDateTime(cfgs['CSV2REAL']['ref_time_list'][idx])
        
        f_cnv = open('./VELESTRes/{}_final.CNV'.format(cfgs['CSV2REAL']['save_prefix'][idx][:-1]), 'r')
        for line in f_cnv.readlines():
            if len(line) > 25:
                pass
            else:
                continue

            if line[25] == 'N' or line[25] == 'S':
                e_ID = '{}'.format(e_counter)
                e_dict[e_ID] = dict()
                year = 2000 + int(line[0:2])
                month = int(line[2:4])
                day = int(line[4:6])
                hour = int(line[7:9])
                minute = int(line[9:11])
                second = float(line[11:17])
                if minute == 60:
                    minute -=1
                    second += 60.0
                velest_time = obspy.UTCDateTime(year,month,day,hour,minute) + second
                e_dict[e_ID]['VELEST_TIME'] = velest_time
                e_dict[e_ID]['VELEST_LAT'] = float(line[17:25])
                if  line[25] == 'S':
                    e_dict['VELEST_LAT'] *= -1.0
                
                e_dict[e_ID]['VELEST_LON'] = float(line[26:35])
                if  line[35] == 'W':
                    e_dict[e_ID]['VELEST_LON'] *= -1.0
                e_dict[e_ID]['VELEST_DEP'] = float(line[36:43])
                e_counter += 1            
        f_cnv.close()
        
        e_ID = None

        f_sel = open('./REAL_Res/{}_phase_sel.txt'.format(cfgs['CSV2REAL']['save_prefix'][idx][:-1]) ,'r')
        for line in f_sel.readlines():
            line_split = re.sub('\s{2,}',' ',line).split(' ')
            if len(line_split) > 11:
                e_ID = '{}'.format(int(line_split[1]))
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
        np.save('./VELESTRes/{}_e_dict.npy'.format(cfgs['CSV2REAL']['save_prefix'][idx][:-1]),e_dict)
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

if __name__ == '__main__':
    
    cfgs = yaml.load(open('SEQT_config.LA.yaml','r'), Loader=yaml.FullLoader)

    data_folder_list = cfgs['CSV2REAL']['data_folder_list']
    print(data_folder_list)
    for idx in range(len(data_folder_list)):
        for outfolder in Path(data_folder_list[idx]).glob('*_outputs'):
            csv_name = str(outfolder) + '/X_prediction_results.csv'
            state = convert_csv_to_real(csv_name, cfgs, idx)
            if state == 0:
                print('Error On {}'.format(outfolder.name))
            else:
                print('Success On {}'.format(outfolder.name))
    
    #generate_real_sta_dat(cfgs)
    #pad_empty_sta(cfgs)
    
    cfgs = yaml.load(open('SEQT_config.LA.yaml','r'), Loader=yaml.FullLoader)
    runREAL(cfgs)
    #runVELEST(cfgs)
    #merge_CNV_phasesel(cfgs)
