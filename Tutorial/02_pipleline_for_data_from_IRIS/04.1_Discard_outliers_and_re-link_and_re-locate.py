"""
TO BE UPDATED BEFORE 05.19.2020!!!
"""

"""

import yaml
import re
import obspy
import matplotlib.pyplot as plt
from pyproj import Geod
import numpy as np
GEOD = Geod(ellps='WGS84')

def build_sta_dict_zdp_format(sta_file_name):
    sta_dict = dict()
    
    f = open(sta_file_name,'r')
    for line in f.readlines():
        line = re.sub('\s{2,}',' ',line)
        splits = line.split(' ')
        sta_key = splits[1]
        sta_name = splits[2]
        sta_lat = float(splits[3])
        sta_lon = float(splits[4])
        sta_dep = float(splits[5])
        sta_dict[sta_key] = (sta_name,sta_lat,sta_lon,sta_dep)
    f.close()

    return sta_dict


if __name__ == '__main__':
    cfgs = yaml.load(open('./config.yaml','r'),Loader=yaml.SafeLoader)
    sta_dict = build_sta_dict_zdp_format(cfgs['STA_FILE'])
    
    ori_phase_file_name = cfgs['PHA_FILE']
    output_phase_file_name = cfgs['PHA_FILE'] + '_outliers_discarded'
    
    output_phase_file = open(output_phase_file_name, 'w')
    ori_phase_file = open(ori_phase_file_name, 'r')
    
    Vp_min = cfgs['Vp_min']
    Vp_max = cfgs['Vp_max']
    Vs_min = cfgs['Vs_min']
    Vs_max = cfgs['Vs_max']

    plot_x_ori_p = list()
    plot_y_ori_p = list()
    plot_x_good_p = list()
    plot_y_good_p = list()

    plot_x_ori_s = list()
    plot_y_ori_s = list()
    plot_x_good_s = list()
    plot_y_good_s = list()

    out_write_list = list()

    for line in ori_phase_file.readlines():
        re_line = re.sub('\s{2,}',' ',line)
        re_splits = re_line.split(' ')
        if len(re_splits) > 16:
            if len(out_write_list) > 2:
                t_line = out_write_list[0]
                t_line = t_line[:86] + '{:4.0f} {:4.0f} {:4.0f}'.format(P_NUM,S_NUM,P_NUM+S_NUM) + t_line[94:]
                out_write_list[0] = t_line
                for write_line_out in out_write_list:
                    output_phase_file.write(write_line_out)
            out_write_list = list()
            out_write_list.append(line)
            P_NUM = 0
            S_NUM = 0
            #output_phase_file.write(line)
            e_time = obspy.UTCDateTime(int(re_splits[2]),int(re_splits[3]),int(re_splits[4]),int(re_splits[5]),int(re_splits[6]), 0.0) + float(re_splits[7])
            e_time_base = obspy.UTCDateTime(int(re_splits[2]),int(re_splits[3]),int(re_splits[4]),int(re_splits[5]),int(re_splits[6]), 0.0)
            e_lat = float(re_splits[9])
            e_lon = float(re_splits[11])
            e_dep = float(re_splits[13])
        else:
            sta_key = re_splits[1]
            arrival = float(re_splits[2])
            phase_type = int(re_splits[3])

            s_lat = sta_dict[sta_key][1]
            s_lon = sta_dict[sta_key][2]
            s_dep = sta_dict[sta_key][3]
            h_dis = GEOD.inv(e_lon,e_lat,s_lon,s_lat)[2]/1000.0
            v_dis = float(e_dep - s_dep)
            e_s_dis = np.sqrt(h_dis**2 + v_dis**2)
            travel_time = e_time_base + arrival - e_time
            v_vis = e_s_dis/travel_time
            
            if phase_type == 1:
                plot_x_ori_p.append(e_s_dis)
                plot_y_ori_p.append(travel_time)
                if v_vis < Vp_max and v_vis > Vp_min:
                    # output_phase_file.write(line)
                    plot_x_good_p.append(e_s_dis)
                    plot_y_good_p.append(travel_time)
                    out_write_list.append(line)
                    P_NUM += 1

            if phase_type == 2:
                plot_x_ori_s.append(e_s_dis)
                plot_y_ori_s.append(travel_time)
                if v_vis < Vs_max and v_vis > Vs_min:
                    # output_phase_file.write(line)            
                    plot_x_good_s.append(e_s_dis)
                    plot_y_good_s.append(travel_time)
                    out_write_list.append(line)
                    S_NUM += 1

    if len(out_write_list) > 2:
        t_line = out_write_list[0]
        t_line = t_line[:86] + '{:4.0f} {:4.0f} {:4.0f}'.format(P_NUM,S_NUM,P_NUM+S_NUM) + t_line[94:]
        out_write_list[0] = t_line
        for write_line_out in out_write_list:
            output_phase_file.write(write_line_out)
    
    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    plt.title('Ori P NUM {}'.format(len(plot_x_ori_p)))
    plt.scatter(plot_x_ori_p,plot_y_ori_p,color='b')
    plt.plot([0,1200],[0,1200.0/Vp_max],label='Vp_max {:.2f} km'.format(Vp_max),color='k',linestyle='--')
    plt.plot([0,1200],[0,1200.0/Vp_min],label='Vp_min {:.2f} km'.format(Vp_min),color='k',linestyle='dotted')
    plt.legend(loc='upper left')
    plt.xlabel('Distance (km)')
    plt.ylabel('Travel Time (s)')
    plt.xlim([0,1200])
    plt.ylim([0,150])

    plt.subplot(2,2,3)
    plt.title('Ori S SUM {}'.format(len(plot_x_ori_s)))
    plt.xlim([0,1200])
    plt.ylim([0,260])
    plt.scatter(plot_x_ori_s,plot_y_ori_s,color='r')
    plt.plot([0,1200],[0,1200.0/Vs_max],label='Vs_max {:.2f} km'.format(Vs_max),color='k',linestyle='--')
    plt.plot([0,1200],[0,1200.0/Vs_min],label='Vs_min {:.2f} km'.format(Vs_min),color='k',linestyle='dotted')
    plt.legend(loc='upper left')
    plt.xlabel('Distance (km)')
    plt.ylabel('Travel Time (s)')

    plt.subplot(2,2,2)
    plt.title('Selected P NUM {}'.format(len(plot_x_good_p)))
    plt.xlim([0,1200])
    plt.ylim([0,150])
    plt.scatter(plot_x_good_p,plot_y_good_p,color='b')
    plt.plot([0,1200],[0,1200.0/Vp_max],label='Vp_max {:.2f} km'.format(Vp_max),color='k',linestyle='--')
    plt.plot([0,1200],[0,1200.0/Vp_min],label='Vp_min {:.2f} km'.format(Vp_min),color='k',linestyle='dotted')
    plt.legend(loc='upper left')
    plt.xlabel('Distance (km)')
    plt.ylabel('Travel Time (s)')
    
    plt.subplot(2,2,4)
    plt.title('Selected S NUM {}'.format(len(plot_x_good_s)))
    plt.scatter(plot_x_good_s,plot_y_good_s,color='r')
    plt.plot([0,1200],[0,1200.0/Vs_max],label='Vs_max {:.2f} km'.format(Vs_max),color='k',linestyle='--')
    plt.plot([0,1200],[0,1200.0/Vs_min],label='Vs_min {:.2f} km'.format(Vs_min),color='k',linestyle='dotted')
    plt.legend(loc='upper left')
    plt.xlim([0,1200])
    plt.ylim([0,260])
    plt.xlabel('Distance (km)')
    plt.ylabel('Travel Time (s)')

    plt.tight_layout()
    plt.savefig('discard_vis.png',dpi=500)
    plt.close()
"""