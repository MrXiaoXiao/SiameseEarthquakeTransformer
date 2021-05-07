import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import re
from matplotlib.ticker import FuncFormatter
import numpy as np
import yaml
from pathlib import Path
import argparse

def create_e_dict_from_sum_csv(csv_file_path, cfgs):
    lat_sign = cfgs['HypoInverse']['lat_sign']
    lon_sign = cfgs['HypoInverse']['lon_sign']
    csv_e_list = list()
    f = open(csv_file_path,'r')
    e_id = 0
    for line in f.readlines():
        e_id += 1
        codes = line.split()
        date, hrmn, sec = codes[0:3]
        dtime = date + hrmn + sec.zfill(5)
        time = dtime
        is_loc = 1 # whether loc reliable
        if '-' in codes or '#' in codes: is_loc = 0
        npha = float(line[52:55])
        azm  = float(line[56:59])
        rms  = float(line[64:69])
        erh = float(line[70:74])
        erz = float(line[75:79])
        lat_deg = float(line[20:22])
        lat_min = float(line[23:28])
        lat = lat_deg + lat_min/60
        lat *= lat_sign
        lon_deg = float(line[29:32])
        lon_min = float(line[33:38])
        lon = lon_deg + lon_min/60
        lon *= lon_sign
        dep = float(line[38:44])
        temp_list = [e_id,time,lat,lon,dep,is_loc,npha,azm,rms,erh,erz]
        csv_e_list.append(temp_list)
    f.close()
    return csv_e_list

def visualize_profile(cfgs, s_lat, s_lon, e_lat, e_lon, lat_list, lon_list, dep_list, name_flag = 'S_EqT'):
    
    dist_list = list()
    A = e_lat - s_lat
    B = s_lon - e_lon
    C = e_lon*s_lat - s_lon*e_lat
    
    MAX_Range = cfgs['VIS']['profile_range']

    dep_list_in_profile = list()
    MAX_X = np.sqrt( (e_lat - s_lat)**2 + (e_lon- s_lon)**2 ) * 111.0
    for idx in range(len(lat_list)):
        t_lon = lon_list[idx]
        t_lat = lat_list[idx]

        project_lon = (B*B*t_lon - A*B*t_lat - A*C)/(A*A + B*B)
        project_lat = (A*A*t_lat - A*B*t_lon - B*C)/(A*A + B*B)

        project_dist = np.sqrt((project_lon - t_lon)**2 + (project_lat - t_lat)**2 ) * 111.0
        if project_dist > MAX_Range:
            continue
        
        t_dist = np.sqrt( (project_lat -s_lat)**2 + (project_lon-s_lon)**2 ) * 111.0
        dist_list.append(t_dist)
        dep_list_in_profile.append( dep_list[idx] )
    vis_key = cfgs['VIS']['VIS_key']
    MAXDEP = cfgs['VIS']['MAXDEP']
    ratio = 1
    plt.figure(figsize=(12,4))
    plt.scatter(dist_list,dep_list_in_profile)
    plt.title('{} Profile {} earthquakes TASK {}'.format(name_flag, len(dist_list),vis_key))
    plt.xlim([0,MAX_X])
    plt.ylim([MAXDEP,0.0])
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.gca().set_aspect(ratio)
    plt.savefig('./VISRes/{}_Profile.png'.format(name_flag),dpi=300)
    plt.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='05_simple_visualization')
    parser.add_argument('--config-file', dest='config_file', 
                        type=str, help='Configuration file path',default='./default_pipline_config.yaml')
    args = parser.parse_args()
    cfgs = yaml.load(open(args.config_file,'r'),Loader=yaml.SafeLoader)

    # visualize EqT results         
    vis_key = cfgs['VIS']['VIS_key']
    e_list = create_e_dict_from_sum_csv('./catalogs/eqt_hypoInverse.sum', cfgs)

    MINLAT = cfgs['VIS']['MINLAT']
    MAXLAT = cfgs['VIS']['MAXLAT']
    MINLON = cfgs['VIS']['MINLON']
    MAXLON = cfgs['VIS']['MAXLON']
    MAXERH = cfgs['VIS']['MAXERH']
    MAXERZ = cfgs['VIS']['MAXERZ']
    MAXAZM = cfgs['VIS']['MAXAZM']
    MAXDEP = cfgs['VIS']['MAXDEP']

    PRO_SX = cfgs['VIS']['profile_start_lon']
    PRO_SY = cfgs['VIS']['profile_start_lat']
    PRO_EX = cfgs['VIS']['profile_end_lon']
    PRO_EY = cfgs['VIS']['profile_end_lat']

    e_good_list = list()
    for e in e_list:
        if e[2] > MINLAT and e[2] < MAXLAT and e[3] > MINLON and e[3] < MAXLON:
            pass
        else:
            continue

        if e[4] > MAXDEP:
            continue

        if e[-2] > MAXERH or e[-1] > MAXERZ or e[-4] > MAXAZM:
            continue

        e_good_list.append(e)

    e_lat = list()
    e_lon = list()
    e_dep = list()

    for e in e_good_list:
        e_lat.append(e[2])
        e_lon.append(e[3])
        e_dep.append(e[4])

    plt.figure(figsize=(10,10))
    m = Basemap(llcrnrlon=MINLON,llcrnrlat=MINLAT,urcrnrlon=MAXLON,urcrnrlat=MAXLAT)
    m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1000, verbose= True)

    plt.title('EqT {} earthquakes TASK {}'.format(len(e_lon), vis_key))
    plt.plot([PRO_SX,PRO_EX],[PRO_SY,PRO_EY],color='b', linewidth=1.5)
    plt.scatter(e_lon,e_lat,color='r',label='Detected Earthquakes')
    plt.xlim([MINLON,MAXLON])
    plt.ylim([MINLAT,MAXLAT])
    m.drawparallels(np.arange(2,8,1.0),labels=[0,1,0,0],fontsize=14)
    m.drawmeridians(np.arange(114,120,1.0),labels=[0,0,0,1],fontsize=14)
    plt.legend()
    plt.savefig('./VISRes/eqt_HypoInverse_Horizontal.png',dpi=300)
    plt.close()

    ratio = 1/111.0

    plt.figure(figsize=(12,4))
    plt.scatter(e_lon,e_dep)
    plt.title('EqT {} earthquakes TASK {}'.format(len(e_lon), vis_key))
    plt.xlim([MINLON,MAXLON])
    plt.ylim([MAXDEP,0.0])
    plt.xlabel('Longitude')
    plt.ylabel('Depth (km)')
    plt.gca().set_aspect(ratio)
    plt.savefig('./VISRes/eqt_HypoInverse_W-E_Vertical.png',dpi=300)
    plt.close()

    plt.figure(figsize=(12,4))
    plt.scatter(e_lat,e_dep)
    plt.title('EqT {} earthquakes TASK {}'.format(len(e_lat),vis_key))
    plt.xlim([MAXLAT,MINLAT])
    plt.ylim([MAXDEP,0.0])
    plt.xlabel('Latitude')
    plt.ylabel('Depth (km)')
    plt.gca().set_aspect(ratio)
    plt.savefig('./VISRes/eqt_HypoInverse_N-S_Vertical.png',dpi=300)
    plt.close()

    visualize_profile(cfgs, PRO_SY, PRO_SX, PRO_EY, PRO_EX, e_lat, e_lon, e_dep, name_flag = 'EqT')
    
    # visualize S-EqT results         
    vis_key = cfgs['VIS']['VIS_key']
    e_list = create_e_dict_from_sum_csv('./catalogs/seqt_hypoInverse.sum', cfgs)

    e_good_list = list()
    for e in e_list:
        if e[2] > MINLAT and e[2] < MAXLAT and e[3] > MINLON and e[3] < MAXLON:
            pass
        else:
            continue

        if e[4] > MAXDEP:
            continue

        if e[-2] > MAXERH or e[-1] > MAXERZ or e[-4] > MAXAZM:
            continue

        e_good_list.append(e)

    e_lat = list()
    e_lon = list()
    e_dep = list()

    for e in e_good_list:
        e_lat.append(e[2])
        e_lon.append(e[3])
        e_dep.append(e[4])

    plt.figure(figsize=(10,10))
    m = Basemap(llcrnrlon=MINLON,llcrnrlat=MINLAT,urcrnrlon=MAXLON,urcrnrlat=MAXLAT)
    m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1000, verbose= True)

    plt.title('S-EqT {} earthquakes TASK {}'.format(len(e_lon), vis_key))
    plt.plot([PRO_SX,PRO_EX],[PRO_SY,PRO_EY],color='b', linewidth=1.5)
    plt.scatter(e_lon,e_lat,color='r',label='Detected Earthquakes')
    plt.xlim([MINLON,MAXLON])
    plt.ylim([MINLAT,MAXLAT])
    m.drawparallels(np.arange(2,8,1.0),labels=[0,1,0,0],fontsize=14)
    m.drawmeridians(np.arange(114,120,1.0),labels=[0,0,0,1],fontsize=14)
    plt.legend()
    plt.savefig('./VISRes/seqt_HypoInverse_Horizontal.png',dpi=300)
    plt.close()

    ratio = 1/111.0

    plt.figure(figsize=(12,4))
    plt.scatter(e_lon,e_dep)
    plt.title('S-EqT {} earthquakes TASK {}'.format(len(e_lon), vis_key))
    plt.xlim([MINLON,MAXLON])
    plt.ylim([MAXDEP,0.0])
    plt.xlabel('Longitude')
    plt.ylabel('Depth (km)')
    plt.gca().set_aspect(ratio)
    plt.savefig('./VISRes/seqt_HypoInverse_W-E_Vertical.png',dpi=300)
    plt.close()

    plt.figure(figsize=(12,4))
    plt.scatter(e_lat,e_dep)
    plt.title('S-EqT {} earthquakes TASK {}'.format(len(e_lat),vis_key))
    plt.xlim([MAXLAT,MINLAT])
    plt.ylim([MAXDEP,0.0])
    plt.xlabel('Latitude')
    plt.ylabel('Depth (km)')
    plt.gca().set_aspect(ratio)
    plt.savefig('./VISRes/seqt_HypoInverse_N-S_Vertical.png',dpi=300)
    plt.close()

    visualize_profile(cfgs, PRO_SY, PRO_SX, PRO_EY, PRO_EX, e_lat, e_lon, e_dep, name_flag = 'S_EqT')
    