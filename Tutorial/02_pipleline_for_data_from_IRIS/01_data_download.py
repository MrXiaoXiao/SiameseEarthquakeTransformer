"""
Download data from IRIS using obspy and EqT libraries
"""
import sys
sys.path.append('../../S_EqT_codes/src/EqT_libs')
from downloader import makeStationList, downloadMseeds
import yaml
import argparse
import os
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='01_download_data_from_IRIS')
    parser.add_argument('--config-file', dest='config_file', 
                        type=str, help='Configuration file path',default='./default_pipline_config.yaml')
    args = parser.parse_args()
    cfgs = yaml.load(open(args.config_file,'r'),Loader=yaml.SafeLoader)

    # set download params
    MINLON=cfgs['DataDownload']['minlon']
    MAXLON=cfgs['DataDownload']['maxlon']
    MINLAT=cfgs['DataDownload']['minlat']
    MAXLAT=cfgs['DataDownload']['maxlat']
    STIME=cfgs['DataDownload']['start_time']
    ETIME=cfgs['DataDownload']['end_time']
    CHANLIST=cfgs['DataDownload']['channel_list']
    FILTERNETWORK=cfgs['DataDownload']['exclude_network']
    CLIENTLIST=cfgs['DataDownload']['client_list']
    STAJSONPATH=cfgs['DataDownload']['sta_json_path']
    DATASAVEPATH=cfgs['DataDownload']['data_save_path']
    
    makeStationList(client_list=["IRIS"],  
                    min_lat=MINLAT,
                    max_lat=MAXLAT,
                    min_lon=MINLON, 
                    max_lon=MAXLON,                      
                    start_time=STIME, 
                    end_time=ETIME,
                    channel_list=CHANLIST,
                    filter_network=FILTERNETWORK,
                    filter_station=[])

    downloadMseeds(client_list=CLIENTLIST, 
            stations_json=STAJSONPATH, 
            output_dir=DATASAVEPATH, 
            start_time=STIME, 
            end_time=ETIME, 
            min_lat=MINLAT, 
            max_lat=MAXLAT, 
            min_lon=MINLON, 
            max_lon=MAXLON,
            chunk_size=1,
            channel_list=CHANLIST,
            n_processor=2)
    
    # remove empty folders
    mseed_path = Path(DATASAVEPATH)
    for sub_path in mseed_path.glob('*'):
        if len(list(sub_path.glob('*'))) == 0:
            print('Remove Empty Folder: {}'.format(str(sub_path)))
            os.rmdir(str(sub_path))
        