import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from random import shuffle

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