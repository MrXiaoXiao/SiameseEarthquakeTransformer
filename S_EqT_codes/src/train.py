from src.S_EqT_model import S_EqT_Model_create
from src.S_EqT_model import S_EqT_Model_seprate
from src.S_EqT_model import S_EqT_RSRN_Model
from src.S_EqT_model import S_EqT_HED_Model
from src.S_EqT_concate_fix_corr import S_EqT_Concate_RSRN_Model
from src.misc import get_train_list_v2
import numpy as np
import pandas as pd
import h5py
import keras.backend
import matplotlib.pyplot as plt

# get 
def train_debug(cfgs):
    keras.backend.set_floatx('float32')
    pick_type = int(cfgs['Model']['PickType'])
    noise_type = int(cfgs['Model']['Noise_type'])
    min_rate = float(cfgs['Model']['Noise_min_rate'])
    max_rate = float(cfgs['Model']['Noise_max_rate'])
    if noise_type == 2:
        noise_csv = pd.read_csv(cfgs['Model']['Noise_csv_path'])
        noise_keys = noise_csv['trace_name']
        noise_f = h5py.File(cfgs['Model']['Noise_data_path'], 'r')

    if_encoder_concate = int(cfgs['Model']['Encoder_concate'])
    if if_encoder_concate == 1:
        encode_model, siamese_model, EqT_model = S_EqT_Concate_RSRN_Model(cfgs)
        encoder_encoded_list = cfgs['Model']['Encoder_concate_list']
        encoder_encoded_lengths = cfgs['Model']['Encoder_concate_lengths']
        encoder_encoded_channels = cfgs['Model']['Encoder_concate_channels']
        
    elif int(cfgs['Model']['MODEL_RSRN'] == 0):
        encode_model, siamese_model, EqT_model = S_EqT_HED_Model(cfgs)
    else:
        encode_model, siamese_model, EqT_model = S_EqT_RSRN_Model(cfgs)
    
    if int(cfgs['Model']['LoadPretrainedModel']) == 1:
        siamese_model.load_weights(cfgs['Model']['PretrainModelPath'])
    # load h5py file
    # use with to aviod memeory leak
    # training params
    STEAD_base_dir =  cfgs['Train']['STEAD_path']
    train_dict = np.load(STEAD_base_dir +  cfgs['Train']['Train_dict_path'],allow_pickle=True)[()]
    train_keys = list(train_dict.keys())
    train_E_num = len(train_keys)

    print('Train E Num: {}'.format(train_E_num))

    train_set_path = STEAD_base_dir
    step_num = 0
    
    print_interval = int(cfgs['Train']['Train_print_interval'])
    save_interval = int(cfgs['Train']['Model_save_interval'])
    save_path = cfgs['Train']['Model_save_path']

    RSRN_lengths = cfgs['Model']['RSRN_Encoded_lengths']
    RSRN_channels = cfgs['Model']['RSRN_Encoded_channels']

    h5py_chunk_sets = dict()
    for chunk_dx in range(2,7):
        h5py_chunk_sets[chunk_dx] = h5py.File(STEAD_base_dir+'chunk{}.hdf5'.format(chunk_dx),mode='r')

    for mdx in range(int(cfgs['Train']['Total_Steps'])):

        train_id = np.random.randint(low=0,high=train_E_num)
        name = train_dict[train_keys[train_id]]
        e_name = train_set_path + cfgs['Train']['e_sort_path'] + 'file{}.csv'.format(name)
        # generate inputs and labels
        input_list, label_list = get_train_list_v2(cfgs, h5py_chunk_sets, e_name)
        id_chosen = np.random.choice(len(input_list),2,replace=False)

        idx = id_chosen[0]
        jdx = id_chosen[1]

        # train
        # prepare input data
        data_t = input_list[idx]
        data_t -= np.mean(data_t, axis=0 ,keepdims=True)
        t_spt_t = float(label_list[idx][0]/6000.0)
        t_sst_t= float(label_list[idx][1]/6000.0)

        """
        max_data = np.max(data_t, axis=0, keepdims=True)
        max_data[max_data == 0] = 1
        data_t /= max_data
        """
        std_data = np.std(data_t, axis = 0, keepdims=True)
        std_data[std_data == 0] = 1
        data_t /= std_data

        data_t_in = np.zeros([1,6000,3])
        data_t_in[0,:,:] = data_t

        data_s = input_list[jdx]
        s_spt_t = label_list[jdx][0]
        s_sst_t= label_list[jdx][1]
        s_coda_end_t = label_list[jdx][2]

        data_s, s_spt_t, s_sst_t, s_coda_end_t = _shift_event(data_s,s_spt_t,s_sst_t,s_coda_end_t,0.8)

        if noise_type == 0:
            pass
        elif noise_type == 1:
            add_noise_or_not = np.random.randint(low=0,high=10)
            if add_noise_or_not < 5:
                noise = np.zeros_like(data_s)
                noise = np.random.uniform(low=np.min(data_s), high=np.max(data_s), size=np.shape(noise))
                rate = np.random.uniform(low=min_rate,high=max_rate)
                data_s = data_s + noise*rate
        elif noise_type == 2:
            noise_id_chosen = int(np.random.choice(len(noise_keys),1))
            noise = noise_f.get('data/'+noise_keys[noise_id_chosen])
            noise = np.array(noise)
            noise = np.reshape(noise,[1,6000,3])
            # scale noise
            noise /= np.max(np.abs(noise))
            noise *= np.max(np.abs(data_s))
            rate = np.random.uniform(low=min_rate,high=max_rate)
            data_s = data_s + noise*rate
        elif noise_type == 3:
            noise = np.zeros_like(data_s)
            noise = np.random.normal(loc=0.0, scale=np.max(data_s), size=np.shape(noise))
            rate = np.random.uniform(low=min_rate,high=max_rate)
            data_s = data_s + noise*rate

        data_s -= np.mean(data_s, axis=0 ,keepdims=True)
        """
        max_data = np.max(data_s, axis=0, keepdims=True)
        max_data[max_data == 0] = 1
        data_s /= max_data
        """
        std_data = np.std(data_s, axis = 0, keepdims=True)
        std_data[std_data == 0] = 1
        data_s /= std_data
        data_s_in = np.zeros([1,6000,3])
        data_s_in[0,:,:] = data_s

        encoded_t = encode_model.predict(data_t_in)
        encoded_s = encode_model.predict(data_s_in)

        spt = np.zeros([1,1])
        spt[0,0] = float(s_spt_t/6000.0)
        sst = np.zeros([1,1])
        sst[0,0] = float(s_sst_t/6000.0)
        coda_end = np.zeros([1,1])
        coda_end[0,0] = float(s_sst_t/6000.0)

        siamese_input_list = list()
        siamese_label_list = list()

        if pick_type == 1:
            sst_f = s_sst_t
        elif pick_type == 0:
            sst_f = s_spt_t

        sst = np.zeros([1,6000,1,1])
        sst[0, sst_f-20:sst_f+21,0, 0] = _label()

        try:
            for rdx in range(len(RSRN_lengths)):
                temp_length = float(RSRN_lengths[rdx])
                template_s = int(t_spt_t*temp_length) - 1
                template_e = int(t_sst_t*temp_length) + 1
                template_w = int(template_e - template_s)

                encoded_t[rdx] = encoded_t[rdx][:,template_s:template_e,:]/float(template_w)
                encoded_t[rdx] = encoded_t[rdx].reshape([1,template_w,1,int(RSRN_channels[rdx])])
                encoded_s[rdx] = encoded_s[rdx].reshape([1,int(RSRN_lengths[rdx]),1,int(RSRN_channels[rdx])])

                # channel normalization
                for channel_dx in range(int(RSRN_channels[rdx])):
                    encoded_s[rdx][0,:,0,channel_dx] -= np.max(encoded_s[rdx][0,:,0,channel_dx])
                    half_window_len = int( 200.0*temp_length/6000.0   ) + 1
                    #window_mean = np.mean(encoded_s[rdx][0,half_window_len:-half_window_len,0,channel_dx])

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
                siamese_label_list.append(sst)
            #print('RSRN Channel_normal OK')
            if if_encoder_concate == 1:
                for rdx in range(len(RSRN_lengths), len(RSRN_lengths) + len(encoder_encoded_list)):
                    rdx_2 = rdx - len(RSRN_lengths) 
                    temp_length = float(encoder_encoded_lengths[rdx_2])
                    template_s = int(spt[0,0]*temp_length) - 1
                    template_e = int(coda_end[0,0]*temp_length) + 1
                    template_w = int(template_e - template_s)
                    #print('Concate 1 OK')
                    encoded_t[rdx] = encoded_t[rdx][:,template_s:template_e,:]/float(template_w)
                    encoded_t[rdx] = encoded_t[rdx].reshape([1,template_w,1,int(encoder_encoded_channels[rdx_2])])
                    encoded_s[rdx] = encoded_s[rdx].reshape([1,int(encoder_encoded_lengths[rdx_2]),1,int(encoder_encoded_channels[rdx_2])])
                    #print('Concate 2 OK')
                    # channel normalization
                    for channel_dx in range(int(encoder_encoded_channels[rdx_2])):
                        encoded_s[rdx][0,:,0,channel_dx] -= np.max(encoded_s[rdx][0,:,0,channel_dx])
                        half_window_len = int( 200.0*temp_length/6000.0   ) + 1
                        #window_mean = np.mean(encoded_s[rdx][0,half_window_len:-half_window_len,0,channel_dx])
                        #print('Concate 3 OK')
                        encoded_s[rdx][0,:half_window_len,0,channel_dx] = encoded_s[rdx][0,half_window_len,0,channel_dx]
                        encoded_s[rdx][0,-half_window_len:,0,channel_dx] =  encoded_s[rdx][0,-half_window_len,0,channel_dx]
                        #print('Concate 4 OK')
                        encoded_s[rdx][0,:,0,channel_dx] *= -1.0
                        encoded_s[rdx][0,:,0,channel_dx] -= np.mean(encoded_s[rdx][0,:,0,channel_dx])
                        t_max = np.max(np.abs(encoded_s[rdx][0,:,0,channel_dx]))
                        if t_max < 0.001:
                            t_max = 1
                        encoded_s[rdx][0,:,0,channel_dx] /= t_max
                        #print('Concate 5 OK')
                        encoded_t[rdx][0,:,0,channel_dx] -= np.max(encoded_t[rdx][0,:,0,channel_dx])
                        encoded_t[rdx][0,:,0,channel_dx] *= -1.0
                        encoded_t[rdx][0,:,0,channel_dx] -= np.mean(encoded_t[rdx][0,:,0,channel_dx])
                        t_max = np.max(np.abs(encoded_t[rdx][0,:,0,channel_dx]))
                        if t_max < 0.001:
                            t_max = 1
                        encoded_t[rdx][0,:,0,channel_dx] /= t_max
                        encoded_t[rdx][0,:,0,channel_dx] /= float(template_w)
                        #print('Concate 6 OK')
                    siamese_input_list.append(encoded_t[rdx])
                    siamese_input_list.append(encoded_s[rdx])
                    #print('Concate 7 OK')                        
        except:
            print('BAD DATA')
            #print(np.shape(encoded_t))
            #print(np.shape(data_t))
            #print(template_s)
            #print(template_e)
            #print(template_w)
            continue

        siamese_label_list.append(sst)

        if step_num % print_interval == 0:
            pred_res = siamese_model.predict(siamese_input_list)
            eqt_res = EqT_model.predict(data_s_in)

        hist = siamese_model.train_on_batch(x = siamese_input_list, y = siamese_label_list)

        if step_num % print_interval == 0:
            if pick_type == 1:
                print('Step {}: Hist : {}'.format(step_num, hist))
                print('GT : {}'.format(s_sst_t))
                eqt_s_t = np.argmax(eqt_res[2][0,:,0])
                print('EQT Pred : {}'.format(eqt_s_t))
            elif pick_type == 0:
                print('Step {}: Hist : {}'.format(step_num, hist))
                print('GT : {}'.format(s_spt_t))
                eqt_s_t = np.argmax(eqt_res[1][0,:,0])
                print('EQT Pred : {}'.format(eqt_s_t))

            for stage_dx in range(len(siamese_label_list)):
                pred_s_t = np.argmax(pred_res[stage_dx][0,:,0,0])
                min_amp = np.min(pred_res[stage_dx][0,:,0,0])
                print('Stage ID: {} Pred S: {} Amp: {} MinAmp: {}'.format(stage_dx,pred_s_t,pred_res[stage_dx][0,pred_s_t,0,0],min_amp))
                
            
        if step_num % save_interval == 0:
            siamese_model.save(save_path+'model{}.hdf5'.format(step_num))
        step_num += 1

    noise_f.close()

    return

def train(cfgs):
    # load encode_model and siamese_model
    if int(cfgs['Model']['Sepearte']) == 1:
        print(int(cfgs['Model']['Sepearte']))
        print(cfgs['Model']['Sepearte'])
        encode_model, siamese_model = S_EqT_Model_seprate(cfgs)
    else:
        if int(cfgs['Model']['Cascade']) == 1:
            encode_model, siamese_model, feature_cal, cascade_1, cascade_2 = S_EqT_Model_create(cfgs)
            #TODO add cascade params
        else:
            encode_model, siamese_model = S_EqT_Model_create(cfgs)
    
    siamese_model.summary()
    # load h5py file
    h5File = h5py.File(cfgs['Train']['File_path'],mode='r')
    
    train_E_num = int(cfgs['Train']['Train_E_num'])
    train_set_path = cfgs['Train']['Train_data_path']
    step_num = 0

    print_interval = int(cfgs['Train']['Train_print_interval'])
    save_interval = int(cfgs['Train']['Model_save_interval'])
    save_path = cfgs['Train']['Model_save_path']

    p_width = int(cfgs['Model']['P_width'])
    s_width = int(cfgs['Model']['S_width'])
    coda_width = int(cfgs['Model']['Coda_width'])

    p_half = int( (p_width -1)/2 )
    s_half = int( (s_width -1)/2 )
    coda_half = int( (coda_width -1)/2 )

    channel_num = int(cfgs['Model']['Encode_channel'])
    Siamese_type = int(cfgs['Model']['Siamese_Type'])

    for mdx in range(int(cfgs['Train']['Total_Steps'])):
        train_id = np.random.randint(low=0,high=train_E_num)
        e_name = train_set_path + 'file{}'.format(train_id)
        # generate inputs and labels
        input_list, label_list = get_train_list(cfgs, h5File, e_name)
        id_chosen = np.random.choice(len(input_list),2,replace=False)

        idx = id_chosen[0]
        jdx = id_chosen[1]

        # train
        # prepare input data
        data_t = input_list[idx]
        data_t -= np.mean(data_t, axis=0 ,keepdims=True)
        
        max_data = np.max(data_t, axis=0, keepdims=True)
        max_data[max_data == 0] = 1
        data_t /= max_data
        data_t_in = np.zeros([1,6000,3])
        data_t_in[0,:,:] = data_t

        data_s = input_list[jdx]
        data_s -= np.mean(data_s, axis=0 ,keepdims=True)
        max_data = np.max(data_s, axis=0, keepdims=True)
        max_data[max_data == 0] = 1
        data_s /= max_data
        data_s_in = np.zeros([1,6000,3])
        data_s_in[0,:,:] = data_s

        encoded_t = encode_model.predict(data_t_in)
        encoded_s = encode_model.predict(data_s_in)

        # prepare label
        spt = np.zeros([1,1])
        spt[0,0] = float(label_list[jdx][0]/6000.0)
        sst = np.zeros([1,1])
        sst[0,0] = float(label_list[jdx][1]/6000.0)
        coda_end = np.zeros([1,1])
        coda_end[0,0] = float(label_list[jdx][2]/6000.0)        
        
        if int(cfgs['Model']['Sepearte']) == 1:
            # p_encoded
            p_encoded = np.zeros([1,p_width,channel_num])
            p_start = int(label_list[idx][0]*47.0/6000.0) - p_half
            p_end = int(label_list[idx][0]*47.0/6000.0) + p_half + 1
            if p_start < 0:
                p_encoded[:,-p_start:,:] = encoded_t[:,0:p_end,:] 
            else:
                p_encoded = encoded_t[:,p_start:p_end,:] 
            #p_encoded /= float(p_width)
            p_encoded = p_encoded.reshape([1,p_width,1,16])

            # s_encoded
            s_encoded = np.zeros([1,s_width,channel_num])
            s_start = int(label_list[idx][1]*47.0/6000.0) - s_half
            s_end = int(label_list[idx][1]*47.0/6000.0) + s_half + 1
            
            if s_start < 0:
                s_encoded[:,-s_start:,:] = encoded_t[:,0:s_end,:] 
            elif s_end > 47:
                s_stop = -1*(s_end - 47)
                s_encoded[:,:s_stop,:] = encoded_t[:,s_start:,:] 
            else:
                s_encoded = encoded_t[:,s_start:s_end,:] 
            #s_encoded /= float(p_width)
            s_encoded = s_encoded.reshape([1,s_width,1,16])

            # coda_enocded
            coda_encoded = np.zeros([1,coda_width,channel_num])
            coda_start = int(label_list[idx][2]*47.0/6000.0) - coda_half
            coda_end = int(label_list[idx][2]*47.0/6000.0) + coda_half + 1

            if coda_end > 47:
                coda_stop = -1*(coda_end - 47)
                coda_encoded[:,:coda_stop,:] = encoded_t[:,coda_start:,:] 
            else:
                coda_encoded = encoded_t[:,coda_start:coda_end,:] 
            #coda_encoded /= float(p_width)
            coda_encoded = coda_encoded.reshape([1,coda_width,1,16])        
        else:
            template_s = int(spt[0,0]*47)
            template_e = int(coda_end[0,0]*47) + 1
            template_w = int(template_e - template_s)
            encoded_t = encoded_t[:,template_s:template_e,:]
            encoded_t = encoded_t.reshape([1,template_w,1,16])

        # search feature map
        encoded_s = encoded_s.reshape([1,47,1,16])
                
        if Siamese_type == 0:
            res = siamese_model.predict([p_encoded, 
                                            s_encoded, 
                                            coda_encoded, 
                                            encoded_s])
            """
            plt.figure(figsize=(16,8))
            for pdx in range(16):
                plt.plot(res[0][0,0,:,pdx]*1.0+pdx*2.0,color='k')
            plt.savefig('res0.png',dpi=500)
            #plt.show()
            plt.close()

            plt.figure(figsize=(16,8))
            for pdx in range(16):
                plt.plot(res[1][0,0,:,pdx]*1.0+pdx*2.0,color='k')
            plt.savefig('res1.png',dpi=500)
            #plt.show()
            plt.close()

            plt.figure(figsize=(16,8))
            for pdx in range(16):
                plt.plot(res[2][0,0,:,pdx]*1.0+pdx*2.0,color='k')
            plt.savefig('res2.png',dpi=500)
            #plt.show()
            plt.close()
            break
            """

            naive_p = float(np.argmax(res[0][0,0,:,0])/6000.0)
            naive_s = float(np.argmax(res[1][0,0,:,0])/6000.0)
            naive_code_e = float(np.argmax(res[2][0,:,0,0])/6000.0)
            print('Real {:.5f} {:.5f} {:.5f}\nPred {:.5f} {:.5f} {:.5f}'.format(spt[0,0],sst[0,0],coda_end[0,0],naive_p,naive_s,naive_code_e))
        
        elif int(Siamese_type) == 1:
            if int(cfgs['Model']['Cascade']) == 1:
                if int(cfgs['Model']['Sepearte']) == 1:
                    pass
                else:
                    # train on Raw data
                    hist_raw = siamese_model.train_on_batch(x = [encoded_t, encoded_s],
                                                            y = [spt,sst,coda_end])
                    feature_corr = feature_cal.predict([encoded_t, encoded_s])
                    
                    # stage 1
                    zero_stage_p_res, zero_stage_s_res, empty = siamese_model.predict([encoded_t, encoded_s])

                    # slice feature corr according to current model
                    zero_stage_p_center = int(zero_stage_p_res[0,0]*47)
                    zero_stage_s_center = int(zero_stage_s_res[0,0]*47)
                    
                    if zero_stage_p_res < 0 or zero_stage_p_res > 1:
                        continue
                    if zero_stage_s_res < 0 or zero_stage_s_res > 1:
                        continue
                    
                    zero_stage_p = np.zeros([1,23,1,16])

                    if zero_stage_p_center - 11 < 0:
                        zero_stage_p[:,11-zero_stage_p_center:,:,:] = feature_corr[:,:zero_stage_p_center + 11 + 1,:,:]
                    elif zero_stage_p_center + 11 >= 47:
                        zero_stage_p[:,:11 + 47 - zero_stage_p_center,:,:] = feature_corr[:,zero_stage_p_center - 11:,:,:]
                    else:
                        zero_stage_p[:,:,:,:] = feature_corr[:,zero_stage_p_center - 11:zero_stage_p_center + 11 +1,:,:]
                    
                    zero_stage_p_label = np.zeros([1,1])
                    zero_stage_p_label[0,0] = 10.0 * (spt[0,0] - zero_stage_p_res[0,0])

                    zero_stage_s = np.zeros([1,23,1,16])

                    if zero_stage_s_center - 11 < 0:
                        zero_stage_s[:,11-zero_stage_s_center:,:,:] = feature_corr[:,:zero_stage_s_center + 11 + 1,:,:]
                    elif zero_stage_s_center + 11 >= 47:
                        zero_stage_s[:,:11 + 47 - zero_stage_s_center,:,:] = feature_corr[:,zero_stage_s_center - 11:,:,:]
                    else:
                        zero_stage_s[:,:,:,:] = feature_corr[:,zero_stage_s_center - 11:zero_stage_s_center + 11 +1,:,:]
                    
                    zero_stage_s_label = np.zeros([1,1])
                    zero_stage_s_label[0,0] = 10.0 * (sst[0,0] - zero_stage_s_res[0,0])

                    hist_stage_1 = cascade_1.train_on_batch(x = [zero_stage_p, zero_stage_s],
                                                            y = [zero_stage_p_label,zero_stage_s_label])

                    one_stage_p_res, one_stage_s_res = cascade_1.predict([zero_stage_p, zero_stage_s])

                    # stage 2
                    one_stage_p_center = int( (zero_stage_p_res[0,0] + (one_stage_p_res[0,0]/10.0) )*47)
                    one_stage_s_center = int( (zero_stage_s_res[0,0] + (one_stage_s_res[0,0]/10.0) )*47)
                    if one_stage_p_center < 0 or one_stage_p_center > 47:
                        continue
                    if one_stage_s_center < 0 or one_stage_s_center > 47:
                        continue
                    one_stage_p = np.ones([1,23,1,16])

                    if one_stage_p_center - 11 < 0:
                        one_stage_p[:,11-one_stage_p_center:,:,:] = feature_corr[:,:one_stage_p_center + 11 + 1,:,:]
                    elif one_stage_p_center + 11 >= 47:
                        one_stage_p[:,:11 + 47 - one_stage_p_center:,:] = feature_corr[:,one_stage_p_center - 11:,:,:]
                    else:
                        one_stage_p[:,:,:,:] = feature_corr[:,one_stage_p_center - 11:one_stage_p_center + 11 +1,:,:]
                    
                    one_stage_p_label = np.ones([1,1])
                    one_stage_p_label[0,0] = 10.0 * (spt[0,0] - zero_stage_p_res[0,0] - (one_stage_p_res[0,0]/10.0) )

                    one_stage_s = np.ones([1,23,1,16])

                    if one_stage_s_center - 11 < 0:
                        one_stage_s[:,11-one_stage_s_center:,:,:] = feature_corr[:,:one_stage_s_center + 11 + 1,:,:]
                    elif one_stage_s_center + 11 >= 47:
                        one_stage_s[:,:11 + 47 - one_stage_s_center,:,:] = feature_corr[:,one_stage_s_center - 11:,:,:]
                    else:
                        one_stage_s[:,:,:,:] = feature_corr[:,one_stage_s_center - 11:one_stage_s_center + 11 +1,:,:]
                    
                    one_stage_s_label = np.zeros([1,1])
                    one_stage_s_label[0,0] = 10.0 * (sst[0,0] - zero_stage_s_res[0,0] - (one_stage_s_res[0,0]/10.0) )

                    hist_stage_2 = cascade_2.train_on_batch(x = [one_stage_p, one_stage_s],
                                                            y = [one_stage_p_label,one_stage_s_label])

                    two_stage_p_res, two_stage_s_res = cascade_1.predict([one_stage_p, one_stage_s])
                    # print loss and pick
                    if step_num % print_interval == 0:
                        print('Hist Raw: {}'.format(hist_raw))
                        print('Hist Stage 1: {}'.format(hist_stage_1))
                        print('Hist Stage 2: {}'.format(hist_stage_2))
                        print('True：P: {} S: {}\nRaw: P: {} S: {}\nOne-Stage: P: {} S: {}\nTwo-Stage: P: {} S: {}\n'.format(
                            spt, sst, zero_stage_p_res, zero_stage_s_res, zero_stage_p_res + one_stage_p_res/10.0, zero_stage_s_res + one_stage_s_res/10.0,
                            zero_stage_p_res + one_stage_p_res/10.0 + two_stage_p_res/10.0, zero_stage_s_res + one_stage_s_res/10.0 + two_stage_s_res/10.0))

                    if step_num % save_interval == 0:
                        siamese_model.save(save_path+'modelSeg_{}.hdf5'.format(step_num))
                        cascade_1.save(save_path+'modelSeg_{}_ca1.hdf5'.format(step_num))
                        cascade_2.save(save_path+'modelSeg_{}_ca2.hdf5'.format(step_num))
            else:
                hist = siamese_model.train_on_batch(x = [p_encoded, s_encoded, coda_encoded, encoded_s],
                                                    y = [spt,sst,coda_end])
                

                if step_num % print_interval == 0:
                    res = siamese_model.predict([encoded_t,encoded_s])
                    print('Step {} Hist {}'.format(step_num, hist))
                    print('Real {:.5f} {:.5f} {:.5f}\nPred {:.5f} {:.5f} {:.5f}'.format(spt[0,0],sst[0,0],coda_end[0,0],
                                                                res[0][0,0],res[1][0,0],res[2][0,0]))
                if step_num % save_interval == 0:
                    siamese_model.save(save_path+'model_{}.hdf5'.format(step_num))
        
        elif Siamese_type == 2:

            spt = np.zeros([1,6000,1,1])
            spt[0,label_list[jdx][0]-2:label_list[jdx][0]+3,0,0] = 1
            sst = np.zeros([1,6000,1,1])
            sst[0,label_list[jdx][1]-2:label_list[jdx][1]+3,0,0] = 1
            coda_end = np.zeros([1,6000,1,1])
            coda_end[0,label_list[jdx][2]-2:label_list[jdx][2]+3,0,0] = 1

            hist = siamese_model.train_on_batch(x = [encoded_t, encoded_s],
                                                y = [spt,sst,coda_end])
            
            if step_num % print_interval == 0:
                res = siamese_model.predict([encoded_t,encoded_s])
                print('Step {} Hist {}'.format(step_num, hist))
                real_p = np.argmax(spt[0,:,0,0]) + 2 
                real_s = np.argmax(sst[0,:,0,0]) + 2
                real_e = np.argmax(coda_end[0,:,0,0]) + 2
                seg_p = np.argmax(res[0][0,:,0,0])
                seg_s = np.argmax(res[1][0,:,0,0])
                seg_coda_e = np.argmax(res[2][0,:,0,0])
                print('Real {} {} {}\nPred {} {} {}'.format(real_p,
                                                            real_s,
                                                            real_e,
                                                            seg_p,seg_s,
                                                            seg_coda_e))
            
            if step_num % save_interval == 0:
                siamese_model.save(save_path+'model_{}.hdf5'.format(step_num))

        step_num += 1
        # evaluate
    return

def _label(a=0, b=20, c=40):  
    'Used for triangolar labeling'
    
    z = np.linspace(a, c, num = 2*(b-a)+1)
    y = np.zeros(z.shape)
    y[z <= a] = 0
    y[z >= c] = 0
    first_half = np.logical_and(a < z, z <= b)
    y[first_half] = (z[first_half]-a) / (b-a)
    second_half = np.logical_and(b < z, z < c)
    y[second_half] = (c-z[second_half]) / (c-b)
    return y

def _shift_event(data, addp, adds, coda_end, rate): 
    'Randomly rotate the array to shift the event location'
    org_len = len(data)
    data2 = np.copy(data)
    addp2 = adds2 = coda_end2 = None;
    if np.random.uniform(0, 1) < rate:             
        nrotate = int(np.random.uniform(1, int(org_len - coda_end - 200)))
        data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
        data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
        data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]
                
        if addp+nrotate >= 0 and addp+nrotate < org_len:
            addp2 = addp+nrotate;
        else:
            addp2 = None;
        if adds+nrotate >= 0 and adds+nrotate < org_len:               
            adds2 = adds+nrotate;
        else:
            adds2 = None;                   
        if coda_end+nrotate < org_len:                              
            coda_end2 = coda_end+nrotate 
        else:
            coda_end2 = org_len                 
        if addp2 and adds2:
            data = data2;
            addp = addp2;
            adds = adds2;
            coda_end= coda_end2;                                      
    return data, addp, adds, coda_end  