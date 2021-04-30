import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import sys
sys.path.append('../../S_EqT_codes/src/EqT_libs')
from hdf5_maker import preprocessor
from predictor import predictor
from EqT_utils import DataGeneratorPrediction, picker, generate_arrays_from_file
from EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
from keras.models import load_model
from keras.optimizers import Adam
import yaml
import os
from multiprocessing import  Pool
import argparse

def convert(cfgs):
    mseed_dir = cfgs['EqT']['mseed_dir']
    sta_json_path = cfgs['EqT']['sta_json_path']
    overlap = cfgs['EqT']['overlap']
    n_processor = cfgs['EqT']['n_processor']

    preprocessor(mseed_dir=mseed_dir,
                stations_json=sta_json_path,
                overlap=overlap,n_processor=n_processor)
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='01_download_data_from_IRIS')
    parser.add_argument('--config-file', dest='config_file', 
                        type=str, help='Configuration file path',default='./default_pipline_config.yaml')
    args = parser.parse_args()
    cfgs = yaml.load(open(args.config_file,'r'),Loader=yaml.SafeLoader)
    
    # convert mseed data to hdf5
    convert(cfgs)
    
    # earthquake detection and phase picking by the EqT model
    os.environ['CUDA_VISIBLE_DEVICES']  = cfgs['EqT']['gpuid']

    print(' *** Loading the model ...', flush=True)        
    model = load_model(cfgs['EqT']['model_path'],
                       custom_objects={'SeqSelfAttention': SeqSelfAttention, 
                                       'FeedForward': FeedForward,
                                       'LayerNormalization': LayerNormalization, 
                                       'f1': f1                                                                            
                                        })

    model.compile(loss = ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                  loss_weights = [0.03, 0.40, 0.58],       
                  optimizer = Adam(lr = 0.001),
                  metrics = [f1])
    print('Done Loading Model')
    
    predictor(input_dir= cfgs['EqT']['mseed_dir'] + '_processed_hdfs/',
            input_model=model,
            output_dir=cfgs['EqT']['det_res'],
            estimate_uncertainty=False, 
            output_probabilities=False,
            number_of_sampling=cfgs['EqT']['number_of_sampling'],
            loss_weights=[0.02, 0.40, 0.58],          
            detection_threshold=cfgs['EqT']['EQ_threshold'],                
            P_threshold=cfgs['EqT']['P_threshold'],
            S_threshold=cfgs['EqT']['S_threshold'], 
            number_of_plots=0,
            plot_mode='time',
            batch_size=500,gpuid=0,
            number_of_cpus=4,
            keepPS=True,
            spLimit=60)