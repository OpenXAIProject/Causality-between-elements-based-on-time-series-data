import tensorflow as tf 
from retain_asym_model import *
from data_loader import *
import tensorflow as tf


config = {}
config['model_tag'] = 'multitask_asym_lastlayer_sqrtn'

tag = '_prediction_1_2002_2013_'
config['tag'] = tag
t = tag.split('_')

# Data info
config['data_path'] = '_'
config['diseases'] = ['cerebralinfarction','anginapectoris','myocardialinfarction']
config['num_features'] = 73
config['steps'] = int(t[4]) - int(t[3]) - int(t[2]) +1

config['tasks'] = [disease + tag for disease in config['diseases']]
config['num_tasks'] = len(config['tasks'])

# Model info
config['num_layers'] = 1
config['hidden_units'] = 16
config['embed_size'] = 16
config['lr'] = 1e-3
config['batch_size'] = 128
config['check_iter'] = 200
config['total_iter'] = 1000
config['save_iter'] = 5000

config['save_dir'] = '/checkpoints/'

config['asym_mu'] = 0.005
config['asym_lambda'] = 0.005
config['ld_l2'] = [0.0]*config['num_tasks']

def main():
    config['train_x'] = []
    config['train_y'] = []
    config['eval_x'] = []
    config['eval_y'] = []
    for disease in config['diseases']:
        path = config['data_path'] + '/' + disease + '/'
        file_name = disease + tag + 'train.txt'
        t_x, t_y = load_data(config['num_features'],config['steps'],path,file_name)
        config['train_x'].append(t_x)
        config['train_y'].append(t_y)
        

        file_name = disease + tag + 'test.txt'
        e_x, e_y = load_data(config['num_features'],config['steps'],path,file_name)
        config['eval_x'].append(e_x)
        config['eval_y'].append(e_y)


    #GPU Option
    gpu_usage = 0.95
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    config['sess'] = sess


    with tf.Session() as sess:

        model = RETAIN_ASYM(config)
        model.build_model()
        model.run()


if __name__ == '__main__':
    main()
