# -*- coding: utf-8 -*-
from pprint import pprint
from utils.util import get_common_params, dict_merge, dir_check, Platforms
import random
import torch
import platform
def run(params):
    # pprint(params)
    model = params['model']

    if model == 'transformer':
        from algorithm.transformer.train import main
        return main(params)

    if model == 'transformer_uc':
        from algorithm.transfomer_uc.train import main
        return main(params)

    if model == 'dual_transformer':
        from algorithm.dual_transformer.train import main
        return main(params)

    if model == 'dual_transformer_uc':
        from algorithm.dual_transformer_uc.train import main
        return main(params)

def get_params():
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    params = vars(get_params())

    if platform.platform() == Platforms.la3:
        cuda_id = random.choice([1,2,3,4])
        device = f"cuda:{cuda_id}"
        torch.cuda.set_device(device)
    else:
        device = "cuda:0"  # for windows

    # add parameter
    seed = 2021
    is_test = False
    args_lst = []

    # common args
    params = vars(get_params())

    # record the position encoding in the input
    POSITION_LIST = [
        {'idx': 0, 'name': 'seq_pos'},
        {'idx': 1, 'name': 'within_day_pos'},
        {'idx': 2, 'name': 'day_pos'},
    ]

    model = 'transfomer'
    position2idx = {f['name']: f['idx'] for f in POSITION_LIST}
    common_params = {'model': model, 'user_mode': 'no_use', 'pos_mode': 'vanilla_pe', 'num_head': 4,
                     'dataset': '7_22_trial2_test', 'is_test': is_test, 'early_stop_start_epoch': 10, 'early_stop': 5, 'norm': 'minmax',
                     'batch_size': 128, 'workers': 8, 'seed': seed, 'position2idx': position2idx}
    '''
    Model-related parameter:
    user_mode: use the user embedding or not
    pos_mode:  use positional embeding or not. 
        - vanilla_pe: use positional embedding
        - no_use: no use of positional embeeding
    num_head:  number of head for transformer
    
    Training-related parameter
    early_stop_start_epoch: from which epoch to start evaluate the model in val dataset
    early_stop: the early stop patience
    norm: normalization method for features
    '''


    params = dict_merge([params, common_params])

    # feature setting
    FEATURE_LIST = [
        # continous
        {'idx': 0, 'name': 'x', 'type': 'continuous', 'is_predict': 'yes'},
        {'idx': 1, 'name': 'y', 'type': 'continuous', 'is_predict': 'yes'},
        {'idx': 2, 'name': 'start_time_minute', 'type': 'continuous', 'is_predict': 'yes'},
        {'idx': 3, 'name': 'stay_duration', 'type': 'continuous', 'is_predict': 'yes'},
        # discrete
        {'idx': 4, 'name': 'poi', 'type': 'discrete', 'class_num': 40, 'is_predict': 'yes'},
        {'idx': 5, 'name': 'dow', 'type': 'discrete', 'class_num': 7, 'is_predict': 'yes'},
    ]
    Feature2Idx = {f['name']: f['idx'] for f in FEATURE_LIST}
    NeedNormFeature = [f['name'] for f in FEATURE_LIST if f['type'] == 'continuous']
    params['feature_list'] = FEATURE_LIST
    params['feature2idx'] = Feature2Idx
    params['need_norm_feature'] = NeedNormFeature
    params['need_norm_feature_idx'] = [Feature2Idx[f] for f in NeedNormFeature]

    # paramter grid search
    for mask_ratio in [0.05]:
        for hidden_size in [64, 128]:
            for num_layer in [2]:
                temp_params = {'d_h': hidden_size,  'num_layer': num_layer, 'mask_ratio': mask_ratio, 'dropout': 0.05}
                temp_params = dict_merge([params, temp_params])
                args_lst.append(temp_params)

    # run all parameters
    for p in args_lst:
        run(p)
        print('work done')











