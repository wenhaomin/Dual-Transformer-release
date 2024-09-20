# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

import torch.utils.checkpoint as checkpoint
import time
from einops import repeat
from einops import rearrange
from torch.nn import functional as F


from algorithm.dataset_mask_one import KNOWN_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN

class PositionalEncode(nn.Module):
    """Non-learnable positional encoding layer proposed in the Transformer.
    """

    def __init__(self, hidden_size):
        super(PositionalEncode, self).__init__()

        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        B, L = pos_seq.shape
        sinusoid_inp = torch.ger(rearrange(pos_seq, 'B L -> (B L)'), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = rearrange(pos_emb, '(B L) E -> B L E', B=B, L=L)

        return pos_emb





class DualTransformer(nn.Module):
    """
    Transformer for Learning the patterns in human mobility and anomaly detection,
    It can process both continous feature and discrete features
    """
    def __init__(self, args: dict):
        super().__init__()

        self.args = args

        self.d_h = args['d_h'] # hidden size for the transformer
        self.num_head = args['num_head'] # number of heads for transformer
        self.num_layer = args['num_layer'] # number of the layers for transformer

        self.feature_list = args['feature_list']
        self.con_feature_list = [f for f in self.feature_list if f['type'] == 'continuous']
        self.con_feature_idx = [f['idx'] for f in self.con_feature_list]
        self.position2idx = args['position2idx']

        self.dis_feature_list = [f for f in self.feature_list if f['type'] == 'discrete']

        self.feature2idx = args['feature2idx']

        # Embedding the continous features together, d_c is the number of the continous features
        self.d_c = len(self.con_feature_list)
        self.emb_con = nn.Linear(self.d_c, self.d_c * self.d_h)

        # user embedding
        u_emb_size =  10
        # self.emb_user = nn.Sequential(nn.LayerNorm(u_emb_size),
        #                               nn.Linear(u_emb_size, self.d_h))
        self.emb_user = nn.Linear(u_emb_size, self.d_h)


        # Embedding the discrete features
        emb_size = 10
        self.emb_dis = []

        for f in self.dis_feature_list:
            class_num = f['class_num']
            emb_f = nn.Sequential(nn.Embedding(class_num, emb_size), nn.Linear(emb_size, self.d_h))
            self.emb_dis.append(emb_f)
        self.emb_dis = nn.ModuleList(self.emb_dis)

        # pos encoding
        self.pos_encode_layer = PositionalEncode(self.d_h)

        # decoding continous features from hidden space
        self.dec_con = nn.Linear(self.d_h, self.d_c)

        # decoding discrete features
        self.dec_dis = []
        for f in self.dis_feature_list:
            class_num = f['class_num']
            dec_f = nn.Linear(self.d_h, class_num)
            self.dec_dis.append(dec_f)
        self.dec_dis = nn.ModuleList(self.dec_dis)

        self.feature_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.d_h, nhead=self.num_head, batch_first=True), num_layers=self.num_layer)

        self.event_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.d_h, nhead=self.num_head, batch_first=True), num_layers=self.num_layer)

    def forward(self, input_seq, position, u, return_raw = True):
        """
        The forward function
        params:
          input_seq (torch.FloatTensor): contains the input feature
             (B, L, F),  B is the batch size, L is the sequence length, F is the feature size in one tuple

          position (torch.Long tensor):
             (B, L)
             the positional encoding

          u: (torch.FloatTensor)
            (B, d_u), d_u: user_embedding size
        
        return_raw: bool
            True: we also return the probability over all the discrete class
            False: directly return the predicted class
        """

        input_x = input_seq[:,:,:, 0]
        token = input_seq[:, :, :, 1]

        B, L, _ = input_x.shape


        # embed continous features
        con_x = input_x[:, :, self.con_feature_idx]
        con_e = self.emb_con(con_x) #(B, L, d_c * d_h)
        con_e = repeat(con_e, 'B L (d_c d_h) -> B L d_c d_h', d_c=self.d_c) # (B, L, d_c, d_h)

        # embed discrete features
        dis_e = []
        for i, f in enumerate(self.dis_feature_list):
            f_idx = f['idx']
            f_e = self.emb_dis[i](input_x[:, :,  f_idx].long())
            dis_e.append(f_e)
        dis_e = torch.stack(dis_e, dim=2) # # (B, L, d_d, d_h)
        e = torch.concatenate([con_e, dis_e], dim=2) # # (B, L, F, d_h)

        # intra transformer encoding
        e = repeat(e, 'B L F d_h -> (B L) F d_h')

        e1 = self.feature_transformer(e) # note: for feature transformer, there is no need to set the model src mask

        e2 = torch.sum(e1, dim=1) # e1: (B L) F d_h -> (B L) d_h, the event embedding

        e2 = repeat(e2, '(B L) d_h -> B L d_h', B = B)


        # embed user information
        u = u.float()
        user_e = self.emb_user(u)
        user_e = repeat(user_e, 'B d_h -> B L d_h', L = L)

        # positional embedding
        seq_pos = position[..., self.position2idx['seq_pos']]
        seq_pos = seq_pos.long()
        pos_e = self.pos_encode_layer(seq_pos) # B L d_h


        # position = position.long()
        # pos_e = self.pos_encode_layer(position) # B L d_h

        # event transformer
        # init hidden state
        h = e2

        if self.args['pos_mode'] != 'no_use':
            h += pos_e

        if self.args['user_mode'] == 'before_transformer':
            h += user_e

        src_key_padding_mask = token[..., 0] == PAD_TOKEN  # for event
        h = self.event_transformer(h, src_key_padding_mask=src_key_padding_mask) # (B, L, d_h) --> (B, L, d_h)

        if self.args['user_mode'] == 'after_transformer':
            h += user_e

        # decode all continous features
        pred_con = self.dec_con(h)

        # decode all discrete features
        pred_dis = []
        for i, f in enumerate(self.dis_feature_list):
            pred_f = self.dec_dis[i](h)
            if not return_raw:
                pred_f = torch.argmax(pred_f, 2).unsqueeze(-1).long()
            pred_dis.append(pred_f)

        # # decode poi
        # pred_poi = self.dec_poi(h)
        #
        # # decode dow
        # pred_dow = self.dec_dow(h)

        if return_raw:
            # note: make sure the return here follows the order in DISCRETE_FEATURE_NAME
            # pred_dis = [pred_poi, pred_dow]
            return pred_con, pred_dis
        else:
            # if no need to return the probability, then return the combined prediction
            # pred_poi = torch.argmax(pred_poi, 2).unsqueeze(-1).long()
            # pred_dow = torch.argmax(pred_dow, 2).unsqueeze(-1).long()

            # pred_dis = [pred_poi, pred_dow]
            pred = [pred_con] + pred_dis
            pred = torch.cat(pred, dim=-1)
            return pred


    def loss(self, input_seq, target_seq, position, u, uid):
        """
        Args:
            input_seq: (torch.FloatTensor),  shape:  (B, L, F, 2), the input features.
            target_seq: (torch.FloatTensor), shape: (B, L, F, 2),  the generation target features.
            positions: (torch.FloatTensor), shape:  (B, L), positional encoding
            u: (torch.FloatTensor), shape:  (B, d_u), user embedding
            uid: user id: (currently no use)

        Returns:
            loss value
        """


        # get the target token, for calculating the loss
        target_token = target_seq[..., 1].long() # (B, L, F)
        feature_mask = target_token != UNKNOWN_TOKEN  # (B, L, F)

        # get both continous prediction and discrete prediction
        pred_con, pred_dis = self.forward(input_seq, position, u)

        # loss for continous features
        target_con = target_seq[:, :, :self.d_c, 0]  # (B, L, d_c)

        loss_con = F.mse_loss(pred_con, target_con, reduction='none')

        loss_dict = {}
        for f in self.con_feature_list:
            f_idx, f_name = f['idx'], f['name']
            loss_f = masked_mean(loss_con[..., f_idx], feature_mask[..., f_idx])
            loss_dict[f_name] = loss_f



        rebalance_weight = 0.01
        # loss for discrete features
        # for f_idx, f_name, f_pred in zip(DISCRETE_FEATURE_IDX, DISCRETE_FEATURE_NAME, pred_dis):
        for f, f_pred in zip(self.dis_feature_list, pred_dis):
            f_idx, f_name = f['idx'], f['name']
            f_target = target_seq[:, :, f_idx, 0].long()
            f_target = rearrange(f_target, 'B L -> (B L)') # we need to rearrange the tensor to fit the input format of CE loss

            f_mask = feature_mask[..., f_idx]
            f_mask = rearrange(f_mask, 'B L -> (B L)')

            f_pred = rearrange(f_pred, 'B L N -> (B L) N')

            loss_f = F.cross_entropy(f_pred, f_target, reduction='none')

            loss_f =  masked_mean(loss_f, f_mask)

            loss_dict[f_name] =  rebalance_weight * loss_f
            
            
        ## sum all loss for continous features and discreate 
        loss_total = sum([v for k, v in loss_dict.items()])
        loss_value_dict = {k: v.item() for k, v in loss_dict.items()}
        loss_value_dict['total'] = loss_total.item()
        return loss_total, loss_value_dict

    def model_file_name(self):
        file_name = '+'.join([f'{k}-{self.args[k]}' for k in ['d_h']])
        file_name = f'vanilla__dis_transformer_-{file_name}'
        return file_name


def masked_mean(values, mask):
    values = values.masked_fill(mask, 0).sum()
    count = (~mask).long().sum()
    return values / count



# ---Log--
from utils.util import save2file_meta, ws
def save2file(params):
    file_name = ws + f'/output/metrics/dual_transformer.csv'
    head = [
        # data setting
        'dataset', 'seed',
        # model parameters
        'model', 'd_h', 'num_head', 'num_layer', 'user_mode', 'pos_mode',
        # evalution setting

        # training set
        'batch_size', 'lr', 'wd', 'early_stop', 'is_test', 'log_time', 'best_epoch', 'best_metric',

        'model_path', 'log_path', 'forecast_path',
    ]

    # metric result
    from utils.eval import Metric
    # from data.dataset import FEATURE_LIST

    metric_lst = ['test_time'] + list(Metric(params['feature_list']).metrics.keys())
    head = metric_lst + head

    save2file_meta(params,file_name,head)


if __name__ == '__main__':
    pass
