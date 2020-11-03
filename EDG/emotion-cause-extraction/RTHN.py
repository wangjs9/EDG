import sys
sys.path.append("..")

import warnings
warnings.filterwarnings('ignore')

import torch, os
import torch.nn as nn
import numpy as np
from transformers import BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.common_layer import RTHNLayer, _gen_bias_mask

class RTHN(nn.Module):
    def __init__(self, model_dir, embed_dim, pos_embed_dim, hidden_size, n_layers, posembedding_path,
                 max_seq_len, max_doc_len, program_class, num_heads=8, use_mask=False, input_dropout=0.0,
                 word_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, learning_rate=1e-3,
                 l2_reg=1e-5):
        super(RTHN, self).__init__()

        self.model_dir = model_dir

        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len

        ## word embedding
        self.embed_dim = embed_dim
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=False,
                    output_hidden_states=False)
        self.bert_proj = nn.Linear(768, self.embed_dim)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.input_dropout = nn.Dropout(input_dropout)

        ## position embedding
        self.embed_dim_pos = pos_embed_dim
        posembedding = torch.FloatTensor(np.load(posembedding_path, allow_pickle=True))
        self.posembed = nn.Embedding.from_pretrained(posembedding, freeze=True)

        ## word level encoding
        self.hidden_size = hidden_size
        self.WordEncoder = nn.LSTM(self.embed_dim, hidden_size, bidirectional=True, batch_first=True, dropout=word_dropout)

        self.wordlinear_1 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.wordlinear_2 = nn.Linear(self.hidden_size * 2, 1, bias=False)

        ## sentence level encoding
        self.n_layers = n_layers
        self.program_class = program_class

        params_layer_1 = ((self.hidden_size * 2 + self.embed_dim_pos,
                          self.hidden_size * 2 + self.embed_dim_pos,
                          self.hidden_size * 2),
                          self.hidden_size * 2 + self.embed_dim_pos,
                          self.hidden_size * 2 + self.embed_dim_pos,
                          num_heads,
                          self.hidden_size * 2,
                          program_class,
                          max_doc_len,
                          _gen_bias_mask(max_doc_len) if use_mask else None,
                          layer_dropout,
                          attention_dropout)

        params_layers = ((self.hidden_size * 2 + self.max_doc_len,
                         self.hidden_size * 2 + self.max_doc_len,
                         self.hidden_size * 2),
                         self.hidden_size * 2 + self.max_doc_len,
                         self.hidden_size * 2 + self.max_doc_len,
                         num_heads,
                         self.hidden_size * 2,
                         program_class,
                         max_doc_len,
                         _gen_bias_mask(max_doc_len) if use_mask else None,
                         layer_dropout,
                         attention_dropout)
        self.rthn = nn.ModuleList([RTHNLayer(*params_layer_1)]+[RTHNLayer(*params_layers) for _ in range(n_layers-1)])

        ## training
        self.l2_reg = l2_reg
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        ## model save path
        self.best_path = ''

    def forward(self, input_ids, attention_mask, output_ids, sen_len, doc_len, word_dis, emotion, train=True):

        self.device = input_ids.device

        ### word level encoder based on RNNs
        clause, _ = self.bert(input_ids.reshape(-1, self.max_seq_len), token_type_ids=None, attention_mask=attention_mask.reshape(-1, self.max_seq_len))
        clause = self.bert_proj(clause)
        clause = self.input_dropout(clause).reshape(-1, self.max_seq_len, self.embed_dim)

        word_encode, _ = self.WordEncoder(clause)
        word_encode = word_encode.view(-1, self.max_seq_len, 2, self.hidden_size)
        word_encode = torch.cat((word_encode[:,:,-2,:].squeeze(2), word_encode[:,:,-1,:].squeeze(2)), dim=-1) # (batch_size * doc_len, seq_len, hidden_size *2)

        alpha = self.wordlinear_2(self.wordlinear_1(word_encode)).reshape(-1, 1, self.max_seq_len) # (batch_size * doc_len, 1, seq_len)
        mask = torch.arange(0, self.max_seq_len, step=1).to(self.device).repeat(alpha.size(0), 1) \
               < sen_len.reshape(-1, 1)
        alpha = torch.exp(alpha) * mask.unsqueeze(1).float()
        alpha = torch.softmax(alpha, dim=-1)

        sen_encode = torch.matmul(alpha, word_encode).reshape(-1, self.max_doc_len, self.hidden_size * 2) # (batch_size, doc_len, hidden_size *2)
        word_dis = self.posembed(word_dis) # (batch_size, doc_len, seq_len, embed_size)
        word_dis = word_dis[:, :, 0, :].reshape(-1, self.max_doc_len, self.embed_dim_pos) # (batch_size, doc_len, pos_embed_size)
        sen_encode_value = torch.cat((sen_encode, word_dis), dim=-1) # (batch_size, doc_len, hidden_size * 2 + pos_embed_size)

        ### clause level encoder based on Transformer
        attn_mask = torch.arange(0, self.max_doc_len, step=1).to(self.device).expand(sen_encode.size()[:2]) \
                    < doc_len.reshape(-1, 1)
        for l in range(self.n_layers):
            sen_encode, pred, pred_label, reg = self.rthn[l](sen_encode_value, sen_encode_value, sen_encode, doc_len, attn_mask)
            # shape --> sen_encode: (batch_size, doc_len, hidden_size * 2)
            sen_encode_value = torch.cat((sen_encode, pred_label), dim=-1)

        if train:
            valid_num = torch.sum(doc_len).to(self.device)
            loss_op = -torch.sum(output_ids * torch.log(pred)) / valid_num + reg * self.l2_reg
            self.optimizer.zero_grad()
            loss_op.backward(retain_graph=True)
            self.optimizer.step()

        true_y_op = torch.argmax(output_ids, -1).cpu().reshape(-1, )
        pred_y_op = torch.argmax(pred, -1).cpu().reshape(-1, )
        score_mask = torch.arange(0, self.max_doc_len, step=1).repeat(len(doc_len), 1) < doc_len.reshape(-1, 1).cpu()

        accuracy = accuracy_score(true_y_op, pred_y_op, sample_weight=score_mask.reshape(-1, ))
        precision = precision_score(true_y_op, pred_y_op, pos_label=0, sample_weight=score_mask.reshape(-1, ))
        recall = recall_score(true_y_op, pred_y_op, pos_label=0, sample_weight=score_mask.reshape(-1, ))
        F1 = f1_score(true_y_op, pred_y_op, pos_label=0, sample_weight=score_mask.reshape(-1, ))

        return accuracy, precision, recall, F1

    def save_model(self, iter, precision, recall, F1):

        state = {
            'iter': iter,
            'bert_proj_state_dict': self.bert_proj.state_dict(),
            'WordEncoder_state_dict': self.WordEncoder.state_dict(),
            'wordlinear_1_state_dict': self.wordlinear_1.state_dict(),
            'wordlinear_2_state_dict': self.wordlinear_2.state_dict(),
            'rthn_state_dict': self.rthn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        model_save_path = os.path.join(self.model_dir,
                                       'model_{}_{:.4f}_{:.4f}_{:.4f}'.format(iter, precision, recall, F1))

        torch.save(state, model_save_path)

    def predict(self):
        pass



