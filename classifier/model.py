import torch, os
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertPreTrainedModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Config(object):
    def __init__(self):
        # self.glove = False
        self.glove = True
        # self.external_knowledge = False
        self.external_knowledge = True
        self.batch_size = 128
        self.learning_rate = 1e-5
        self.train_iter = 5000
        self.max_length = 50

        self.embed_size = 768 # Bert embedding size
        self.emb_dim = 100 # glove embedding size
        self.hidden_dim = 256 # rnn hidden size
        self.ex_emb_dim = 5

        self.class_num = 32

        self.save_path = './save/'
        self.emb_file = '../glove.6B/glove.6B.{}d.txt'.format(
            str(self.emb_dim))
        self.emb_path = './data/embeddings.npy'
        self.vocab_path = './data/vocab.txt'
        self.senticnet_file = '../EDG/senticnet-6.0/'
        self.conceptnet_file = '../EDG/cpnet/'

def evaluate(model, data, ty='valid'):
    loss, prediction, accuracy, precision, recall, f1 = [], [], [], [], [], []
    if ty == 'test':
        print('Predicting the emotions of inputs...')

    for j, batch in enumerate(data):
        pred_label, scores, l = model.evaluate(batch)
        acc, pre, rec, f = scores
        prediction.append(pred_label)
        accuracy.append(acc)
        precision.append(pre)
        recall.append(rec)
        f1.append(f)
        loss.append(l)

    acc = np.mean(accuracy)
    pre = np.mean(precision)
    rec = np.mean(recall)
    f = np.mean(f1)
    l = np.mean(loss)
    print("loss: {:2f}, accuracy: {:2f}, precision: {:2f}, recall: {:2f}, f1: {:2f}".format(l, acc, pre, rec, f))
    if ty == 'valid':
       return acc, pre, rec, f, l
    else:
        torch.cat(prediction)

def get_input_from_batch(batch):
    lengths = batch['lengths']
    sequence = batch['sequence']
    label = batch['label']
    emotion = batch['emotion']
    mask = batch['mask']
    batch_size, max_len = sequence.size()

    try:
        scores, ex_embed = batch['scores'], batch['ex_emb']
        assert len(lengths) == len(label) == len(emotion) == len(mask) == batch_size

        return sequence, label, mask, lengths, emotion, scores, ex_embed

    except KeyError:
        assert len(lengths) == len(label) == len(emotion) == len(mask) == batch_size
        return  sequence, label, mask, lengths, emotion

class CLF(nn.Module):
    def __init__(self, config):
        super(CLF, self).__init__()
        self.model_dir = config.save_path + 'lstm-clf/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        glove_embed = np.load(config.emb_path, allow_pickle=True)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove_embed))
        self.rnn = nn.LSTM(config.emb_dim, config.hidden_dim, batch_first=True, bidirectional=True)
        # self.linear = nn.Linear(config.emb_dim * config.max_length, config.class_num)
        self.linear = nn.Linear(config.hidden_dim * 2, config.class_num)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    def forward(self, batch):
        input_ids, cls, mask, lengths, emotion = get_input_from_batch(batch)
        # batch_size = cls.size(0)
        x = self.embedding(input_ids)
        _, hid = self.rnn(x)
        # pred_cls = self.linear(x.reshape(batch_size, -1))
        x = torch.cat((hid[0][-2], hid[0][-1]), dim=-1)
        pred_cls = self.linear(x)
        pred_cls = torch.softmax(pred_cls, dim=-1)
        pred_label = torch.argmax(pred_cls, dim=-1)
        loss = nn.CrossEntropyLoss()(pred_cls, cls)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        accuracy, precision, recall, f1 = self.compute_scores(pred_label.cpu(), cls.cpu())
        return loss.item(), (accuracy, precision, recall, f1)

    def compute_scores(self, pred_label, true_label):
        accuracy = accuracy_score(pred_label, true_label)
        precision = precision_score(pred_label, true_label, average='weighted')
        recall = recall_score(pred_label, true_label, average='weighted')
        f1 = f1_score(pred_label, true_label, average='weighted')

        return accuracy, precision, recall, f1

    def evaluate(self, batch):
        input_ids, cls, mask, lengths, emotion =  get_input_from_batch(batch)
        x = self.embedding(input_ids)
        _, hid = self.rnn(x)
        # pred_cls = self.linear(x.reshape(batch_size, -1))
        x = torch.cat((hid[0][-2], hid[0][-1]), dim=-1)
        pred_cls = self.linear(x)
        pred_cls = torch.softmax(pred_cls, dim=-1)
        pred_label = torch.argmax(pred_cls, dim=-1)
        loss = nn.CrossEntropyLoss()(pred_cls, cls)
        accuracy, precision, recall, f1 = self.compute_scores(pred_label.cpu(), cls.cpu())
        return pred_label, (accuracy, precision, recall, f1), loss.item()

    def save(self, iter, acc, pre, recall, f1):
        state = {
            'iter': iter,
            'embedding_state_dict': self.embedding.state_dict(),
            'rnn_state_dict': self.rnn.state_dict(),
            'linear_state_dict': self.linear.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:2f}_{:2f}_{:2f}_{:2f}'.format(iter, acc, pre, recall, f1))

        torch.save(state, model_save_path)

def EKCLF(CLF):
    def __init__(self, config):
        super().__init__(config)
        self.model_dir = config.save_path + 'lstm-sigmoid-ekclf/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        glove_embed = np.load(config.emb_path, allow_pickle=True)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove_embed))
        self.rnn = nn.LSTM(config.emb_dim + config.ex_emb_dim, config.hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(config.hidden_dim * 2 + 2, config.class_num)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    def forward(self, batch):
        input_ids, cls, mask, lengths, emotion, scores, ex_embed = get_input_from_batch(batch)
        x = self.embedding(input_ids)
        x = torch.cat((x, ex_embed), dim=-1)
        _, hid = self.rnn(x)
        x = torch.cat((scores, hid[0][-2], hid[0][-1]), dim=-1)
        pred_cls = self.linear(x)
        pred_cls = torch.softmax(pred_cls, dim=-1)
        pred_label = torch.argmax(pred_cls, dim=-1)
        loss = nn.CrossEntropyLoss()(pred_cls, cls)
        loss.backward()
        self.optimizer.step()
        accuracy, precision, recall, f1 = self.compute_scores(pred_label.cpu(), cls.cpu())
        return loss.item(), (accuracy, precision, recall, f1)

    def evaluate(self, batch):
        input_ids, cls, mask, lengths, emotion, scores, ex_embed = get_input_from_batch(batch)
        x = self.embedding(input_ids)
        x = torch.cat((x, ex_embed), dim=-1)
        _, hid = self.rnn(x)
        x = torch.cat((scores, hid[0][-2], hid[0][-1]), dim=-1)
        pred_cls = self.linear(x)
        pred_cls = torch.softmax(pred_cls, dim=-1)
        pred_label = torch.argmax(pred_cls, dim=-1)
        loss = nn.CrossEntropyLoss()(pred_cls, cls)
        accuracy, precision, recall, f1 = self.compute_scores(pred_label.cpu(), cls.cpu())
        return pred_label, (accuracy, precision, recall, f1), loss.item()

class BertCLF(CLF):
    def __init__(self, config):
        super().__init__(config)
        # self.model_dir = config.save_path + 'bert-clf/'
        self.model_dir = config.save_path + 'bert-sigmoid-clf/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=False, output_hidden_states=False)
        self.rnn = nn.LSTM(config.embed_size, config.hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(config.hidden_dim * 2, config.class_num)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    def forward(self, batch):
        input_ids, cls, mask, lengths, emotion = get_input_from_batch(batch)
        x, _ = self.bert(input_ids)
        _, hid = self.rnn(x)
        x = torch.cat((hid[0][-2], hid[0][-1]), dim=-1)
        pred_cls = self.linear(x)
        pred_cls = torch.softmax(pred_cls, dim=-1)
        pred_label = torch.argmax(pred_cls, dim=-1)
        loss = nn.CrossEntropyLoss()(pred_cls, cls)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        accuracy, precision, recall, f1 = self.compute_scores(pred_label.cpu(), cls.cpu())
        return loss.item(), (accuracy, precision, recall, f1)

    def evaluate(self, batch):
        input_ids, cls, mask, lengths, emotion =  get_input_from_batch(batch)
        x, _ = self.bert(input_ids)
        _, hid = self.rnn(x)
        x = torch.cat((hid[0][-2], hid[0][-1]), dim=-1)
        pred_cls = self.linear(x)
        pred_cls = torch.softmax(pred_cls, dim=-1)
        pred_label = torch.argmax(pred_cls, dim=-1)
        accuracy, precision, recall, f1 = self.compute_scores(pred_label.cpu(), cls.cpu())
        loss = nn.CrossEntropyLoss()(pred_cls, cls)
        return pred_label, (accuracy, precision, recall, f1), loss.item()

    def save(self, iter, acc, pre, recall, f1):
        state = {
            'iter': iter,
            'bert_state_dict': self.bert.state_dict(),
            'rnn_state_dict': self.rnn.state_dict(),
            'linear_state_dict': self.linear.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:2f}_{:2f}_{:2f}_{:2f}'.format(iter, acc, pre, recall, f1))

        torch.save(state, model_save_path)

class EKBertCLF(BertCLF):
    def __int__(self, config):
        super().__init__(config)
        self.model_dir = config.save_path + 'ekclf/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=False, output_hidden_states=False)
        self.ScoreLinear = nn.Linear(config.embed_size, config.emb_dim)
        self.rnn = nn.LSTM(config.emb_dim + config.ex_emb_dim, config.hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(config.hidden_dim * 2 + 2, config.class_num)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    def forward(self, batch):
        input_ids, cls, mask, lengths, emotion, scores, ex_embed = get_input_from_batch(batch)
        x, _ = self.bert(input_ids)
        x = torch.cat((self.ScoreLinear(x), ex_embed), dim=-1)
        _, hid = self.rnn(x)
        x = torch.cat((scores, hid[0][-2], hid[0][-1]), dim=-1)
        pred_cls = self.linear(x)
        pred_cls = torch.softmax(pred_cls, dim=-1)
        pred_label = torch.argmax(pred_cls, dim=-1)
        loss = nn.CrossEntropyLoss(pred_cls, cls)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        accuracy, precision, recall, f1 = self.compute_scores(pred_label, cls)
        return loss.item(), (accuracy, precision, recall, f1)

    def evaluate(self, batch):
        input_ids, cls, mask, lengths, emotion, scores, ex_embed = get_input_from_batch(batch)
        x, _ = self.bert(input_ids)
        x = torch.cat((self.ScoreLinear(x), ex_embed), dim=-1)
        _, hid = self.rnn(x)
        x = torch.cat((hid[0][-2], hid[0][-1]), dim=-1)
        pred_cls = self.linear(x)
        pred_cls = torch.softmax(pred_cls, dim=-1)
        pred_label = torch.argmax(pred_cls, dim=-1)
        accuracy, precision, recall, f1 = self.compute_scores(pred_label.cpu(), cls.cpu())
        loss = nn.CrossEntropyLoss(pred_cls, cls)
        return pred_label, (accuracy, precision, recall, f1), loss.item()

    def save(self, iter, acc, pre, recall, f1):
        state = {
            'iter': iter,
            'bert_state_dict': self.bert.state_dict(),
            'ScoreLinear_state_dict': self.ScoreLinear.state_dict(),
            'rnn_state_dict': self.rnn.state_dict(),
            'linear_state_dict': self.linear.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:2f}_{:2f}_{:2f}_{:2f}'.format(iter, acc, pre, recall, f1))

        torch.save(state, model_save_path)
