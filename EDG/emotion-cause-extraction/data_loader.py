import numpy as np
import pickle as pk
import pandas as pd
import re,gc, torch, random
path = '../reman/'
test_path = '../empathetic-dialogue/'
max_doc_len = 32
max_sen_len = 50

emotion_dict = {'NA': 0, 'SADNESS': 1, 'ANGER': 2, 'FEAR': 3, 'SURPRISE': 4,
                'ANTICIPATION': 5, 'TRUST': 6, 'DISGUST': 7, 'OTHER': 8, 'JOY': 0}

class DatasetIterator(object):
    def __init__(self, batch_size, dataset, device):

        self.batch_size = batch_size
        self.dataset = dataset

        self.total_batch = len(self.dataset)
        self.n_batches = self.total_batch // self.batch_size
        self.residue = False if self.n_batches * self.batch_size == self.total_batch else True
        self.start = 0
        self.end = self.start + self.batch_size
        self.device = device

    def _to_tensor(self, data):
        x = torch.LongTensor([_[0] for _ in data]).to(self.device)
        mask = torch.LongTensor([_[1] for _ in data]).to(self.device)
        y = torch.LongTensor([_[2] for _ in data]).to(self.device)
        sen_len = torch.LongTensor([_[3] for _ in data]).to(self.device)
        doc_len = torch.LongTensor([_[4] for _ in data]).to(self.device)
        relative_pos = torch.LongTensor([_[5] for _ in data]).to(self.device)
        emotion = torch.LongTensor([_[6] for _ in data]).to(self.device)

        return x, mask, y, sen_len, doc_len, relative_pos, emotion

    def __next__(self):
        if self.end > self.total_batch:
            batches = self.dataset[self.start: ] + self.dataset[:self.end % self.total_batch]
        else:
            batches = self.dataset[self.start: self.end]
        self.start = self.end % self.total_batch
        self.end = self.start + self.batch_size

        batches = self._to_tensor(batches)
        return batches

    def __iter__(self):
        return self

def DataIter(data_path, batch_size, device):

    def load_data(data_path):
        x = pk.load(open(data_path + 'x.txt', 'rb'))
        mask = pk.load(open(data_path + 'mask.txt', 'rb'))
        y = pk.load(open(data_path + 'y.txt', 'rb'))
        sen_len = pk.load(open(data_path + 'sen_len.txt', 'rb'))
        doc_len = pk.load(open(data_path + 'doc_len.txt', 'rb'))
        relative_pos = pk.load(open(data_path + 'relative_pos.txt', 'rb'))
        emotion = pk.load(open(data_path + 'emotion.txt', 'rb'))
        print('x.shape {}\nmask.shape \ny.shape {} \nsen_len.shape {} \ndoc_len.shape '
              '{}\nrelative_pos.shape {}\nemotion.shape {}'.format(x.shape, mask.shape,
                y.shape, sen_len.shape, doc_len.shape, relative_pos.shape, emotion.shape))

        dataset = list()

        for i in range(len(x)):
            dataset.append([x[i], mask[i], y[i], sen_len[i], doc_len[i], relative_pos[i], emotion[i]])
        random.shuffle(dataset)
        return dataset

    dataset = load_data(data_path)
    total_len = len(dataset)
    train_len = int(total_len * 0.9)
    train_data = dataset[:train_len]
    dev_data = dataset[train_len:total_len]

    train_DataIter = DatasetIterator(batch_size, train_data, device)
    dev_DataIter = DatasetIterator(batch_size, dev_data, device)

    return train_DataIter, dev_DataIter

def process_data(text_file, input_file, output_file, context=False):
    doctext = pd.read_csv(text_file, sep='\t', header=None, index_col=False, encoding='UTF-8')
    ece = pd.read_csv(input_file, sep='\t', header=0, index_col=False, encoding='UTF-8')

    punc = r'[!",-.:;?~]\s'

    if context:
        col_name = ['conv_id', 'no', 'context', 'emotion_clause', 'cause_clause', 'label', 'relative_pos', 'cause', 'clause']
    else:
        col_name = ['conv_id', 'no', 'label', 'relative_pos', 'cause', 'clause']
    data = []
    SenID = 0
    for conv_id, row in ece.iterrows():
        text = doctext.loc[conv_id,0]
        text_tokens = re.split(punc, text)
        lengths = [len(tokens) for tokens in text_tokens]
        interval = [0]

        for i, length in enumerate(lengths[:-1]):
            interval.append(length+interval[i])

        emotion_label, emotion_spans, cause_spans = row
        emotion_label = emotion_label[1:-1].replace('\'', '').split(',')
        emotion_spans = emotion_spans[1:-1].replace('\'', '').split(',')
        cause_spans = cause_spans[1:-1].replace('\'', '').split(',')
        for no, label in enumerate(emotion_label):
            emotion_span1, emotion_span2 = int(emotion_spans[no*2]), int(emotion_spans[no*2+1])
            cause_span1, cause_span2 = int(cause_spans[no*2]), int(cause_spans[no*2+1])
            # No need to delete, the clauses of emotion and cause
            emotion_clause = text[emotion_span1:emotion_span2]
            cause_clause = text[cause_span1:cause_span2]
            emotion_id = 0
            cause_id = 0
            for length_id in range(len(interval)):
                if emotion_span1 > interval[length_id]:
                    emotion_id = length_id
                if cause_span1 > interval[length_id]:
                    cause_id = length_id
            for clause_id, clause in enumerate(text_tokens):
                if context:
                    temp = [SenID, clause_id, text, emotion_clause, cause_clause, label, clause_id-emotion_id, clause_id==cause_id, clause]
                else:
                    temp = [SenID, clause_id, label, clause_id-emotion_id, clause_id==cause_id, clause]
                data.append(temp.copy())
            SenID += 1

    data = pd.DataFrame(data)
    data.to_csv(output_file, encoding='UTF-8', sep='\t', index=False, header=col_name)

def load_w2v(embedding_dim_pos):
    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(
        loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(-68, 34)])
    # embedding.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim)) for i in range(-68,34)])
    embedding_pos = np.array(embedding_pos)
    pk.dump(embedding_pos, open(path + 'embedding_pos.txt', 'wb'))
    print("embedding_pos.shape: {}".format(embedding_pos.shape))
    return embedding_pos

def load_data(input_file, max_doc_len=max_doc_len, max_sen_len=max_sen_len):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print('load data...')

    relative_pos, x, mask, y, sen_len, doc_len, emotion = [], [], [], [], [], [], []
    y_clause_cause, clause_all, mask_all, tmp_clause_len, relative_pos_all, emotion_all = \
        np.zeros((max_doc_len, 2)), [], [], [], [], np.zeros((max_doc_len,))

    next_ID = 1
    n_clause, yes_clause, no_clause, n_cut = [0] * 4

    data = pd.read_csv(input_file, sep='\t', encoding='UTF-8', header=0)

    max_doc = 0
    max_sen = 0

    for index, line in data.iterrows():
        n_clause += 1
        senID, clause_idx, emo_word, sen_pos, cause, words = line
        word_pos = int(sen_pos) + 69
        if sen_pos == 0:
            emotion_all[clause_idx] = emotion_dict[emo_word.strip()]
        if next_ID == senID:
            doc_len.append(len(clause_all))
            if len(clause_all) > max_doc:
                max_doc = len(clause_all)
            for j in range(max_doc_len - len(clause_all)):
                clause_all.append(np.zeros((max_sen_len,)))
                mask_all.append(np.zeros(max_sen_len,))
                tmp_clause_len.append(0)
                relative_pos_all.append(np.zeros((max_sen_len,)))
            relative_pos.append(relative_pos_all)
            x.append(clause_all)
            mask.append(mask_all)
            y.append(y_clause_cause)
            sen_len.append(tmp_clause_len)
            emotion.append(emotion_all)
            y_clause_cause, clause_all, mask_all, tmp_clause_len, relative_pos_all, emotion_all = \
                np.zeros((max_doc_len, 2)), [], [], [], [], np.zeros((max_doc_len,))
            next_ID = senID + 1

        encoded_dict = tokenizer.encode_plus(words,
                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                         max_length=max_sen_len,  # Pad & truncate all sentences.
                         pad_to_max_length=True,
                         return_attention_mask=True,  # Construct attn. masks.
                         truncation=True)

        clause = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        relative_pos_clause = [word_pos] * max_sen_len
        relative_pos_all.append(np.array(relative_pos_clause))
        clause_all.append(clause)
        mask_all.append(attention_mask)
        if sum(encoded_dict['attention_mask']) > max_sen:
            max_sen = sum(encoded_dict['attention_mask'])
        tmp_clause_len.append(sum(encoded_dict['attention_mask']))
        if cause:
            no_clause += 1
            y_clause_cause[clause_idx] = [1,0]
        else:
            yes_clause += 1
            y_clause_cause[clause_idx] = [0,1]

    relative_pos, x, mask, y, sen_len, doc_len, emotion = map(np.array, [relative_pos, x, mask, y, sen_len, doc_len, emotion])
    pk.dump(relative_pos, open(path + 'relative_pos.txt', 'wb'))
    pk.dump(x, open(path + 'x.txt', 'wb'))
    pk.dump(mask, open(path + 'mask.txt', 'wb'))
    pk.dump(y, open(path + 'y.txt', 'wb'))
    pk.dump(sen_len, open(path + 'sen_len.txt', 'wb'))
    pk.dump(doc_len, open(path + 'doc_len.txt', 'wb'))
    pk.dump(emotion, open(path + 'emotion.txt', 'wb'))

    print('relative_pos.shape {}\nx.shape {} \nmask.shape {} \ny.shape {} \nsen_len.shape {} \ndoc_len.shape {}\nemotion.shape {}\n'.format(
        relative_pos.shape, x.shape, mask.shape, y.shape, sen_len.shape, doc_len.shape, emotion.shape
    ))
    print('n_clause {}, yes_clause {}, no_clause {}, n_cut {}'.format(n_clause, yes_clause, no_clause, n_cut))
    print('load data done!\n')
    print('max_doc_len {}\nmax_sen_len {}'.format(max_doc, max_sen))
    return relative_pos, x, mask, y, sen_len, doc_len, emotion

def load_test_data(input_file, round, up_id, bottom_id, max_doc_len=max_doc_len, max_sen_len=max_sen_len):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print('load data...')
    relative_pos, x, mask, y, sen_len, doc_len, emotion = [], [], [], [], [], [], []
    y_clause_cause, clause_all, mask_all, tmp_clause_len, relative_pos_all, emotion_all = \
        np.zeros((max_doc_len, 2)), [], [], [], [], np.zeros((max_doc_len,))
    index_list = []
    next_ID = up_id + 2
    n_clause = 0

    data = pd.read_csv(input_file, sep='\t', encoding='UTF-8', header=0)
    for index, line in data.iterrows():
        n_clause += 1
        senID, clause_no, label, context_w, emotion_w, chatbot_w, words = line
        word_pos = clause_no + 69

        if senID <= up_id:
            continue
        elif senID > bottom_id:
            break

        if next_ID == senID:
            clause_all, mask_all, tmp_clause_len, relative_pos_all = [], [], [], []
            next_ID = senID + 1

        if not context_w and emotion_w:
            doc_len.append(len(clause_all))
            relative_pos_all = [array - clause_no for array in relative_pos_all]
            for j in range(max_doc_len - len(clause_all)):
                clause_all.append(np.zeros((max_sen_len, )))
                mask_all.append(np.zeros((max_sen_len, )))
                tmp_clause_len.append(0)
                relative_pos_all.append(np.zeros((max_sen_len,)))
            relative_pos.append(relative_pos_all)
            x.append(clause_all)
            mask.append(mask_all)
            sen_len.append(tmp_clause_len)
            clause_all, mask_all, tmp_clause_len, relative_pos_all = [], [], [], []
            index_list.append([senID, clause_no, True])
        else:
            index_list.append([senID, clause_no, False])
        encoded_dict = tokenizer.encode_plus(words,
                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                         max_length=max_sen_len,  # Pad & truncate all sentences.
                         pad_to_max_length=True,
                         return_attention_mask=True,  # Construct attn. masks.
                         truncation=True)
        clause = encoded_dict['input_ids']
        attention_mask=encoded_dict['attention_mask']
        relative_pos_clause = [word_pos] * max_sen_len
        relative_pos_all.append(np.array(relative_pos_clause))
        clause_all.append(clause)
        mask_all.append(attention_mask)
        tmp_clause_len.append(sum(encoded_dict['attention_mask']))
        del encoded_dict
        gc.collect()

    relative_pos, x, mask, sen_len, doc_len = map(np.array, [relative_pos, x, mask, sen_len, doc_len])
    pk.dump(relative_pos, open(path + '{}-test_relative_pos.txt'.format(round+1), 'wb'))
    pk.dump(x, open(path + '{}-test_x_pred.txt'.format(round+1), 'wb'))
    pk.dump(mask, open(path + '{}-test_mask_pred.txt'.format(round+1), 'wb'))
    pk.dump(sen_len, open(path + '{}-test_sen_len_pred.txt'.format(round+1), 'wb'))
    pk.dump(doc_len, open(path + '{}-test_doc_len_pred.txt'.format(round+1), 'wb'))
    index_list = np.array(index_list)
    index_list = pd.DataFrame(index_list)
    index_list.to_csv(path + '{}-evaluation_file.csv'.format(round+1), sep='\t', header=None, index=False)
    print('relative_pos.shape {}\nx.shape {}\nmask.shape {} \nsen_len.shape {} \ndoc_len.shape {}\n'.format(
        relative_pos.shape, x.shape, mask.shape, sen_len.shape, doc_len.shape
    ))
    print('test_n_clause {}'.format(n_clause))
    print('load test data done!\n')
    del relative_pos, x, mask, sen_len, doc_len, data
    gc.collect()

def revise_test_data(input_file, predict_path, output_file=None):
    print('load data...')

    pred_prob = list()
    for round in range(5):
        pred_file = predict_path + 'pred_{}_prob.txt'.format(round + 1)
        pred_y = pk.load(open(pred_file, 'rb'))
        pred_prob.extend(pred_y)
    cur_ID = 0
    next_ID = cur_ID + 1
    new_data = list()
    counter = 0
    data = pd.read_csv(input_file, sep='\t', encoding='UTF-8', header=0)
    generator = nextLine(pred_prob)
    for index, line in data.iterrows():
        senID, clause_no, _, context_w, emotion_w, chatbot_w, words = line
        if cur_ID > senID:
            continue
        if senID == next_ID:
            next_ID += 1
        if not context_w and emotion_w:
            try:
                prob = next(generator)
            except StopIteration:
                print(counter)
                break
            counter += 1
            probList = [sen[1] for sen in prob[:clause_no+1]]
        else:
            probList = []
        tmp = [senID, clause_no, context_w, emotion_w, chatbot_w, words, probList]
        new_data.append(tmp)

    col_name = ['conv_id', 'clause_no', 'context?', 'emotion?', 'chatbot?', 'clause', 'prob']
    new_data = pd.DataFrame(new_data)
    new_data.to_csv(output_file, encoding='UTF-8', sep='\t', index=False, header=col_name)

def nextLine(AList):
    for line in AList:
        yield line


# >>>>>>>>>> get some commonsense data >>>>>>>>> #
# process_data(path+'reman-text.csv', path+'reman-ece.csv', path+'clause_keywords.csv', context=False)

# >>>>>>>>>> position information >>>>>>>>> #
# load_w2v(56)

# >>>>>>>>>> load training data >>>>>>>>>> #
# load_data(path+'clause_keywords.csv')

# >>>>>>>>>> load testing data >>>>>>>>>> #total_len = 0
# >>>>>>>>>> There are 19,531 instances >>>>>>>>> #
# add_num = 135
# up_id = 694
# bottom_id = up_id + add_num
# sys.stdout = open('./test_dataset_log_3', 'w')
# round = 5
# while True:
#     print('>>>>>>>>>> round {}, up_id: {}, bottom id: {} >>>>>>>>>>'.format(round+1, up_id, bottom_id))
#     try:
#         load_test_data('./empatheticdialogues/clause_keywords.csv', round, up_id, bottom_id)
#         up_id = bottom_id
#         add_num += 5
#         bottom_id += add_num
#         round += 1
#     except OverflowError:
#         add_num -= 10
#         bottom_id -= 10
#         print('Add {} per turn'.format(add_num))
#     except MemoryError:
#         print('Faced MemoryError')
#         break
#     if up_id > 19531:
#         break
#     gc.collect()

# >>>>>>>>>> combine the prediction and original data >>>>>>>>>> #
# revise_test_data('./empatheticdialogues/clause_keywords.csv', './empatheticdialogues/', './empatheticdialogues/clause_view-.csv')