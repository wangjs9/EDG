import torch
import torch.utils.data as data
import logging
import numpy as np
# import config
from utils import config
import pprint
pp = pprint.PrettyPrinter(indent=1)
from model.common_layer import write_config
from utils.data_reader import load_dataset

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
            'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15,
            'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
            'anticipating': 22, 'embarrassed': 23,
            'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31}

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["situation_text"] = self.data["situation"][index]
        item["context_text"] = self.data["context"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        causeprob = self.data["causeprob"][index]

        item["context"], item["context_mask"], item["causepos"], \
        item["causeprob"], item["causeclz"] = self.preprocess(
            (item["situation_text"], item["context_text"]),
            scores=causeprob)

        item["context_text"] = item["situation_text"] + [ele for lst in item["context_text"] for ele in lst]

        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)

        return item

    def preprocess(self, arr, scores = None, anw=False):
        """Converts words to ids."""
        if anw:
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                        arr] + [config.EOS_idx]
            return torch.LongTensor(sequence)
        else:
            situation, context = arr
            X_dial = [config.CLS_idx]
            X_mask = [config.CLS_idx]
            X_score = [0]
            X_causepos = [np.NINF]
            for i, sentence in enumerate(situation):
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                           sentence]
                X_mask += [self.vocab.word2index["SIT"] for _ in range(len(sentence))]
                score = 0
                length = 0
                cazcounter = [np.NINF for _ in range(len(sentence))]
                for j, s in enumerate(scores):
                    if s == []:
                        continue
                    length += 1
                    score += s[i]
                    if s[i] > 0.5:
                        cazcounter = [j for _ in range(len(sentence))]
                if length:
                    score /= length
                X_score += [score for _ in range(len(sentence))]
                X_causepos += cazcounter
            curId = len(situation)
            for i, sentences in enumerate(context):
                for j, sentence in enumerate(sentences):
                    X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                               sentence]
                    # >>>>>>>>>> spk: whether this sen is from a user or bot >>>>>>>>>> #
                    spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                    X_mask += [spk for _ in range(len(sentence))]
                    score = 0
                    cazcounter = [np.NINF for _ in range(len(sentence))]
                    validscores = scores[curId+i:]
                    if len(validscores):
                        length = 0
                        for j, s in enumerate(validscores):
                            if s != []:
                                length += 1
                                score += s[curId]
                                if s[curId] > 0.5:
                                    cazcounter = [curId for _ in range(len(sentence))]
                            curId += 1
                        if length:
                            score /= length
                    X_score += [score for _ in range(len(sentence))]
                    X_causepos += cazcounter

            tmp = sorted(zip(X_dial, X_causepos, X_score), key=lambda x: (x[1], x[2]), reverse=True)
            X_clause = [config.CLS_idx]
            curref = tmp[0][1:]
            cazno = 0
            for ele in tmp:
                if curref != ele[1:]:
                    cazno += 1
                    if cazno >= 5:
                        break
                    X_clause.append(self.vocab.word2index["SEP"])
                X_clause.append(ele[0])
            X_causepos = [0 if np.isneginf(pos) else pos-curId+65 for pos in X_causepos]
            assert len(X_dial) == len(X_causepos) == len(X_mask) == len(X_score)

            return torch.LongTensor(X_dial), torch.LongTensor(X_mask), \
                   torch.LongTensor(X_causepos), torch.FloatTensor(X_score), torch.LongTensor(X_clause)
            # >>>>>>>>>> context, context mask, cause score >>>>>>>>>> #

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        # >>>>>>>>>> one hot mode and label mode >>>>>>>>>> #
        return program, emo_map[emotion]

def collate_fn(data):
    def merge(sequences, scores=None, positions=None):
        """
        padded_seqs: use 1 to pad the rest
        lengths: the lengths of seq in sequences
        """
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()  ## padding index 1

        if scores == None:
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        else:
            cause_score = torch.zeros(len(sequences), max(lengths)).long()
            cause_pos = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                score = scores[i]
                pos = positions[i]
                padded_seqs[i, :end] = seq[:end]
                cause_score[i, :end] = score[:end]
                cause_pos[i, :end] = pos[:end]
            return padded_seqs, lengths, cause_score, cause_pos

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    # >>>>>>>>>> transfer style:
    # [dict1: {k1: v1, k2: v2, ... km: vm}, dict2, ... dictn]
    # dict: {k1: [], k2: [], ..., km: [n items]}
    # >>>>>>>>>> #
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths, input_causeprob, input_causepos = merge(item_info['context'],
                                            scores=item_info['causeprob'],
                                            positions=item_info['causepos'])

    mask_input, mask_input_lengths = merge(item_info['context_mask'])  # use idx for bot or user to mask the seq

    ## cause
    cause_batch, cause_lengths = merge(item_info['causeclz'])

    ## Target
    target_batch, target_lengths = merge(item_info['target'])

    if config.USE_CUDA:
        input_batch = input_batch.cuda()
        mask_input = mask_input.cuda()
        target_batch = target_batch.cuda()
        cause_batch = cause_batch.cuda()


    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)  # mask_input_lengths equals input_lengths
    d["input_causeprob"] = input_causeprob
    d["input_causepos"] = input_causepos
    d["mask_input"] = mask_input
    ##cause
    d["cause_batch"] = cause_batch
    d["cause_lengths"] = torch.LongTensor(cause_lengths)
    ##target
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    ##program
    d["target_program"] = item_info['emotion']  # one hot format
    d["program_label"] = item_info['emotion_label']

    ##text
    d["input_txt"] = item_info['context_text']
    d["target_txt"] = item_info['target_text']
    d["program_txt"] = item_info['emotion_text']

    return d

def prepare_data_seq(batch_size=32):
    """
    :return:
    vocab: vocabulary including index2word, and word2index
    len(dataset_train.emo_map)
    """
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                              batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)

    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                              batch_size=1,
                              shuffle=False, collate_fn=collate_fn)
    write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)

if __name__ == '__main__':
    prepare_data_seq()