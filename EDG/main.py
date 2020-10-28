from utils.data_loader import prepare_data_seq
from utils import config
from model.transformer import Transformer
from model.transformer_multidec import Transformer_multidec
from model.transformer_reason import Transformer_ECE, Transformer_CVAE
from model.transformer_multihop import Transformer_multihop
from model.common_layer import evaluate, count_parameters, make_infinite
import torch
from torch.nn.init import xavier_uniform_
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import os, datetime

def find_model_path(save_path):
    list = os.listdir(save_path)
    list = [ele for ele in list if ele[:5] == 'model']
    if list == []:
        return None
    list.sort(key=lambda fn: os.path.getmtime(save_path + fn))
    # model_path = datetime.datetime.fromtimestamp(os.path.getmtime(save_path+list[-1]))
    model_path = os.path.join(save_path, list[-1])
    return model_path

def train_eval(test=False):

    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    if config.test:
        print('Test model', config.model)
        if config.model == "trs":
            model = Transformer(vocab, decoder_number=program_number, model_file_path=config.save_path, is_eval=True)

        if config.USE_CUDA:
            model.cuda()
        model = model.eval()
        loss_test, ppl_test, bce_test, acc_test, bleu_score_g, bleu_score_b= evaluate(model, data_loader_tst ,ty="test", max_dec_step=50)
        exit(0)

    model_file_path = find_model_path(config.save_path)
    if config.model == 'trs':
        model = Transformer(vocab, decoder_number=program_number, model_file_path=model_file_path)
        if model_file_path is None:
            for n, p in model.named_parameters():
                if p.dim() > 1 and n != "embedding.lut.weight" and config.pretrain_emb:
                    xavier_uniform_(p)

    elif config.model == 'multidec':
        model = Transformer_multidec(vocab, decoder_number=program_number, model_file_path=model_file_path)
        if model_file_path is None:
            for n, p in model.named_parameters():
                if p.dim() > 1 and n != "embedding.lut.weight" and config.pretrain_emb:
                    xavier_uniform_(p)

    elif config.model == 'reason':
        model = Transformer_ECE(vocab, decoder_number=program_number, model_file_path=model_file_path)
        if model_file_path is None:
            for n, p in model.named_parameters():
                if p.dim() > 1 and n != "embedding.lut.weight" and config.pretrain_emb:
                    xavier_uniform_(p)

    elif config.model == 'cvae':
        model = Transformer_CVAE(vocab, decoder_number=program_number, model_file_path=model_file_path)
        if model_file_path is None:
            for n, p in model.named_parameters():
                if p.dim() > 1 and n != "embedding.lut.weight" and config.pretrain_emb:
                    xavier_uniform_(p)

    elif config.model == 'multihop':
        model = Transformer_multihop(vocab, decoder_number=program_number, model_file_path=model_file_path)
        if model_file_path is None:
            for n, p in model.named_parameters():
                if p.dim() > 1 and n !="embedding.lut.weight" and config.pretrain_emb:
                    xavier_uniform_(p)

    elif config.model == 'Dynamichop':
        model = None # , model_file_path=model_file_path
        if model_file_path is None:
            for n, p in model.named_parameters():
                if p.dim() > 1 and n != "embedding.lut.weight" and config.pretrain_emb:
                    xavier_uniform_(p)

    print('MODEL USED', config.model)
    print('TRAINABLE PARAMETERS', count_parameters(model))

    check_iter = 20

    try:
        if config.USE_CUDA:
            model.cuda()
        model = model.train()
        best_ppl = 1000
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_loader_tra = data_loader_tra
        data_iter = make_infinite(data_loader_tra)
        for n_iter in tqdm(range(10000)): # 1000000
            print(n_iter)
            loss, ppl, bce, acc = model.train_one_batch(next(data_iter), n_iter)
            writer.add_scalars('loss', {'loss_train': loss}, n_iter)
            writer.add_scalars('ppl', {'ppl_train': ppl}, n_iter)
            writer.add_scalars('bce', {'bce_train': bce}, n_iter)
            writer.add_scalars('accuracy', {'acc_train': acc}, n_iter)
            if (config.noam):
                writer.add_scalars('lr', {'learning_rata': model.optimizer._rate}, n_iter)

            if (n_iter + 1) % check_iter == 0:
                model = model.eval()
                model.epoch = n_iter
                model.__id__logger = 0
                loss_val, ppl_val, bce_val, acc_val, bleu_score_g, bleu_score_b = evaluate(model, data_loader_val,
                                                                                           ty="valid", max_dec_step=50)
                writer.add_scalars('loss', {'loss_valid': loss_val}, n_iter)
                writer.add_scalars('ppl', {'ppl_valid': ppl_val}, n_iter)
                writer.add_scalars('bce', {'bce_valid': bce_val}, n_iter)
                writer.add_scalars('accuracy', {'acc_train': acc_val}, n_iter)
                model = model.train()
                if n_iter < 130: # 13000
                    continue
                if ppl_val <= best_ppl:
                    best_ppl = ppl_val
                    patient = 0
                    model.save_model(best_ppl, n_iter, 0, 0, bleu_score_g, bleu_score_b)
                    weights_best = deepcopy(model.state_dict())
                else:
                    patient += 1
                if patient > 2: break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    if test:
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        model.eval()
        model.epoch = 100
        loss_test, ppl_test, bce_test, acc_test, bleu_score_g, bleu_score_b = evaluate(model, data_loader_tst,
                                                                                       ty="test", max_dec_step=50)

        file_summary = config.save_path + "summary.txt"
        with open(file_summary, 'w') as the_file:
            the_file.write("EVAL\tLoss\tPPL\tAccuracy\tBleu_g\tBleu_b\n")
            the_file.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\n".format("test", loss_test, ppl_test, acc_test,
                                                                                 bleu_score_g, bleu_score_b))

if __name__ == '__main__':
    train_eval(True)