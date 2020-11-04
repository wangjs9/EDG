import numpy as np
import torch, sys, time, os
from tqdm import tqdm
import config
from data_loader import DataIter
from RTHN import RTHN
from tensorboardX import SummaryWriter

sys.path.append("..")

def find_model_path(save_path):
    list = os.listdir(save_path)
    list = [ele for ele in list if ele[:5] == 'model']
    if list == []:
        return None
    list.sort(key=lambda fn: os.path.getmtime(save_path + fn))
    # model_path = datetime.datetime.fromtimestamp(os.path.getmtime(save_path+list[-1]))
    model_path = os.path.join(save_path, list[-1])
    return model_path

def run():
    if config.log_file_name:
        sys.stdout = open(config.log_file_name, 'w')
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("***********localtime: ", localtime)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    train_DataIter, dev_DataIter = DataIter('../reman/', config.batch_size, device)
    hyper_params = (config.save_path,
                    config.embed_dim,
                    config.embed_dim_pos,
                    config.n_hidden,
                    config.n_layers,
                    config.posembedding_path,
                    config.max_seq_len,
                    config.max_doc_len,
                    config.n_class)
    model = RTHN(*hyper_params, num_heads=config.num_heads, use_mask=True, input_dropout=0.0, word_dropout=config.word_dropout,
                 layer_dropout=config.layer_dropout, attention_dropout=config.attention_dropout, lr=config.lr_main, lr_assist=config.lr_assist,
                 l2_reg=config.l2_reg).to(device)

    try:
        model = model.train()
        check_iter = 500
        best_accuracy, best_precision, best_recall, best_F1 = 0, 0, 0, 0
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path)
        for n_iter in tqdm(range(10000)):
            print(n_iter)
            accuracy_train, precision_train, recall_train, F1_train = model(*next(train_DataIter))

            writer.add_scalars('accuracy', {'accuracy_train': accuracy_train}, n_iter)
            writer.add_scalars('precision', {'precision_train': precision_train}, n_iter)
            writer.add_scalars('recall', {'recall_train': recall_train}, n_iter)
            writer.add_scalars('F1', {'F1_train': F1_train}, n_iter)
            # writer.add_scalars('lr', {'learning_rata': model.optimizer._rate}, n_iter)
            if (n_iter + 1) % check_iter == 0:
                model = model.eval()
                accuracy, precision, recall, F1 = model(*next(train_DataIter), train=False)
                writer.add_scalars('accuracy', {'accuracy_train': accuracy}, n_iter)
                writer.add_scalars('precision', {'precision_train': precision}, n_iter)
                writer.add_scalars('recall', {'recall_train': recall}, n_iter)
                writer.add_scalars('F1', {'F1_train': F1}, n_iter)
                model = model.train()

                if n_iter < 1300:
                    continue
                if F1 >= best_F1:
                    best_F1 = F1
                    patient = 0
                    model.save_model(n_iter, precision, recall, F1)
                else:
                    patient += 1
                if patient > 5:
                    break

                print("accuracy:{:.2f} precision:{:.2f} recall:{:.2f} F1:{:.2f}".format(accuracy, precision, recall, F1))
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    return accuracy, precision, recall, F1


def main():
    accuracy, precision, recall, F1 = run()
    print(" socres:")
    print(accuracy, precision, recall, F1)


if __name__ == '__main__':
    main()