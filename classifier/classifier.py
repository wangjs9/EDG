import torch
from tqdm import tqdm
from model import CLF, BertCLF, EKCLF, EKBertCLF, Config, evaluate
from utils import prepare_data_seq, make_infinite
from tensorboardX import SummaryWriter

def run(config):
    data_tra, data_val = prepare_data_seq(config)
    if not config.external_knowledge:
        if config.glove:
            model = CLF(config)
        else:
            model = BertCLF(config)
    else:
        if config.glove:
            model = EKCLF(config)
        else:
            model = EKCLF(config)
    if torch.cuda.is_available():
        model.cuda()

    check_iter = 200
    writer = SummaryWriter(log_dir=model.model_dir)
    best_f1 = 0
    patient = 0

    try:
        loader = make_infinite(data_tra)
        for n_iter in tqdm(range(config.train_iter)):
            loss, scores = model(next(loader))
            accuracy, precision, recall, f1 = scores
            writer.add_scalars('loss', {'loss_train': loss}, n_iter)
            writer.add_scalars('accuracy', {'accuracy_train': accuracy}, n_iter)
            writer.add_scalars('precision', {'precision_train': precision}, n_iter)
            writer.add_scalars('recall', {'recall_train': recall}, n_iter)
            writer.add_scalars('f1', {'f1_train': f1}, n_iter)
            print('gobal step {} loss {:2f} accuracy {:2f} f1 {:2f}'.format(n_iter, loss, accuracy, f1))
            if (n_iter + 1) % check_iter == 0:
                model = model.eval()
                acc_val, pre_val, recall_val, f1_val, _ = evaluate(model, data_val, ty="valid")
                writer.add_scalars('accuracy', {'loss_val': acc_val}, n_iter)
                writer.add_scalars('precision', {'precision_val': pre_val}, n_iter)
                writer.add_scalars('recall', {'recall_val': recall_val}, n_iter)
                writer.add_scalars('f1', {'f1_train': f1}, n_iter)
                model = model.train()
                if n_iter < 500:
                    continue
                if best_f1 < f1_val:
                    best_f1 = f1_val
                    patient = 0
                    model.save(n_iter, acc_val, pre_val, recall_val, f1_val)
                else:
                    patient += 1
                if patient > 5:
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Existing from training early')


def main():
    config = Config()
    run(config)


if __name__ == '__main__':
    main()