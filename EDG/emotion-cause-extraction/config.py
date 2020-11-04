import argparse

parser = argparse.ArgumentParser()
# embedding parameters ##
parser.add_argument("--embed_dim", type=int, default=768, help="dimension of word embedding")
parser.add_argument("--embed_dim_pos", type=int, default=56, help="dimension of position embedding")
# input struct ##
parser.add_argument("--max_doc_len", type=int, default=32, help="max number of tokens per documents")
parser.add_argument("--max_seq_len", type=int, default=50, help="max number of tokens per sentence")
# model struct ##
parser.add_argument("--n_hidden", type=int, default=100, help="number of hidden unit")
parser.add_argument("--n_class", type=int, default=2, help="number of distinct class")
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
parser.add_argument("--log_file_name", type=str, default='', help="name of log file")
parser.add_argument("--data_path", type=str, default='../reman', help="name of log file")
parser.add_argument("--save_path", type=str, default='../reman/save', help="name of save file")
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
parser.add_argument("--batch_size", type=int, default=32, help="number of example per batch")
parser.add_argument("--lr_main", type=int, default=0.001, help="learning rate")
parser.add_argument("--lr_assist", type=int, default=1e-5, help="learning rate")
parser.add_argument("--l2_reg", type=int, default=1e-5, help="l2 regularization")
parser.add_argument("--num_heads", type=int, default=8, help="the num heads of attention")
parser.add_argument("--n_layers", type=int, default=1, help="the layers of transformer beside main")
parser.add_argument("--word_dropout", type=int, default=0.5, help="word embedding training dropout keep prob")
parser.add_argument("--layer_dropout", type=int, default=0, help="softmax layer dropout keep prob")
parser.add_argument("--attention_dropout", type=int, default=0.5, help="transformer dropout keep prob")

arg = parser.parse_args()

embed_dim = arg.embed_dim
embed_dim_pos = arg.embed_dim_pos

max_doc_len = arg.max_doc_len
max_seq_len = arg.max_seq_len

n_hidden = arg.n_hidden
n_class = arg.n_class

log_file_name = arg.log_file_name
save_path = arg.save_path

batch_size = arg.batch_size
lr_main = arg.lr_main
lr_assist = arg.lr_main
l2_reg = arg.l2_reg
num_heads = arg.num_heads
n_layers = arg.n_layers
word_dropout = arg.word_dropout
layer_dropout = arg.layer_dropout
attention_dropout = arg.attention_dropout


posembedding_path = '../reman/embedding_pos.txt'