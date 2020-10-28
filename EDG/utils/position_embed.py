import numpy as np
import pickle as pk
def load_w2v(embedding_dim_pos):
    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(
        loc=0.0, scale=0.1, size=embedding_dim_pos
    )) for i in range(-64, 0)])

    embedding_pos = np.array(embedding_pos)
    pk.dump(embedding_pos, open('embedding_pos.txt', 'wb'))
    print("embedding_pos.shape: {}".format(embedding_pos.shape))
    return embedding_pos

embedding_pos = load_w2v(300)
print(embedding_pos)
