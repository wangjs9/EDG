import numpy as np
import pickle as pk

train_context = np.load('sys_dialog_texts.train.npy', allow_pickle=True)
# >>>>>>>>>> causes from historical utterances >>>>>>>>>> #
# train_concause = np.load('sys_dialog_cause_scores.train.npy', allow_pickle=True)
# >>>>>>>>>> next expected utterance from the bot >>>>>>>>>> #
train_target = np.load('sys_target_texts.train.npy', allow_pickle=True)
# >>>>>>>>>> emotions of the conversation >>>>>>>>>> #
train_emotion = np.load('sys_emotion_texts.train.npy', allow_pickle=True)
# >>>>>>>>>> prompts of the conversation >>>>>>>>>> #
train_situation = np.load('sys_situation_texts.train.npy', allow_pickle=True)
# >>>>>>>>>> causes from prompts of the conversation >>>>>>>>>> #
# train_situcause = np.load('sys_situation_cause_scores.train.npy', allow_pickle=True)

print(train_situation)
x=input()
print(train_context)
