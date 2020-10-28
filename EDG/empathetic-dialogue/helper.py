import numpy as np
import pandas as pd
import ast

nextID = 1
contexts, targets, emotions, situations, dialogcause = [], [], [], [], []
original = pd.read_csv('prompt.csv', sep='\t', header=0)
clauses = pd.read_csv('clause_view.csv', sep='\t', header=0)
curcause = []

label = original['emotion'][0]
iter = clauses.iterrows()
curiter = next(iter)[1]
covID = 0
min_num = 0
while True:
    if min_num > 200:
        break
    min_num += 1
    try:
        situ, ctx, diacuase = [], [], []
        while curiter['context?']:
            situ.append(curiter['clause'])
            curiter = next(iter)[1]

        while covID == curiter['conv_id']:
            user, chatbot, cause = [], [], []
            while not curiter['chatbot?'] and not curiter['context?']:
                user.append(curiter['clause'])
                prob = ast.literal_eval(curiter['prob'])
                if prob == []:
                    curiter = next(iter)[1]
                elif cause == []:
                    cause = prob[:-1]
                    curiter = next(iter)[1]
                else:
                    cause = [cause[i] + prob[i] for i in range(len(cause))]
                    curiter = next(iter)[1]
            if covID != curiter['conv_id']:
                break
            ctx.append(user)
            diacuase.append(cause)
            cause = []
            while curiter['chatbot?']:
                chatbot.append(curiter['clause'])
                prob = ast.literal_eval(curiter['prob'])

                if prob == []:
                    curiter = next(iter)[1]
                elif cause == []:
                    cause = prob[:-1]
                    curiter = next(iter)[1]
                else:
                    cause = [cause[i] + prob[i] for i in range(len(cause))]
                    curiter = next(iter)[1]

            situations.append(situ.copy())
            contexts.append(ctx.copy())
            targets.append(' '.join(chatbot))
            curcause.append(cause)
            emotions.append(label)
            dialogcause.append(diacuase.copy())

            ctx.append(chatbot)
            diacuase.append(cause)

        covID += 1
        label = original['emotion'][covID]

    except StopIteration:
        break

situations = np.array(situations)
contexts = np.array(contexts)
targets = np.array(targets)
emotions = np.array(emotions)
dialogcause = np.array(dialogcause)

assert len(situations) == len(contexts) == len(targets) == len(emotions) == len(dialogcause) == len(curcause)

# np.save('sys_situation_texts.dev.npy', situations)
# np.save('sys_dialog_texts.dev.npy', contexts)
# np.save('sys_target_texts.dev.npy', targets)
# np.save('sys_emotion_texts.dev.npy', emotions)
# np.save('sys_dialogcause_probs.dev.npy', dialogcause)

train_situations = np.array(situations[:150])
train_contexts = np.array(contexts[:150])
train_targets = np.array(targets[:150])
train_emotions = np.array(emotions[:150])
train_dialogcause = np.array(dialogcause[:150])

dev_situations = np.array(situations[150:175])
dev_contexts = np.array(contexts[150:175])
dev_targets = np.array(targets[150:175])
dev_emotions = np.array(emotions[150:175])
dev_dialogcause = np.array(dialogcause[150:175])

test_situations = np.array(situations[175:])
test_contexts = np.array(contexts[175:])
test_targets = np.array(targets[175:])
test_emotions = np.array(emotions[175:])
test_dialogcause = np.array(dialogcause[175:])
#
# np.save('min_situation_texts.train.npy', train_situations)
# np.save('min_dialog_texts.train.npy', train_contexts)
# np.save('min_target_texts.train.npy', train_targets)
# np.save('min_emotion_texts.train.npy', train_emotions)
# np.save('min_dialogcause_probs.train.npy', train_dialogcause)
np.save('min_curcause_probs.train.npy', train_dialogcause)
#
# np.save('min_situation_texts.dev.npy', dev_situations)
# np.save('min_dialog_texts.dev.npy', dev_contexts)
# np.save('min_target_texts.dev.npy', dev_targets)
# np.save('min_emotion_texts.dev.npy', dev_emotions)
# np.save('min_dialogcause_probs.dev.npy', dev_dialogcause)
np.save('min_curcause_probs.dev.npy', dev_dialogcause)
#
# np.save('min_situation_texts.test.npy', test_situations)
# np.save('min_dialog_texts.test.npy', test_contexts)
# np.save('min_target_texts.test.npy', test_targets)
# np.save('min_emotion_texts.test.npy', test_emotions)
# np.save('min_dialogcause_probs.test.npy', test_dialogcause)
np.save('min_curcause_probs.test.npy', test_dialogcause)

