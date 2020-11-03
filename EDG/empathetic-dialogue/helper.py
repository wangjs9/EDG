import numpy as np
import pandas as pd
import ast, copy

nextID = 1
contexts, targets, emotions, situations, dialogcause = [], [], [], [], []
# original = pd.read_csv('prompt.csv', sep='\t', header=0)
# clauses = pd.read_csv('clause_view.csv', sep='\t', header=0)
original = pd.read_csv('valid_prompt.csv', sep='\t', header=0)
clauses = pd.read_csv('valid_clause_view.csv', sep='\t', header=0)
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
            counter = 1
            cazlen = curiter['clause_no']
            while not curiter['chatbot?'] and not curiter['context?']:
                user.append(curiter['clause'])
                prob = ast.literal_eval(curiter['prob'])
                if prob == []:
                    curiter = next(iter)[1]
                elif cause == []:
                    cause = prob[:cazlen]
                    curiter = next(iter)[1]

                else:
                    cause = [cause[i] + prob[i] for i in range(cazlen)]
                    curiter = next(iter)[1]
                    counter += 1

            cause = [prob/counter for prob in cause]
            if covID != curiter['conv_id']:
                break
            ctx.append(user)
            diacuase.append(cause)

            ### for chatbot
            cause = []
            counter = 1
            cazlen = curiter['clause_no']
            while curiter['chatbot?']:
                chatbot.append(curiter['clause'])
                prob = ast.literal_eval(curiter['prob'])

                if prob == []:
                    curiter = next(iter)[1]
                elif cause == []:
                    cause = prob[:cazlen]
                    curiter = next(iter)[1]
                else:
                    cause = [cause[i] + prob[i] for i in range(cazlen)]
                    curiter = next(iter)[1]
                    counter += 1

            cause = [prob / counter for prob in cause]

            situations.append(copy.deepcopy(situ))
            contexts.append(copy.deepcopy(ctx))
            targets.append(' '.join(chatbot))
            curcause.append(copy.deepcopy(cause))
            emotions.append(label)
            dialogcause.append(copy.deepcopy(diacuase))

            ctx.append(copy.deepcopy(chatbot))
            diacuase.append(copy.deepcopy(cause))

        covID += 1
        label = original['emotion'][covID]

    except StopIteration:
        break

situations = np.array(situations)
contexts = np.array(contexts)
targets = np.array(targets)
emotions = np.array(emotions)
dialogcause = np.array(dialogcause)
curcause = np.array(curcause)

assert len(situations) == len(contexts) == len(targets) == len(emotions) == len(dialogcause) == len(curcause)

#### real data

# np.save('sys_situation_texts.train.npy', situations)
# np.save('sys_dialog_texts.train.npy', contexts)
# np.save('sys_target_texts.train.npy', targets)
# np.save('sys_emotion_texts.train.npy', emotions)
# np.save('sys_dialogcause_probs.train.npy', dialogcause)
# np.save('sys_curcause_probs.train.npy', curcause)

np.save('sys_situation_texts.dev.npy', situations)
np.save('sys_dialog_texts.dev.npy', contexts)
np.save('sys_target_texts.dev.npy', targets)
np.save('sys_emotion_texts.dev.npy', emotions)
np.save('sys_dialogcause_probs.dev.npy', dialogcause)
np.save('sys_curcause_probs.dev.npy', curcause)

np.save('sys_situation_texts.test.npy', situations)
np.save('sys_dialog_texts.test.npy', contexts)
np.save('sys_target_texts.test.npy', targets)
np.save('sys_emotion_texts.test.npy', emotions)
np.save('sys_dialogcause_probs.test.npy', dialogcause)
np.save('sys_curcause_probs.test.npy', curcause)

#### use min data to test the ability of your model

# train_situations = np.array(situations[:150])
# train_contexts = np.array(contexts[:150])
# train_targets = np.array(targets[:150])
# train_emotions = np.array(emotions[:150])
# train_dialogcause = np.array(dialogcause[:150])
# train_curcause = np.array(curcause[:150])
#
# dev_situations = np.array(situations[150:175])
# dev_contexts = np.array(contexts[150:175])
# dev_targets = np.array(targets[150:175])
# dev_emotions = np.array(emotions[150:175])
# dev_dialogcause = np.array(dialogcause[150:175])
# dev_curcause = np.array(curcause[150:175])
#
# test_situations = np.array(situations[175:])
# test_contexts = np.array(contexts[175:])
# test_targets = np.array(targets[175:])
# test_emotions = np.array(emotions[175:])
# test_dialogcause = np.array(dialogcause[175:])
# test_curcause = np.array(curcause[175:])
#
# np.save('min_situation_texts.train.npy', train_situations)
# np.save('min_dialog_texts.train.npy', train_contexts)
# np.save('min_target_texts.train.npy', train_targets)
# np.save('min_emotion_texts.train.npy', train_emotions)
# np.save('min_dialogcause_probs.train.npy', train_dialogcause)
# np.save('min_curcause_probs.train.npy', train_curcause)
#
# np.save('min_situation_texts.dev.npy', dev_situations)
# np.save('min_dialog_texts.dev.npy', dev_contexts)
# np.save('min_target_texts.dev.npy', dev_targets)
# np.save('min_emotion_texts.dev.npy', dev_emotions)
# np.save('min_dialogcause_probs.dev.npy', dev_dialogcause)
# np.save('min_curcause_probs.dev.npy', dev_curcause)
#
# np.save('min_situation_texts.test.npy', test_situations)
# np.save('min_dialog_texts.test.npy', test_contexts)
# np.save('min_target_texts.test.npy', test_targets)
# np.save('min_emotion_texts.test.npy', test_emotions)
# np.save('min_dialogcause_probs.test.npy', test_dialogcause)
# np.save('min_curcause_probs.test.npy', test_curcause)

