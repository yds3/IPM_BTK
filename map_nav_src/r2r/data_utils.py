import os
import json
import numpy as np


def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, '%s_%s_enc_goals.json' % (dataset.upper(), split))
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, '%s_%s_enc_xlmr.json' % (dataset.upper(), split))
            else:
                raise NotImplementedError('unspported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

            if not is_test:
                if dataset == 'r4r' and split == 'val_unseen':
                    ridxs = np.random.permutation(len(new_data))[:200]
                    new_data = [new_data[ridx] for ridx in ridxs]
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)
        # Join
        data += new_data
    return data




def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            # print(f"new_item['instruction']{new_item['instruction']}")
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]           
            new_item['goal'] = item['goals_encodings'][j][:max_instr_len]   

            new_item['path_id'] = item['path_id']
            scan = item['scan']
            id_path = new_item['path_id']
            new_item['img_know_dir'] = f'{scan}_{id_path}_goals_{j}'
            # print(f"new_item['img_know_dir']{new_item['img_know_dir']}")
            del new_item['instructions']
            del new_item['instr_encodings']

            del new_item['goals_encodings']
            data.append(new_item)
    return data


