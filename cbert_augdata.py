from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import shutil
import logging
import argparse
import random
from tqdm import tqdm, trange
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW

import cbert_utils
#import train_text_classifier
import pandas as pd

#PYTORCH_PRETRAINED_BERT_CACHE = ".pytorch_pretrained_bert"

"""initialize logger"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

"""cuda or cpu"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_ids_to_str(ids, tokenizer):
    """converts token_ids into str."""
    tokens = []
    for token_id in ids:
        token = tokenizer._convert_id_to_token(token_id)
        tokens.append(token)
    outputs = cbert_utils.rev_wordpiece(tokens)
    return outputs

def main():
    parser = argparse.ArgumentParser()

    ## required parameters
    parser.add_argument("--data_dir", default="datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="aug_data", type=str,
                        help="The output dir for augmented dataset")
    parser.add_argument("--save_model_dir", default="CBERT", type=str,
                        help="The cache dir for saved model.")
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="The path of pretrained bert model.")
    parser.add_argument("--task_name", default="TREC",type=str,
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=8.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', default=42, type=int, 
                        help="random seed for initialization")
    parser.add_argument('--sample_num', default=1, type=int,
                        help="sample number")
    parser.add_argument('--sample_ratio', default=6, type=int,
                        help="sample ratio")
    parser.add_argument('--gpu', default=0, type=int,
                        help="gpu id")
    parser.add_argument('--temp', default=1.0, type=float,
                        help="temperature")
    

    args = parser.parse_args()
    #with open("global.config", 'r') as f:
    #   configs_dict = json.load(f)

    #args.task_name = configs_dict.get("dataset")
    #args.output_dir = args.output_dir + '_{}_{}_{}_{}'.format(args.sample_num, args.sample_ratio, args.gpu, args.temp)
    #print(args)
    
    """prepare processors"""
    AugProcessor = cbert_utils.AugProcessor()
    processors = {
        ## you can add your processor here
        "TREC": AugProcessor,
        "stsa5": AugProcessor,
        "stsa2": AugProcessor,
        "mpqa": AugProcessor,
        "rt-polarity": AugProcessor,
        "subj": AugProcessor,
    }

    task_name = args.task_name
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]
    label_list = processor.get_labels(task_name)

    ## prepare for model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    args.data_dir = os.path.join(args.data_dir, task_name)
    args.output_dir = os.path.join(args.output_dir, task_name)


    ## prepare for training
    train_examples = processor.get_train_examples(args.data_dir)
    train_features, num_train_steps, train_dataloader = \
        cbert_utils.construct_train_dataloader(train_examples, label_list, args.max_seq_length, 
        args.train_batch_size, args.num_train_epochs, tokenizer, device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    model_dir = os.path.join(args.save_model_dir, task_name)

    MASK_id = cbert_utils.convert_tokens_to_ids(['[MASK]'], tokenizer)[0]

    origin_train_path = os.path.join(args.output_dir, "train_origin.tsv")
    save_train_path = os.path.join(args.output_dir, "train.tsv")
    shutil.copy(origin_train_path, save_train_path)           #将原数据拷贝到增强数据的文件夹
    #best_test_acc = train_text_classifier.train("aug_data_{}_{}_{}_{}".format(args.sample_num, args.sample_ratio, args.gpu, args.temp))
    #print("before augment best acc:{}".format(best_test_acc))


    torch.cuda.empty_cache()
    weights_path = model_dir+"/BertForMaskedLM_{}_epoch_{}".format(task_name, int(0.8*args.num_train_epochs))
    model = torch.load(weights_path)                               #使用0.8*train_epoch的参数来预测，以免过拟合
    model.cuda()
    
    #save_train_file = open('aug_data/stsa2/train.tsv', 'a',newline="")  这个语句无法写入
    #tsv_writer = csv.writer(save_train_file, delimiter='\t')
    for i in [1,2]:
        print("Augment",i,":")
        for _, batch in enumerate(tqdm(train_dataloader,desc="Iters")):
            
            model.eval()
            batch = tuple(t.cuda() for t in batch)
            init_ids, _, input_mask, segment_ids, _ = batch       #原始句子id，句子长度向量，句子标签参加预测
            input_lens = [sum(mask).item() for mask in input_mask]      #每个句子长度
            
            masked_idx = [np.random.randint(0, l, max(l//args.sample_ratio, 1)) for l in input_lens]  #随机抽取被遮蔽的位置
            for ids, idx in zip(init_ids, masked_idx):
                ids[idx] = MASK_id                                                         #将被遮蔽位置替换成[MASK]的id103
            predictions = model(init_ids, input_mask, segment_ids)                         #batch size * max seq length * per word logits
            predictions = torch.nn.functional.softmax(predictions[0]/args.temp, dim=2)     #batch size * max seq length * per word probability
            for ids, idx, preds, seg in zip(init_ids, masked_idx, predictions, segment_ids):
                preds = torch.multinomial(preds, args.sample_num, replacement=True)[idx]   #对被遮蔽位置选出概率最大的那个词
                if len(preds.size()) == 2:                                                 #有多个预测词时转置
                    preds = torch.transpose(preds, 0, 1)
                for pred in preds:
                    ids[idx] = pred                                                        #将id转回句子并写入
                    new_str = convert_ids_to_str(ids.cpu().numpy(), tokenizer)
                    df=pd.DataFrame([[new_str, seg[0].item()]])
                    df.to_csv(save_train_path, sep='\t', mode='a',index=False, header=False)

            torch.cuda.empty_cache()
        predictions = predictions.detach().cpu()
    model.cpu()
    torch.cuda.empty_cache()
    #best_test_acc = train_text_classifier.train("aug_data_{}_{}_{}_{}".format(args.sample_num, args.sample_ratio, args.gpu, args.temp))
    #print("epoch {} augment best acc:{}".format(e, best_test_acc))

if __name__ == "__main__":
    main()
