import os
import sys
import numpy as np
import torch
import tqdm
import json
import evaluation
import lib.utils as utils
import datasets.data_loader as data_loader
from lib.config import cfg

# model evaluation
class Evaler(object):
    def __init__(
        self,
        eval_ids,
        att_feats,
        eval_annfile
    ):
        super(Evaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        # './mscoco/txt/coco_val_image_id.txt'  Karpathy验证集  5K张图像
        # './mscoco/txt/coco_test_image_id.txt' Karpathy测试集  5K张图像
        # './mscoco/txt/coco_test4w_image_id.txt' MSCOCO在线测试集 4W张图像
        
        # 读取txt文件，读取的为image_ids的list
        # self.eval_ids = np.array(utils.load_ids(eval_ids))
        
        # 端到端训练时，直接读取annotation的json文件，其中包含了图像id和路径
        # 读取json文件，读取的为{image_id: image_path}的dict
        with open(eval_ids, 'r') as f:
            self.ids2path = json.load(f)           # dict {image_id: image_path}
            self.eval_ids = np.array(list(self.ids2path.keys()))  # array of str
        
        self.eval_loader = data_loader.load_val(eval_ids, att_feats)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)
        
    def __call__(self, model, rname):
        model.eval()
        
        results = []
        with torch.no_grad():
            for _, (indices, att_feats) in enumerate(tqdm.tqdm(self.eval_loader, desc=rname)):
                ids = self.eval_ids[indices]
                att_feats = att_feats.cuda()
                
                out = model.module.decode(att_feats)
                _, seq = torch.max(out, -1)

                sents = utils.decode_sequence(self.vocab, seq.data)
                sents = utils.clip_chongfu(sents)
                
                for sid, sent in enumerate(sents):
                    # result {'image_id': ***, 'caption': 'word1 word2 word3 ...'}
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent}
                    # print(result)
                    results.append(result)
                
        # COCO evaluation
        # eval_res = self.evaler.eval(results)
        # w/o spice
        eval_res = self.evaler.eval_no_spice(results)

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))

        model.train()
        return eval_res