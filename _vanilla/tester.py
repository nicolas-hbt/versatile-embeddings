import torch
from functools import reduce
from dataset import Dataset
import numpy as np
from numpy import genfromtxt
import time
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from torch.utils import data as torch_data
from models import *
from sklearn.utils import shuffle
from collections import defaultdict
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tester:
    def __init__(self, dataset, args, model_inst, valid_or_test):
        self.args = args
        self.name = args.dataset
        self.setting = args.setting
        self.device = args.device
        self.model_name = args.model
        self.proto_kg = dataset.data["prototype"]
        self.unique_classes = len(set(list(self.proto_kg[:,0]) + list(self.proto_kg[:,2])))
        self.proto = args.proto_version
        self.class_mapping = dataset.class_mapping
        self.model = model_inst
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.batch_size = args.batch_size
        self.neg_ratio = args.neg_ratio
        self.sem = args.sem
        self.metric = "all"
        if self.model_name not in ['TuckER', 'ConvE']:
            with open('datasets/' + self.dataset.name + "/pickle/observed_tails_original_kg.pkl", 'rb') as f:
                self.all_possible_ts = pickle.load(f)
            with open('datasets/' + self.dataset.name + "/pickle/observed_heads_original_kg.pkl", 'rb') as f:
                self.all_possible_hs = pickle.load(f)
        else:
            with open('datasets/' + self.dataset.name + "/pickle/observed_tails_inv.pkl", 'rb') as f:
                self.all_possible_ts = pickle.load(f)

    def get_observed_triples(self, train2id, valid2id, test2id):
        all_possible_hs = defaultdict(dict)
        all_possible_ts = defaultdict(dict)
        train2id = torch.as_tensor(train2id.to_numpy(), dtype=torch.int32)
        valid2id = torch.as_tensor(valid2id.to_numpy(), dtype=torch.int32)
        test2id = torch.as_tensor(test2id.to_numpy(), dtype=torch.int32)
        all_triples = torch.cat((train2id, valid2id, test2id))
        X = all_triples.detach().clone()
        for triple in range(X.shape[0]):
            h, r, t = X[triple][0].item(
            ), X[triple][1].item(), X[triple][2].item()
            try:
                all_possible_ts[h][r].append(t)
            except KeyError:
                all_possible_ts[h][r] = [t]

        for triple in range(X.shape[0]):
            h, r, t = X[triple][0].item(
            ), X[triple][1].item(), X[triple][2].item()
            try:
                all_possible_hs[t][r].append(h)
            except KeyError:
                all_possible_hs[t][r] = [h]

        all_possible_ts = dict(all_possible_ts)
        all_possible_hs = dict(all_possible_hs)
        return all_possible_hs, all_possible_ts

    def get_observed_h(self, h, r, t):
        return (
            list(set(self.all_possible_hs[t.item()][r.item()]) - set([h.item()])))

    def get_observed_t(self, h, r, t):
        try:
            return (
                list(set(self.all_possible_ts[h.item()][r.item()]) - set([t.item()])))
        except KeyError:
            return None

    def predictions(self, h, r, t, all_entities):
        heads = h.reshape(-1, 1).repeat(1, all_entities.size()
                                        [1]).to(self.device)
        rels = r.reshape(-1, 1).repeat(1, all_entities.size()
                                       [1]).to(self.device)
        tails = t.reshape(-1, 1).repeat(1, all_entities.size()
                                        [1]).to(self.device)
        triplets = torch.stack((heads, rels, all_entities),
                               dim=2).reshape(-1, 3).to(self.device)
        tails_predictions = self.model.forward(triplets).reshape(1, -1)
        triplets = torch.stack((all_entities, rels, tails),
                               dim=2).reshape(-1, 3).to(self.device)
        heads_predictions = self.model.forward(triplets).reshape(1, -1)
        return heads_predictions.squeeze(), tails_predictions.squeeze()

    def calc_valid_mrr(self):
        schema_CWA = {
            'sem1_h': 0.0,
            'sem3_h': 0.0,
            'sem5_h': 0.0,
            'sem10_h': 0.0,
            'sem1_t': 0.0,
            'sem3_t': 0.0,
            'sem5_t': 0.0,
            'sem10_t': 0.0,
            'sem1': 0.0,
            'sem3': 0.0,
            'sem5': 0.0,
            'sem10': 0.0}
        sem_t_triples_CWA, sem_h_triples_CWA = 0, 0
        ext_CWA = {
            'sem1_h': 0.0,
            'sem3_h': 0.0,
            'sem5_h': 0.0,
            'sem10_h': 0.0,
            'sem1_t': 0.0,
            'sem3_t': 0.0,
            'sem5_t': 0.0,
            'sem10_t': 0.0,
            'sem1': 0.0,
            'sem3': 0.0,
            'sem5': 0.0,
            'sem10': 0.0}
        sem_t_triples_ext, sem_h_triples_ext = 0, 0

        filt_hit1_h, filt_hit1_t, filt_hit3_h, filt_hit3_t, filt_hit5_h, filt_hit5_t, filt_hit10_h, filt_hit10_t = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        h1_h, h1_t, h1_sem1_h, h1_sem1_t = 0, 0, 0, 0
        filt_mrr_h, filt_mrr_t = [], []
        zero_tensor = torch.tensor([0], device=self.device)
        one_tensor = torch.tensor([1], device=self.device)
        X_valid = torch.from_numpy((self.dataset.data[self.valid_or_test]))
        print(X_valid.shape)
        num_ent = self.dataset.num_ent()
        all_entities = torch.arange(end=num_ent, device=self.device).unsqueeze(0)
        start = time.time()
        if self.model_name in ["TuckER", "ConvE"]:
            half_idx = int(X_valid.shape[0] / 2)
            X_valid_tails = X_valid[:half_idx]
            X_valid_tails_inv = X_valid[half_idx:]
            for triple in tqdm(X_valid_tails):
                h, r, t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h, r, t)
                tails_predictions = self.model.get_score(h.to(self.device), r.to(self.device), t.to(self.device)).squeeze()
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                filt_rank_t = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
                filt_mrr_t.append(1.0 / filt_rank_t)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2range2id.keys() and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t'] + \
                                        s1, schema_CWA['sem3_t'] + s3, schema_CWA['sem5_t'] + s5, schema_CWA['sem10_t'] + s10
                                    sem_t_triples_CWA += 1

                    if self.sem == 'extensional' or self.sem == 'both':
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(
                                    indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_t'], ext_CWA['sem3_t'], ext_CWA['sem5_t'], ext_CWA['sem10_t'] = ext_CWA['sem1_t'] + \
                                    s1, ext_CWA['sem3_t'] + s3, ext_CWA['sem5_t'] + s5, ext_CWA['sem10_t'] + s10
                                sem_t_triples_ext += 1
                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_t += torch.where(indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_t += torch.where(indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_t += torch.where(indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_t += torch.where(indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()

            # batch of reversed triples
            for triple in tqdm(X_valid_tails_inv):
                h, r, t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h, r, t)
                tails_predictions = self.model.get_score(h.to(self.device), r.to(self.device), t.to(self.device)).squeeze()
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                filt_rank_h = (indices_tail == t).nonzero(as_tuple=True)[0].item() + 1
                filt_mrr_h.append(1.0 / filt_rank_h)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2range2id.keys(
                        ) and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(
                                    self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h'] + \
                                        s1, schema_CWA['sem3_h'] + s3, schema_CWA['sem5_h'] + s5, schema_CWA['sem10_h'] + s10
                                    sem_h_triples_CWA += 1

                    if self.sem == 'extensional' or self.sem == 'both':
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(
                                    indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_h'], ext_CWA['sem3_h'], ext_CWA['sem5_h'], ext_CWA['sem10_h'] = ext_CWA['sem1_h'] + \
                                    s1, ext_CWA['sem3_h'] + s3, ext_CWA['sem5_h'] + s5, ext_CWA['sem10_h'] + s10
                                sem_h_triples_ext += 1

                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_h += torch.where(indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_h += torch.where(indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_h += torch.where(indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_h += torch.where(indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()

        else:
            for triple in tqdm(X_valid):
                h, r, t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h, r, t)
                rm_idx_h = self.get_observed_h(h, r, t)
                heads_predictions, tails_predictions = self.predictions(
                    h, r, t, all_entities)
                heads_predictions[[rm_idx_h]], tails_predictions[[
                    rm_idx_t]] = -np.inf, -np.inf
                indices_tail, indices_head = tails_predictions.argsort(
                    descending=True), heads_predictions.argsort(descending=True)
                filt_rank_h = (
                    indices_head == h).nonzero(
                    as_tuple=True)[0].item() + 1
                filt_rank_t = (
                    indices_tail == t).nonzero(
                    as_tuple=True)[0].item() + 1
                if indices_head[0].item() == h.item():
                    h1_h += 1
                if indices_tail[0].item() == t.item():
                    h1_t += 1
                filt_mrr_h.append(1.0 / filt_rank_h)
                filt_mrr_t.append(1.0 / filt_rank_t)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2dom2id.keys(
                        ) and self.dataset.r2id2dom2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(
                                    self.dataset.class2id2ent2id[self.dataset.r2id2dom2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_head[:100], r.item(), side='head', k=10, setting='CWA')
                                    schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h'] + \
                                        s1, schema_CWA['sem3_h'] + s3, schema_CWA['sem5_h'] + s5, schema_CWA['sem10_h'] + s10
                                    sem_h_triples_CWA += 1

                        if r.item() in self.dataset.r2id2range2id.keys(
                        ) and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(
                                    self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t'] + \
                                        s1, schema_CWA['sem3_t'] + s3, schema_CWA['sem5_t'] + s5, schema_CWA['sem10_t'] + s10
                                    sem_t_triples_CWA += 1

                    if self.sem == 'extensional' or self.sem == 'both':
                        if r.item() in self.dataset.r2hs.keys():
                            if len(self.dataset.r2hs[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(
                                    indices_head[:10], r.item(), side='head', k=10)
                                ext_CWA['sem1_h'], ext_CWA['sem3_h'], ext_CWA['sem5_h'], ext_CWA['sem10_h'] = ext_CWA['sem1_h'] + \
                                    s1, ext_CWA['sem3_h'] + s3, ext_CWA['sem5_h'] + s5, ext_CWA['sem10_h'] + s10
                                sem_h_triples_ext += 1
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(
                                    indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_t'], ext_CWA['sem3_t'], ext_CWA['sem5_t'], ext_CWA['sem10_t'] = ext_CWA['sem1_t'] + \
                                    s1, ext_CWA['sem3_t'] + s3, ext_CWA['sem5_t'] + s5, ext_CWA['sem10_t'] + s10
                                sem_t_triples_ext += 1

                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_t += torch.where(indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_t += torch.where(indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_t += torch.where(indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_t += torch.where(indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit10_h += torch.where(indices_head[:10] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_h += torch.where(indices_head[:5] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_h += torch.where(indices_head[:3] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_h += torch.where(indices_head[:1] == h.item(), one_tensor, zero_tensor).sum().item()

        filt_mrr_t = np.mean(filt_mrr_t)
        filt_mrr_h = np.mean(filt_mrr_h)
        filt_mrr = (filt_mrr_h + filt_mrr_t) / 2
        if self.model_name in ["TuckER", "ConvE"]:
            filtered_hits_at_10 = (
                filt_hit10_h + filt_hit10_t) / (2 * X_valid_tails.shape[0]) * 100
            filtered_hits_at_5 = (filt_hit5_h + filt_hit5_t) / \
                (2 * X_valid_tails.shape[0]) * 100
            filtered_hits_at_3 = (filt_hit3_h + filt_hit3_t) / \
                (2 * X_valid_tails.shape[0]) * 100
            filtered_hits_at_1 = (filt_hit1_h + filt_hit1_t) / \
                (2 * X_valid_tails.shape[0]) * 100
            filt_h10_t, filt_h5_t, filt_h3_t, filt_h1_t = filt_hit10_t / \
                X_valid_tails.shape[0] * 100, filt_hit5_t / X_valid_tails.shape[0] * 100, filt_hit3_t / X_valid_tails.shape[0] * 100, filt_hit1_t / X_valid_tails.shape[0] * 100
            filt_h10_h, filt_h5_h, filt_h3_h, filt_h1_h = filt_hit10_h / \
                X_valid_tails.shape[0] * 100, filt_hit5_h / X_valid_tails.shape[0] * 100, filt_hit3_h / X_valid_tails.shape[0] * 100, filt_hit1_h / X_valid_tails.shape[0] * 100
        else:
            filtered_hits_at_10 = (
                filt_hit10_h + filt_hit10_t) / (2 * X_valid.shape[0]) * 100
            filtered_hits_at_5 = (filt_hit5_h + filt_hit5_t) / \
                (2 * X_valid.shape[0]) * 100
            filtered_hits_at_3 = (filt_hit3_h + filt_hit3_t) / \
                (2 * X_valid.shape[0]) * 100
            filtered_hits_at_1 = (filt_hit1_h + filt_hit1_t) / \
                (2 * X_valid.shape[0]) * 100
            filt_h10_t, filt_h5_t, filt_h3_t, filt_h1_t = filt_hit10_t / \
                X_valid.shape[0] * 100, filt_hit5_t / X_valid.shape[0] * 100, filt_hit3_t / X_valid.shape[0] * 100, filt_hit1_t / X_valid.shape[0] * 100
            filt_h10_h, filt_h5_h, filt_h3_h, filt_h1_h = filt_hit10_h / \
                X_valid.shape[0] * 100, filt_hit5_h / X_valid.shape[0] * 100, filt_hit3_h / X_valid.shape[0] * 100, filt_hit1_h / X_valid.shape[0] * 100

        logger.info('{} MRR: {}'.format('Filtered', filt_mrr))
        logger.info('{} Hits@{}: {} | Hits@{}: {} | Hits@{}: {}'.format('Filtered',
                    1, filtered_hits_at_1, 3, filtered_hits_at_3, 10, filtered_hits_at_10))

        if (self.sem == 'schema' or self.sem ==
                'both') and self.metric != 'ranks':
            schema_CWA['sem1'] = (
                ((schema_CWA['sem1_h'] / sem_h_triples_CWA) + (
                    schema_CWA['sem1_t'] / sem_t_triples_CWA)) / 2) * 100
            schema_CWA['sem3'] = (
                ((schema_CWA['sem3_h'] / sem_h_triples_CWA) + (
                    schema_CWA['sem3_t'] / sem_t_triples_CWA)) / 2) * 100
            schema_CWA['sem10'] = (
                ((schema_CWA['sem10_h'] / sem_h_triples_CWA) + (
                    schema_CWA['sem10_t'] / sem_t_triples_CWA)) / 2) * 100

            logger.info('[Schema|CWA] Sem@{}: {} | Sem@{}: {} | Sem@{}: {}'.format(1,
                                                                                   (eval("schema_CWA['sem" + str(1) + "']")),
                                                                                   3,
                                                                                   (eval("schema_CWA['sem" + str(3) + "']")),
                                                                                   10,
                                                                                   (eval("schema_CWA['sem" + str(10) + "']"))))


        if (self.sem == 'extensional' or self.sem ==
                'both') and self.metric != 'ranks':
            if True:
                ext_CWA['sem1'] = (((ext_CWA['sem1_h'] / sem_h_triples_ext) +
                                   (ext_CWA['sem1_t'] / sem_t_triples_ext)) / 2) * 100
                ext_CWA['sem3'] = (((ext_CWA['sem3_h'] / sem_h_triples_ext) +
                                   (ext_CWA['sem3_t'] / sem_t_triples_ext)) / 2) * 100
                ext_CWA['sem10'] = (((ext_CWA['sem10_h'] / sem_h_triples_ext) +
                                    (ext_CWA['sem10_t'] / sem_t_triples_ext)) / 2) * 100
                
        if self.metric == 'sem' or self.metric == 'all':
            return filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10, filt_mrr_h, filt_mrr_t, \
                filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, ext_CWA

        else:
            return filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10,\
                filt_mrr_h, filt_mrr_t, filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t

    def test(self):
        schema_CWA = {
            'sem1_h': 0.0,
            'sem3_h': 0.0,
            'sem5_h': 0.0,
            'sem10_h': 0.0,
            'sem1_t': 0.0,
            'sem3_t': 0.0,
            'sem5_t': 0.0,
            'sem10_t': 0.0,
            'sem1': 0.0,
            'sem3': 0.0,
            'sem5': 0.0,
            'sem10': 0.0}
        sem_t_triples_CWA, sem_h_triples_CWA = 0, 0
        ext_CWA = {
            'sem1_h': 0.0,
            'sem3_h': 0.0,
            'sem5_h': 0.0,
            'sem10_h': 0.0,
            'sem1_t': 0.0,
            'sem3_t': 0.0,
            'sem5_t': 0.0,
            'sem10_t': 0.0,
            'sem1': 0.0,
            'sem3': 0.0,
            'sem5': 0.0,
            'sem10': 0.0}
        sem_t_triples_ext, sem_h_triples_ext = 0, 0
        filt_hit1_h, filt_hit1_t, filt_hit3_h, filt_hit3_t, filt_hit5_h, filt_hit5_t, filt_hit10_h, filt_hit10_t = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        filt_mrr_h, filt_mrr_t = [], []
        X_valid_or_test = torch.from_numpy(
            (self.dataset.data[self.valid_or_test]))
        zero_tensor = torch.tensor([0], device=self.device)
        one_tensor = torch.tensor([1], device=self.device)
        num_ent = self.dataset.num_ent()
        all_entities = torch.arange(
            end=num_ent, device=self.device).unsqueeze(0)
        start = time.time()

        if self.model_name in ["TuckER", "ConvE"]:
            half_idx = int(X_valid_or_test.shape[0] / 2)
            X_valid_or_test_tails = X_valid_or_test[:half_idx]
            X_valid_or_test_inv = X_valid_or_test[half_idx:]
            for triple in tqdm(X_valid_or_test_tails):
                h, r, t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h, r, t)
                tails_predictions = self.model.get_score(
                    h.to(
                        self.device), r.to(
                        self.device), t.to(
                        self.device)).squeeze()
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                filt_rank_t = (
                    indices_tail == t).nonzero(
                    as_tuple=True)[0].item() + 1
                filt_mrr_t.append(1.0 / filt_rank_t)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2range2id.keys(
                        ) and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(
                                    self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t'] + \
                                        s1, schema_CWA['sem3_t'] + s3, schema_CWA['sem5_t'] + s5, schema_CWA['sem10_t'] + s10
                                    sem_t_triples_CWA += 1

                    if self.sem == 'extensional' or self.sem == 'both':
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(
                                    indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_t'], ext_CWA['sem3_t'], ext_CWA['sem5_t'], ext_CWA['sem10_t'] = ext_CWA['sem1_t'] + \
                                    s1, ext_CWA['sem3_t'] + s3, ext_CWA['sem5_t'] + s5, ext_CWA['sem10_t'] + s10
                                sem_t_triples_ext += 1
                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_t += torch.where(
                        indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_t += torch.where(
                        indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_t += torch.where(
                        indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_t += torch.where(
                        indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()

            # reversed triples
            for triple in tqdm(X_valid_or_test_inv):
                h, r, t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h, r, t)
                tails_predictions = self.model.get_score(
                    h.to(
                        self.device), r.to(
                        self.device), t.to(
                        self.device)).squeeze()
                tails_predictions[[rm_idx_t]] = - np.inf
                indices_tail = tails_predictions.argsort(descending=True)
                filt_rank_h = (
                    indices_tail == t).nonzero(
                    as_tuple=True)[0].item() + 1
                filt_mrr_h.append(1.0 / filt_rank_h)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2range2id.keys(
                        ) and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(
                                    self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h'] + \
                                        s1, schema_CWA['sem3_h'] + s3, schema_CWA['sem5_h'] + s5, schema_CWA['sem10_h'] + s10
                                    sem_h_triples_CWA += 1

                    if self.sem == 'extensional' or self.sem == 'both':
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(
                                    indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_h'], ext_CWA['sem3_h'], ext_CWA['sem5_h'], ext_CWA['sem10_h'] = ext_CWA['sem1_h'] + \
                                    s1, ext_CWA['sem3_h'] + s3, ext_CWA['sem5_h'] + s5, ext_CWA['sem10_h'] + s10
                                sem_h_triples_ext += 1

                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_h += torch.where(
                        indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_h += torch.where(
                        indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_h += torch.where(
                        indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_h += torch.where(
                        indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()

        else:
            for triple in tqdm(X_valid_or_test):
                h, r, t = triple[0], triple[1], triple[2]
                rm_idx_t = self.get_observed_t(h, r, t)
                rm_idx_h = self.get_observed_h(h, r, t)
                heads_predictions, tails_predictions = self.predictions(
                    h, r, t, all_entities)
                heads_predictions[[rm_idx_h]], tails_predictions[[
                    rm_idx_t]] = -np.inf, -np.inf
                indices_tail, indices_head = tails_predictions.argsort(
                    descending=True), heads_predictions.argsort(descending=True)
                filt_rank_h = (
                    indices_head == h).nonzero(
                    as_tuple=True)[0].item() + 1
                filt_rank_t = (
                    indices_tail == t).nonzero(
                    as_tuple=True)[0].item() + 1
                filt_mrr_h.append(1.0 / filt_rank_h)
                filt_mrr_t.append(1.0 / filt_rank_t)
                if self.metric == 'sem' or self.metric == 'all':
                    if self.sem == 'schema' or self.sem == 'both':
                        if r.item() in self.dataset.r2id2dom2id.keys(
                        ) and self.dataset.r2id2dom2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(
                                    self.dataset.class2id2ent2id[self.dataset.r2id2dom2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_head[:100], r.item(), side='head', k=10, setting='CWA')
                                    schema_CWA['sem1_h'], schema_CWA['sem3_h'], schema_CWA['sem5_h'], schema_CWA['sem10_h'] = schema_CWA['sem1_h'] + \
                                        s1, schema_CWA['sem3_h'] + s3, schema_CWA['sem5_h'] + s5, schema_CWA['sem10_h'] + s10
                                    sem_h_triples_CWA += 1

                        if r.item() in self.dataset.r2id2range2id.keys(
                        ) and self.dataset.r2id2range2id[r.item()] in self.dataset.class2id2ent2id.keys():
                            if len(
                                    self.dataset.class2id2ent2id[self.dataset.r2id2range2id[r.item()]]) >= 10:
                                if self.setting == 'both' or self.setting == 'CWA':
                                    s1, s3, s5, s10 = self.sem_at_k(
                                        indices_tail[:100], r.item(), side='tail', k=10, setting='CWA')
                                    schema_CWA['sem1_t'], schema_CWA['sem3_t'], schema_CWA['sem5_t'], schema_CWA['sem10_t'] = schema_CWA['sem1_t'] + \
                                        s1, schema_CWA['sem3_t'] + s3, schema_CWA['sem5_t'] + s5, schema_CWA['sem10_t'] + s10
                                    sem_t_triples_CWA += 1

                    if self.sem == 'extensional' or self.sem == 'both':
                        if r.item() in self.dataset.r2hs.keys():
                            if len(self.dataset.r2hs[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(
                                    indices_head[:10], r.item(), side='head', k=10)
                                ext_CWA['sem1_h'], ext_CWA['sem3_h'], ext_CWA['sem5_h'], ext_CWA['sem10_h'] = ext_CWA['sem1_h'] + \
                                    s1, ext_CWA['sem3_h'] + s3, ext_CWA['sem5_h'] + s5, ext_CWA['sem10_h'] + s10
                                sem_h_triples_ext += 1
                        if r.item() in self.dataset.r2ts.keys():
                            if len(self.dataset.r2ts[r.item()]) >= 10:
                                s1, s3, s5, s10 = self.sem_at_k_ext(
                                    indices_tail[:10], r.item(), side='tail', k=10)
                                ext_CWA['sem1_t'], ext_CWA['sem3_t'], ext_CWA['sem5_t'], ext_CWA['sem10_t'] = ext_CWA['sem1_t'] + \
                                    s1, ext_CWA['sem3_t'] + s3, ext_CWA['sem5_t'] + s5, ext_CWA['sem10_t'] + s10
                                sem_t_triples_ext += 1

                if self.metric == 'ranks' or self.metric == 'all':
                    filt_hit10_t += torch.where(
                        indices_tail[:10] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_t += torch.where(
                        indices_tail[:5] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_t += torch.where(
                        indices_tail[:3] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_t += torch.where(
                        indices_tail[:1] == t.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit10_h += torch.where(
                        indices_head[:10] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit5_h += torch.where(
                        indices_head[:5] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit3_h += torch.where(
                        indices_head[:3] == h.item(), one_tensor, zero_tensor).sum().item()
                    filt_hit1_h += torch.where(
                        indices_head[:1] == h.item(), one_tensor, zero_tensor).sum().item()

        print(time.time() - start)
        filt_mrr_t = np.mean(filt_mrr_t)
        filt_mrr_h = np.mean(filt_mrr_h)
        filt_mrr = (filt_mrr_h + filt_mrr_t) / 2
        if self.model_name in ["TuckER", "ConvE"]:
            filtered_hits_at_10 = (filt_hit10_h + filt_hit10_t) / (
                X_valid_or_test_tails.shape[0] + X_valid_or_test_inv.shape[0]) * 100
            filtered_hits_at_5 = (filt_hit5_h + filt_hit5_t) / (
                X_valid_or_test_tails.shape[0] + X_valid_or_test_inv.shape[0]) * 100
            filtered_hits_at_3 = (filt_hit3_h + filt_hit3_t) / (
                X_valid_or_test_tails.shape[0] + X_valid_or_test_inv.shape[0]) * 100
            filtered_hits_at_1 = (filt_hit1_h + filt_hit1_t) / (
                X_valid_or_test_tails.shape[0] + X_valid_or_test_inv.shape[0]) * 100
            filt_h10_t, filt_h5_t, filt_h3_t, filt_h1_t = filt_hit10_t / \
                X_valid_or_test_tails.shape[0] * 100, filt_hit5_t / X_valid_or_test_tails.shape[0] * 100, filt_hit3_t / X_valid_or_test_tails.shape[0] * 100, filt_hit1_t / X_valid_or_test_tails.shape[0] * 100
            filt_h10_h, filt_h5_h, filt_h3_h, filt_h1_h = filt_hit10_h / \
                X_valid_or_test_inv.shape[0] * 100, filt_hit5_h / X_valid_or_test_inv.shape[0] * 100, filt_hit3_h / X_valid_or_test_inv.shape[0] * 100, filt_hit1_h / X_valid_or_test_inv.shape[0] * 100
        else:
            filtered_hits_at_10 = (
                filt_hit10_h + filt_hit10_t) / (2 * X_valid_or_test.shape[0]) * 100
            filtered_hits_at_5 = (filt_hit5_h + filt_hit5_t) / \
                (2 * X_valid_or_test.shape[0]) * 100
            filtered_hits_at_3 = (filt_hit3_h + filt_hit3_t) / \
                (2 * X_valid_or_test.shape[0]) * 100
            filtered_hits_at_1 = (filt_hit1_h + filt_hit1_t) / \
                (2 * X_valid_or_test.shape[0]) * 100
            filt_h10_t, filt_h5_t, filt_h3_t, filt_h1_t = filt_hit10_t / \
                X_valid_or_test.shape[0] * 100, filt_hit5_t / X_valid_or_test.shape[0] * 100, filt_hit3_t / X_valid_or_test.shape[0] * 100, filt_hit1_t / X_valid_or_test.shape[0] * 100
            filt_h10_h, filt_h5_h, filt_h3_h, filt_h1_h = filt_hit10_h / \
                X_valid_or_test.shape[0] * 100, filt_hit5_h / X_valid_or_test.shape[0] * 100, filt_hit3_h / X_valid_or_test.shape[0] * 100, filt_hit1_h / X_valid_or_test.shape[0] * 100

        logger.info('{} MRR: {}'.format('Filtered', filt_mrr))
        logger.info('{} Hits@{}: {}'.format('Filtered', 1, filtered_hits_at_1))
        logger.info('{} Hits@{}: {}'.format('Filtered', 3, filtered_hits_at_3))
        logger.info('{} Hits@{}: {}'.format('Filtered',
                    10, filtered_hits_at_10))

        if self.sem == 'schema' or self.sem == 'both':
            if self.setting == 'both' or self.setting == 'CWA':
                schema_CWA['sem1'] = (
                    ((schema_CWA['sem1_h'] / sem_h_triples_CWA) + (
                        schema_CWA['sem1_t'] / sem_t_triples_CWA)) / 2) * 100
                schema_CWA['sem3'] = (
                    ((schema_CWA['sem3_h'] / sem_h_triples_CWA) + (
                        schema_CWA['sem3_t'] / sem_t_triples_CWA)) / 2) * 100
                schema_CWA['sem10'] = (
                    ((schema_CWA['sem10_h'] / sem_h_triples_CWA) + (
                        schema_CWA['sem10_t'] / sem_t_triples_CWA)) / 2) * 100
                for k in [1, 3, 10]:
                    logger.info('[Schema|CWA] Sem@{}: {}'.format(k,
                                (eval("schema_CWA['sem" + str(k) + "']"))))

        if self.sem == 'extensional' or self.sem == 'both':
            ext_CWA['sem1'] = (((ext_CWA['sem1_h'] / sem_h_triples_ext) +
                               (ext_CWA['sem1_t'] / sem_t_triples_ext)) / 2) * 100
            ext_CWA['sem3'] = (((ext_CWA['sem3_h'] / sem_h_triples_ext) +
                               (ext_CWA['sem3_t'] / sem_t_triples_ext)) / 2) * 100
            ext_CWA['sem10'] = (((ext_CWA['sem10_h'] / sem_h_triples_ext) +
                                (ext_CWA['sem10_t'] / sem_t_triples_ext)) / 2) * 100

        if self.metric == 'sem' or self.metric == 'all':
            return filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10, filt_mrr_h, filt_mrr_t, \
                filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, ext_CWA

        else:
            return filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10,\
                filt_mrr_h, filt_mrr_t, filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t


    def sem_at_k(self, preds, rel, side='head', k=10, setting='CWA'):
        valid_types = []
        for pred in preds.tolist():
            if len(valid_types) == 10:
                return ((valid_types[0]),
                        np.mean(valid_types[:3]),
                        np.mean(valid_types[:5]),
                        np.mean(valid_types[:10]))
            else:
                try:
                    classes = self.dataset.instype_all[pred]
                    if side == 'head':
                        dom = self.dataset.r2id2dom2id[rel]
                        valid_types.append(1 if dom in classes else 0)
                    elif side == 'tail':
                        rang = self.dataset.r2id2range2id[rel]
                        valid_types.append(1 if rang in classes else 0)
                except KeyError:
                    valid_types.append(0)

    def sem_at_k_ext(self, preds, rel, side='head', k=10):
        preds = preds[:10]
        if side =='head':
            seen_entities = self.dataset.r2hs[rel]
        else:
            seen_entities = self.dataset.r2ts[rel]
        valid_types = []
        for pred in preds.tolist():
            valid_types.append(1 if pred in (seen_entities) else 0)
            
        return ((valid_types[0]), np.mean(valid_types[:3]), np.mean(valid_types[:5]), np.mean(valid_types[:10]))