from dataset import Dataset
from tester import Tester
from models import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from sklearn.utils import shuffle
torch.manual_seed(7)
random.seed(7)

class JointTrainer:
    def __init__(self, dataset, model_name_proto, model_name_inst, args):
        self.device = args.device
        self.factor = args.factor
        self.model_name_proto = model_name_proto
        self.model_name_inst = model_name_inst
        self.monitor = args.monitor_metrics
        self.use_proto = args.use_proto
        self.weights = args.weights
        self.dataset = dataset
        self.args = args
        self.switch = args.switch
        self.aggreg_option = args.aggreg_option
        self.aggreg_operator = args.aggreg_operator
        self.validate_every = args.validate_every
        self.burn_in = args.burn_in
        self.patience = args.patience
        self.distribution = args.distribution
        self.erase_proto = args.erase_proto
        self.erase_inst = args.erase_inst
        self.hard_reset_proto = args.hard_reset_proto
        self.proto_kg = dataset.data["prototype"]
        self.unique_classes = len(set(list(self.proto_kg[:,0]) + list(self.proto_kg[:,2])))
        self.class_mapping = dataset.class_mapping

        if self.model_name_inst == 'TransE':
            self.model_inst = TransE(dataset.num_ent(), dataset.num_rel(), args.dim, self.device)
            if self.use_proto:
                self.model_proto = TransE(self.unique_classes, dataset.num_rel(), args.dim, self.device)

        elif self.model_name_inst == 'DistMult':
            self.model_inst = DistMult(dataset.num_ent(), dataset.num_rel(), args.dim, self.device)
            if self.use_proto:
                self.model_proto = DistMult(self.unique_classes, dataset.num_rel(), args.dim, self.device)

        elif self.model_name_inst == 'ComplEx':
            self.model_inst = ComplEx(dataset.num_ent(), dataset.num_rel(), args.dim, self.device)
            if self.use_proto:
                self.model_proto = ComplEx(self.unique_classes, dataset.num_rel(), args.dim, self.device)
                
        elif self.model_name_inst == 'TuckER':
            self.model_inst = TuckER(dataset.num_ent(), dataset.num_rel(), args, self.device)
            if self.use_proto:
                self.model_proto = TuckER(self.unique_classes, dataset.num_rel(), args, self.device)

        elif self.model_name_inst == 'ConvE':
            self.model_inst = ConvE(dataset.num_ent(), dataset.num_rel(), args, self.device)
            if self.use_proto:
                self.model_proto = ConvE(self.unique_classes, dataset.num_rel(), args, self.device)

        self.epoch = 0
        self.need_inverse = self.dataset.need_inverse
        print(self.need_inverse)

    def split_negatives(self, neg_batch):
        neg_sem, neg_unsem = neg_batch[neg_batch[:,4]==1][:,:4], neg_batch[neg_batch[:,4]==0][:,:4]
        return neg_sem, neg_unsem
        
    def train(self):
        if self.use_proto:
            self.model_proto.train()
            optimizer_proto = torch.optim.AdamW(self.model_proto.parameters(), lr=self.args.lr, weight_decay=0.0)     
        self.model_inst.train()
        optimizer_inst = torch.optim.AdamW(
            self.model_inst.parameters(),
            lr=self.args.lr,
            weight_decay=0.0
        )

        if not self.args.resume_training:
            start = 1
        else:
            start = self.args.resume_epoch + 1
        best_mrr = - 1.0
        best_epoch = 0
        proto_bs = 128
        for self.epoch in range(start, self.switch + 1):
            last_batch = False
            nb_batch = 0
            if self.model_name_proto in ["TuckER", "ConvE"]:
                while not last_batch:
                    nb_batch += 1
                    batch_pos = self.dataset.next_pos_batch_prototype_inv(proto_bs)
                    last_batch = self.dataset.was_last_batch_proto_inv()
                    optimizer_proto.zero_grad()
                    training_loss = self.model_proto._bce(batch_pos)
                    training_loss.backward()
                    optimizer_proto.step()
            else:
                while not last_batch:
                    nb_batch += 1
                    batch = self.dataset.next_batch_prototype(proto_bs, self.args.neg_ratio, self.device)
                    last_batch = self.dataset.was_last_batch_proto()
                    optimizer_proto.zero_grad()
                    scores = self.model_proto.forward(batch)
                    if last_batch:
                        nb_pos = self.dataset.data["prototype"].shape[0] % proto_bs
                        batch_pos = batch[:nb_pos]
                        pos_scores, neg_scores = scores[:nb_pos], scores[nb_pos:]
                    else:
                        batch_pos = batch[:proto_bs]
                        pos_scores, neg_scores = scores[:proto_bs], scores[proto_bs:]
                    if self.model_name_proto in ["ComplEx", "SimplE"]:
                        training_loss = self.model_proto._softplus(pos_scores, neg_scores, self.args.neg_ratio)
                    elif self.model_name_proto in ["TransE", "TransH", "DistMult"]:
                        training_loss = self.model_proto._pairwise_hinge(pos_scores, neg_scores, self.args.neg_ratio, gamma=2.0)
                    if self.args.reg != 0.0:
                        training_loss += self.args.reg * self.model_proto._regularization(batch[:, 0].to(
                            self.device), batch[:, 1].to(self.device), batch[:, 2].to(self.device))
                    training_loss.backward()
                    optimizer_proto.step()

        # Init
        self.init_inst_with_proto()

        for self.epoch in range(start, self.args.ne + 1):
            last_batch = False
            nb_batch = 0
            if self.model_name_inst in ["TuckER", "ConvE"]:
                while not last_batch:
                    nb_batch += 1
                    batch_pos = self.dataset.next_pos_batch_inv(128)
                    last_batch = self.dataset.was_last_batch_inst_inv()
                    optimizer_inst.zero_grad()
                    training_loss = self.model_inst._bce(batch_pos)
                    training_loss.backward()
                    optimizer_inst.step()
            else:
                while not last_batch:
                    nb_batch += 1
                    batch_pos, batch_neg, _ = self.dataset.next_batch(self.args.batch_size, neg_ratio = self.args.neg_ratio, device = self.device, cache = None)
                    last_batch = self.dataset.was_last_batch_inst()
                    optimizer_inst.zero_grad()
                    pos_scores = self.model_inst.forward(batch_pos)
                    neg_scores = self.model_inst.forward(batch_neg)
                    if self.model_name_inst in ["ComplEx", "SimplE"]:
                        training_loss = self.model_inst._softplus(pos_scores, neg_scores, self.args.neg_ratio)
                    elif self.model_name_inst in ["TransE", "TransH", "DistMult"]:
                        training_loss = self.model_inst._pairwise_hinge(pos_scores, neg_scores, self.args.neg_ratio, gamma=2.0)
                    if self.args.reg != 0.0:
                        batch = np.append(batch_pos, batch_neg[:,[0,1,2]], axis=0)
                        batch = torch.tensor(batch)
                        training_loss += self.args.reg * self.model_inst._regularization(batch[:, 0].to(
                            self.device), batch[:, 1].to(self.device), batch[:, 2].to(self.device))
                    training_loss.backward()
                    optimizer_inst.step()
            if self.epoch % self.validate_every == 0:
                self.save_model_inst(self.model_inst, self.epoch)
            # Validation
            if self.monitor == 1 and self.epoch % self.validate_every == 0 and self.epoch >= self.burn_in:
                print('epoch', self.epoch)
                if self.model_name_inst in ["TuckER", "ConvE"]:
                    tester = Tester(self.dataset, self.args, self.model_inst, "valid_inv")
                else:
                    tester = Tester(self.dataset, self.args, self.model_inst, "valid")
                filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10, filt_mrr_h, filt_mrr_t, \
                filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, ext_CWA = tester.calc_valid_mrr()
                if filt_mrr > best_mrr:
                    best_mrr = filt_mrr
                    best_epoch = self.epoch
                    best_h1, best_h3, best_h10 = filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_10
                    best_sem1, best_sem3, best_sem10 = schema_CWA['sem1'], schema_CWA['sem3'], schema_CWA['sem10']
        if self.monitor == 1:
            # Save best epoch results in a file
            f = open("results-valid-"+ self.model_name_inst +"-"+ self.dataset.name +"-proto" + str(self.dataset.proto) +".txt", "w")
            f.write("Model:" + self.model_name_inst + "\nDataset:" + self.dataset.name + "\nProto_v:" + str(self.dataset.proto) + \
                    "\nEp:" + str(best_epoch) + \
                    "\nMRR:" + str(best_mrr) + \
                    "\nH@1:" + str(best_h1) + "\nH@3:" + str(best_h3) + "\nH@10:" + str(best_h10) + \
                    "\nS@1:" + str(best_sem1) + "\nS@3:" + str(best_sem3) + "\nS@10:" + str(best_sem10))
            f.close()

    def init_inst_with_proto(self):
        if self.model_name_inst in ['TransE', 'DistMult', 'TuckER', 'ConvE']:
            weights_e = []
        if self.model_name_inst == "ComplEx":
            weights_e_re = []
            weights_e_im = []
        for e in range(self.dataset.num_ent()):
            if self.aggreg_option == 'most_specific':
                classes = self.dataset.instype_spec[e]
            else:  
                classes = self.dataset.instype_appearing[e]
            if classes != []:
                # STR 1 : picking a class randomly
                if self.aggreg_operator == "random":
                    if self.model_name_inst in ['TransE', 'DistMult', 'TuckER', 'ConvE']:
                        ent_embs = np.array(self.model_proto.ent_embs.weight.data[random.choice(classes)].cpu())
                    if self.model_name_inst == "ComplEx":
                        rnd = random.choice(classes)
                        ent_embs_re = np.array(self.model_proto.ent_re_embeddings.weight.data[rnd].cpu())
                        ent_embs_im = np.array(self.model_proto.ent_im_embeddings.weight.data[rnd].cpu())
                # STR 2 : take the mean over all the classes
                if self.aggreg_operator == "mean":
                    if self.model_name_inst in ['TransE', 'DistMult', 'TuckER', 'ConvE']:
                        ent_embs = np.mean(np.array(self.model_proto.ent_embs.weight.data[classes,:].cpu()),axis=0)
                    if self.model_name_inst == "ComplEx":
                        ent_embs_re = np.mean(np.array(self.model_proto.ent_re_embeddings.weight.data[classes,:].cpu()),axis=0)
                        ent_embs_im = np.mean(np.array(self.model_proto.ent_im_embeddings.weight.data[classes,:].cpu()),axis=0)
                # STR 3 : sum of all the class embeddings
                if self.aggreg_operator == "sum":
                    if self.model_name_inst in ['TransE', 'DistMult', 'TuckER', 'ConvE']:
                        ent_embs = np.sum(np.array(self.model_proto.ent_embs.weight.data[classes,:].cpu()),axis=0)
                    if self.model_name_inst == "ComplEx":
                        ent_embs_re = np.sum(np.array(self.model_proto.ent_re_embeddings.weight.data[classes,:].cpu()),axis=0)
                        ent_embs_im = np.sum(np.array(self.model_proto.ent_im_embeddings.weight.data[classes,:].cpu()),axis=0)
                # STR 4 : take the most frequently appearing class for that entity
                if self.aggreg_operator == "frequency":
                    if self.dataset.frequency[e] != {}:
                        most_frequent_class = max(self.dataset.frequency[e], key=self.dataset.frequency[e].get)
                        most_frequent_class = self.class_mapping[most_frequent_class]
                        if self.model_name_inst in ['TransE', 'DistMult', 'TuckER', 'ConvE']:
                            ent_embs = np.array(self.model_proto.ent_embs.weight.data[most_frequent_class].cpu())
                        if self.model_name_inst == "ComplEx":
                            ent_embs_re = np.array(self.model_proto.ent_re_embeddings.weight.data[most_frequent_class].cpu())
                            ent_embs_im = np.array(self.model_proto.ent_im_embeddings.weight.data[most_frequent_class].cpu())
                    else:
                        if self.model_name_inst in ['TransE', 'DistMult', 'TuckER', 'ConvE']:
                            ent_embs = np.array(self.model_inst.ent_embs.weight.data[e].cpu())
                        if self.model_name_inst == "ComplEx":
                            ent_embs_re = np.array(self.model_inst.ent_re_embeddings.weight.data[e].cpu())
                            ent_embs_im = np.array(self.model_inst.ent_im_embeddings.weight.data[e].cpu())
            else :
                if self.model_name_inst in ['TransE', 'DistMult', 'TuckER', 'ConvE']:
                    ent_embs = np.array(self.model_inst.ent_embs.weight.data[e].cpu())
                if self.model_name_inst == "ComplEx":
                    ent_embs_re = np.array(self.model_inst.ent_re_embeddings.weight.data[e].cpu())
                    ent_embs_im = np.array(self.model_inst.ent_im_embeddings.weight.data[e].cpu())
            if self.model_name_inst in ['TransE', 'DistMult', 'TuckER', 'ConvE']:
                weights_e.append(ent_embs)
            if self.model_name_inst == "ComplEx":
                weights_e_re.append(ent_embs_re)
                weights_e_im.append(ent_embs_im)

        if self.model_name_inst in ['TransE', 'DistMult', 'TuckER', 'ConvE']:
            weights_e = torch.tensor(weights_e).to(self.device)
            weights_r = (self.model_proto.rel_embs.weight.data).to(self.device)
            self.model_inst.ent_embs.weight.data = weights_e
            self.model_inst.rel_embs.weight.data = weights_r
        if self.model_name_inst == "ComplEx":
            weights_e_re = torch.tensor(weights_e_re).to(self.device)
            weights_e_im = torch.tensor(weights_e_im).to(self.device)
            weights_r_re = (self.model_proto.rel_re_embeddings.weight.data).to(self.device)
            weights_r_im = (self.model_proto.rel_im_embeddings.weight.data).to(self.device)
            self.model_inst.ent_re_embeddings.weight.data = weights_e_re
            self.model_inst.ent_im_embeddings.weight.data = weights_e_im
            self.model_inst.rel_re_embeddings.weight.data = weights_r_re
            self.model_inst.rel_im_embeddings.weight.data = weights_r_im

    def save_model_inst(self, model, chkpnt):
        directory = "models/" + self.dataset.name + "/" + self.model_name_inst + "/instance/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.model_name_inst == 'ConvE':
            model_path = directory + "proto"+str(self.dataset.proto)+"_instance__dim=" + str(self.args.dim) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_drop) + \
                "_hid_dropout=" + str(self.args.hidden_drop) + "_feat_drop=" + str(self.args.feat_drop) + "_bs=" + str(self.args.batch_size) + "__epoch=" + str(chkpnt) + ".pt"
        elif self.model_name_inst == 'TuckER':
            model_path = directory + "proto"+str(self.dataset.proto)+"_instance__dim=" + str(self.args.dim_e) + "_lr=" + str(self.args.lr) + "_input_drop=" + str(self.args.input_dropout) + \
                "_hid_dropout1=" + str(self.args.hidden_dropout1) + "_hid_dropout2=" + str(self.args.hidden_dropout2) + "_bs=" + str(self.args.batch_size) + "__epoch=" + str(chkpnt) + ".pt"
        else:
            model_path = directory + \
                       "proto"+str(self.dataset.proto)+"_instance__dim=" + \
                       str(self.args.dim) + \
                       "_lr=" + \
                       str(self.args.lr) + \
                       "_reg=" + \
                       str(self.args.reg) + \
                       "_bs=" + \
                       str(self.args.batch_size) + \
                       "__epoch=" + \
                       str(chkpnt) + \
                       ".pt"
        torch.save(self.model_inst.state_dict(), model_path)
        return model_path