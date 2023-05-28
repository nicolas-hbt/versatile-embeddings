import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_, xavier_uniform_
torch.manual_seed(7)
np.random.seed(7)


class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(TransE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.distance = 'l2'
        self.p_norm = 1
        torch.manual_seed(7)
        self.ent_embs = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs = nn.Embedding(num_rel, emb_dim).to(device)
        nn.init.xavier_uniform_(self.ent_embs.weight.data, gain=1)
        nn.init.xavier_uniform_(self.rel_embs.weight.data, gain=1)

    def forward2(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        score = self._calc(e_hs, e_rs, e_ts).view(-1, 1).to(self.device)
        return score

    def forward(self, batch):
        hs = torch.tensor(batch[:, 0]).long().to(self.device)
        rs = torch.tensor(batch[:, 1]).long().to(self.device)
        ts = torch.tensor(batch[:, 2]).long().to(self.device)
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        score = self._calc(e_hs, e_rs, e_ts).view(-1, 1).to(self.device)
        return score

    def _calc(self, e_hs, e_rs, e_ts):
        score = (e_hs + e_rs) - e_ts
        if self.distance == 'l1':
            score = torch.norm(score, self.p_norm, -1)
        else:
            score = torch.sqrt(torch.sum((score)**2, 1))
        return -score

    def _pairwise_hinge(self, y_pos, y_neg, neg_ratio, gamma):
        criterion = nn.MarginRankingLoss(margin=gamma)
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(
            np.ones(P * (int(y_neg.shape[0] / P)), dtype=np.int32))).to(self.device)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) +
                 torch.mean(e_rs ** 2)) / 3
        return regul
        

class DistMult(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=None):
        super(DistMult, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        self.gamma = margin
        torch.manual_seed(7)
        self.ent_embs = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs = nn.Embedding(num_rel, emb_dim).to(device)
        nn.init.xavier_uniform_(self.ent_embs.weight.data, gain=1)
        nn.init.xavier_uniform_(self.rel_embs.weight.data, gain=1)

    def forward2(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        score = self._calc(e_hs, e_rs, e_ts)
        return score

    def forward(self, batch):
        hs = torch.tensor(batch[:, 0]).long().to(self.device)
        rs = torch.tensor(batch[:, 1]).long().to(self.device)
        ts = torch.tensor(batch[:, 2]).long().to(self.device)
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        score = self._calc(e_hs, e_rs, e_ts)
        return score

    def _calc(self, e_hs, e_rs, e_ts):
        return torch.sum(e_hs * e_rs * e_ts, -1)

    def _pairwise_hinge(self, y_pos, y_neg, neg_ratio, gamma):
        criterion = nn.MarginRankingLoss(margin=gamma)
        P = y_pos.size(0)
        y_pos = y_pos.view(-1).repeat(int(y_neg.shape[0] / P)).to(self.device)
        y_neg = y_neg.view(-1).to(self.device)
        target = Variable(torch.from_numpy(
            np.ones(P * (int(y_neg.shape[0] / P)), dtype=np.int32))).to(self.device)
        loss = criterion(y_pos, y_neg, target)
        return loss

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(
            self.device), self.rel_embs(rs).to(
            self.device), self.ent_embs(ts).to(
            self.device)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) +
                 torch.mean(e_rs ** 2)) / 3
        return regul


class ComplEx(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device, margin=None):
        super(ComplEx, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device
        torch.manual_seed(7)
        self.criterion = nn.Softplus()
        self.ent_re_embeddings = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_re_embeddings = nn.Embedding(num_rel, emb_dim).to(device)
        self.ent_im_embeddings = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_im_embeddings = nn.Embedding(num_rel, emb_dim).to(device)
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def _calc(self, e_re_h, e_im_h, r_re, r_im, e_re_t, e_im_t):
        return torch.sum( r_re * e_re_h * e_re_t + r_re * e_im_h * e_im_t + r_im * e_re_h * e_im_t - r_im * e_im_h * e_re_t, 1, False)

    def _softplus(self, pos_scores, neg_scores, neg_ratio):
        P, N = pos_scores.size(0), neg_scores.size(0)
        pos_scores, neg_scores = pos_scores.view(-1).to(self.device), neg_scores.view(-1).to(self.device)
        true_y, corrup_y = torch.ones(P).to(self.device), -torch.ones(N).to(self.device)
        target = torch.cat((true_y, corrup_y), 0)
        y = torch.cat((pos_scores, neg_scores), 0).to(self.device)
        return torch.mean(self.criterion(-target * y))

    def forward2(self, hs, rs, ts):
        e_re_h, e_im_h = self.ent_re_embeddings(hs).to(self.device), self.ent_im_embeddings(hs).to(self.device)
        e_re_t, e_im_t = self.ent_re_embeddings(ts).to(self.device), self.ent_im_embeddings(ts).to(self.device)
        r_re, r_im = self.rel_re_embeddings(rs).to(self.device), self.rel_im_embeddings(rs).to(self.device)
        score = self._calc(e_re_h, e_im_h, r_re, r_im, e_re_t, e_im_t)
        return score

    def forward(self, batch):
        hs = torch.tensor(batch[:, 0]).long().to(self.device)
        rs = torch.tensor(batch[:, 1]).long().to(self.device)
        ts = torch.tensor(batch[:, 2]).long().to(self.device)
        e_re_h, e_im_h = self.ent_re_embeddings(hs).to(self.device), self.ent_im_embeddings(hs).to(self.device)
        e_re_t, e_im_t = self.ent_re_embeddings(ts).to(self.device), self.ent_im_embeddings(ts).to(self.device)
        r_re, r_im = self.rel_re_embeddings(rs).to(self.device), self.rel_im_embeddings(rs).to(self.device)
        score = self._calc( e_re_h, e_im_h, r_re, r_im, e_re_t, e_im_t).to(self.device)
        return score

    def _regularization(self, hs, rs, ts):
        e_re_h, e_im_h = self.ent_re_embeddings(hs).to(self.device), self.ent_im_embeddings(hs).to(self.device)
        e_re_t, e_im_t = self.ent_re_embeddings(ts).to(self.device), self.ent_im_embeddings(ts).to(self.device)
        r_re, r_im = self.rel_re_embeddings(rs).to(self.device), self.rel_im_embeddings(rs).to(self.device)
        regul = (torch.mean(e_re_h ** 2) + torch.mean(e_im_h ** 2) + torch.mean(e_re_t ** 2) + torch.mean(e_im_t ** 2) + torch.mean(r_re ** 2) + torch.mean(r_im ** 2)) / 6
        return regul


class TuckER(nn.Module):
    def __init__(self, num_ent, num_rel, args, device):
        super(TuckER, self).__init__()
        self.args = args
        self.num_ent = num_ent
        self.num_rel = num_rel * 2
        self.dim_e = args.dim_e
        self.dim_r = args.dim_r
        self.device = device
        torch.manual_seed(7)
        self.ent_embs = torch.nn.Embedding(self.num_ent, self.dim_e).to(self.device)
        self.rel_embs = torch.nn.Embedding(self.num_rel, self.dim_r).to(self.device)
        xavier_normal_(self.ent_embs.weight.data)
        xavier_normal_(self.rel_embs.weight.data)
        self.W = torch.nn.Parameter(torch.tensor(
            np.random.uniform(-1, 1, (self.dim_r, self.dim_e, self.dim_e)),
            dtype=torch.float,requires_grad=True
            ).to(self.device))

        self.input_dropout = torch.nn.Dropout(self.args.input_dropout).to(self.device)
        self.hidden_dropout1 = torch.nn.Dropout(self.args.hidden_dropout1).to(self.device)
        self.hidden_dropout2 = torch.nn.Dropout(self.args.hidden_dropout2).to(self.device)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(self.dim_e).to(self.device)
        self.bn1 = torch.nn.BatchNorm1d(self.dim_e).to(self.device)

    def _calc(self, e1, r):
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.ent_embs.weight.transpose(1, 0))
        return x

    def forward(self, batch):
        hs = torch.tensor(batch[:, 0]).long().to(self.device)
        rs = torch.tensor(batch[:, 1]).long().to(self.device)
        e1 = self.ent_embs(hs).to(self.device)
        r = self.rel_embs(rs).to(self.device)
        pred = torch.sigmoid(self._calc(e1, r))
        return pred

    def _bce(self, pos_batch):
        preds = self.forward(pos_batch).to(self.device)
        ts = torch.tensor(pos_batch[:, 2]).long().to(self.device)
        labels = self.get_batch(pos_batch.shape[0], ts).to(self.device)
        return self.loss(preds, labels)

    def get_batch(self, batch_size, batch_t):
        targets = torch.zeros(batch_size, self.num_ent).scatter_(1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(self.device)
        return targets

    def get_score(self, h, r, t):
        e1 = self.ent_embs(h).unsqueeze(0).to(self.device)
        r = self.rel_embs(r).unsqueeze(0).to(self.device)
        pred = self._calc(e1, r)
        return pred

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) + torch.mean(e_rs ** 2)) / 3
        return regul


class ConvE(nn.Module):
    def __init__(self, num_ent, num_rel, args, device):
        super(ConvE, self).__init__()
        self.args = args
        self.device = device
        self.num_ent = num_ent
        self.num_rel = num_rel * 2
        torch.manual_seed(7)
        self.ent_embs = nn.Embedding(self.num_ent, self.args.dim, padding_idx=0).to(self.device)
        self.rel_embs = nn.Embedding(self.num_rel, self.args.dim, padding_idx=0).to(self.device)
        self.inp_drop = nn.Dropout(self.args.input_drop).to(self.device)
        self.hidden_drop = nn.Dropout(self.args.hidden_drop).to(self.device)
        self.feature_map_drop = nn.Dropout2d(self.args.feat_drop).to(self.device)
        self.loss = nn.BCELoss()
        self.emb_dim1 = self.args.embedding_shape1
        self.emb_dim2 = self.args.dim // self.emb_dim1
        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), (1, 1), 0, bias=True).to(self.device)
        self.bn0 = nn.BatchNorm2d(1).to(self.device)
        self.bn1 = nn.BatchNorm2d(32).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.args.dim).to(self.device)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.num_ent).to(self.device)))
        self.fc = nn.Linear(self.args.hidden_size, self.args.dim).to(self.device)

    def _calc(self, e1_embedded, rel_embedded):
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embs.weight.transpose(1, 0))
        x += self.b.expand_as(x).to(self.device)
        return x

    def forward(self, batch):
        hs = torch.tensor(batch[:, 0]).long().to(self.device)
        rs = torch.tensor(batch[:, 1]).long().to(self.device)
        e1_embedded = self.ent_embs(
            hs).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        rel_embedded = self.rel_embs(
            rs).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        pred1 = self._calc(e1_embedded, rel_embedded)
        return torch.sigmoid(pred1)

    def _bce(self, pos_batch):
        preds = self.forward(pos_batch).to(self.device)
        ts = torch.tensor(pos_batch[:, 2]).long().to(self.device)
        labels = self.get_batch(pos_batch.shape[0], ts).to(self.device)
        return self.loss(preds, labels)

    def get_batch(self, batch_size, batch_t):
        targets = torch.zeros(batch_size, self.num_ent).scatter_(
            1, batch_t.cpu().view(-1, 1).type(torch.int64), 1).to(self.device)
        return targets

    def get_score(self, h, r, t):
        e1_embedded = self.ent_embs(h).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        rel_embedded = self.rel_embs(r).view(-1, 1, self.emb_dim1, self.emb_dim2).to(self.device)
        pred = self._calc(e1_embedded, rel_embedded)
        return pred

    def _regularization(self, hs, rs, ts):
        e_hs, e_rs, e_ts = self.ent_embs(hs).to(self.device), self.rel_embs(rs).to(self.device), self.ent_embs(ts).to(self.device)
        regul = (torch.mean(e_hs ** 2) + torch.mean(e_ts ** 2) + torch.mean(e_rs ** 2)) / 3
        return regul
