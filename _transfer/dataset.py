import numpy as np
import pandas as pd
import pickle
import random
import torch
import copy
from queue import Queue

class Dataset:
    def __init__(self, ds_name, args):
        self.args = args
        self.model_name = args.model
        self.proto = self.args.proto_version
        self.name = ds_name
        self.weights = args.weights
        self.discard = args.discard
        self.batch_size = args.batch_size
        self.dir = "datasets/" + self.name + "/"
        self.ent2id, self.ent2id_typed, self.entid2typid = self.read_pickle('ent2id'), self.read_pickle('ent2id_typed'), self.read_pickle('entid2typid')
        self.instype_spec, self.instype_transitive, self.instype_all = self.read_pickle('instype_spec'), self.read_pickle('instype_transitive'), self.read_pickle('instype_all')
        self.rel2id = self.read_pickle('rel2id')
        self.class2id, self.class2id2ent2id, self.subclassof2id = self.read_pickle('class2id'), self.read_pickle('class2id2ent2id'), self.read_pickle('subclassof2id')
        self.r2id2dom2id, self.r2id2range2id = self.read_pickle('r2id2dom2id'), self.read_pickle('r2id2range2id')
        self.setting = args.setting
        self.sem = args.sem
        self.need_inverse = self.model_name in ['ConvE', 'TuckER']
        print(self.need_inverse)

        if self.model_name in ['ConvE', 'TuckER']:
            inv_r2id2dom2id = {}
            for k,v in self.r2id2dom2id.items():
                try:
                    inv_r2id2dom2id[k + max(self.rel2id.values()) + 1] = self.r2id2range2id[k]
                except:
                    pass
                
            inv_r2id2range2id = {}
            for k,v in self.r2id2range2id.items():
                try:
                    inv_r2id2range2id[k + max(self.rel2id.values()) + 1] = self.r2id2dom2id[k]
                except:
                    pass
            self.r2id2dom2id.update(inv_r2id2dom2id)
            self.r2id2range2id.update(inv_r2id2range2id)
        
        self.r2hs = self.read_pickle('heads2id_original')
        if self.model_name in ['ConvE', 'TuckER']:
            self.r2ts_origin = self.read_pickle('tails2id_original')
            self.r2ts = self.r2ts_origin.copy()
            for rel, tails in self.r2ts_origin.items():
                self.r2ts[rel+len(self.rel2id)] = self.r2hs[rel]
        else:
            self.r2ts = self.read_pickle('tails2id_original')
        
        self.data = {}
        self.data["pd_train"] = pd.read_csv(self.dir + "train2id.txt", sep='\t', header=None, names=['h','r','t'])
        self.data["pd_valid"] = pd.read_csv(self.dir + "valid2id.txt", sep='\t', header=None, names=['h','r','t'])
        self.data["pd_test"] = pd.read_csv(self.dir + "test2id.txt", sep='\t', header=None, names=['h','r','t'])
        self.data["train"] = self.data["pd_train"].to_numpy()
        self.data["valid"] = self.data["pd_valid"].to_numpy()
        self.data["test"] = self.data["pd_test"].to_numpy()
        self.data["df"] = pd.concat([self.data["pd_train"],self.data["pd_valid"],self.data["pd_test"]]).to_numpy()
        self.data["df_lst"] = self.data["df"].tolist()
        self.data["dft"] = tuple(map(tuple, self.data["df"]))
        if self.model_name in ['ConvE', 'TuckER']:
            self.data["pd_train_inv"] = pd.read_csv(self.dir + "train2id_inv.txt", sep='\t', header=None, names=['h','r','t'])
            self.data["pd_valid_inv"] = pd.read_csv(self.dir + "valid2id_inv.txt", sep='\t', header=None, names=['h','r','t'])
            self.data["pd_test_inv"] = pd.read_csv(self.dir + "test2id_inv.txt", sep='\t', header=None, names=['h','r','t'])
            self.data["train_inv"] = self.data["pd_train_inv"].to_numpy()
            self.data["valid_inv"] = self.data["pd_valid_inv"].to_numpy()
            self.data["test_inv"] = self.data["pd_test_inv"].to_numpy()

        self.neg_ratio = args.neg_ratio
        self.all_rels = list(self.rel2id.values())
        self.all_ents = list(self.ent2id.values())
        self.batch_index_inst = 0
        self.batch_index_proto = 0
        self.batch_index_proto_inv = 0
        self.batch_index_inst_inv = 0
        self.batch_index_inst_valid = 0

        self.id2ent = {v:k for k,v in self.ent2id.items()}
        self.id2rel = {v:k for k,v in self.rel2id.items()}
        if self.name not in ['Codex-S', 'Codex-M', 'WN18RR']:
            self.id2class = {v:k for k,v in self.class2id.items()}

        self.dist_classes2id = self.get_dist_classes()

        self.data["prototype"] = self.get_prototype_graph()
        if self.proto == 3:
            self.ent2proto = self.read_pickle('ent2proto')
            self.proto2ents = self.read_pickle('proto2ents')
        else:
            self.class_mapping = self.build_mapping_classes()
            self.instype_appearing = self.get_appearing_classes()
            self.data["prototype"] = self.map_prototype_classes()
            self.instype_spec = {}
            for k,values in self.instype_all.items():
                self.instype_spec[k] = self.get_most_specific(k)
            self.instype_spec = self.map_instype_spec()
            self.spec_classes2ents = {}
            for key, values in self.instype_spec.items():
                for v in values:
                    self.spec_classes2ents.setdefault(v, []).append(key)
        self.proto_classes = list(set(list(self.data["prototype"][:,0]) + list(self.data["prototype"][:,2])))

        if self.model_name in ['ConvE', 'TuckER']:
            self.data["prototype_inv"] = self.inv_proto_graph()

    def read_pickle(self, file):
        try:
            with open(self.dir + "pickle/" + file + ".pkl", 'rb') as f:
                pckl = pickle.load(f)
                return pckl
        except:
            print(file + ".pkl not found.")

    def inv_proto_graph(self):
        proto_kg = pd.DataFrame(self.data["prototype"], columns = ['h','r','t'])
        inv_proto_kg = proto_kg.copy(deep=True)
        inv_proto_kg['h'], inv_proto_kg['r'], inv_proto_kg['t'] = inv_proto_kg['t'], inv_proto_kg['r'] + self.num_rel(), inv_proto_kg['h']
        return pd.concat([proto_kg, inv_proto_kg]).to_numpy()

    def map_instype_spec(self):
        instype_v2 = copy.deepcopy(self.instype_spec)
        for k,values in self.instype_spec.items():
            instype_v2[k] = []
            for v in values:
                if v in self.class_mapping:
                    instype_v2[k].append(self.class_mapping[v])
        return instype_v2

    def get_dist_classes(self):
        self.subclassof = {}
        self.dist_classes = {}
        for k,v in self.subclassof2id.items():
            if isinstance(v, list):
                self.subclassof[self.id2class[k]] = list(map(lambda x: self.id2class[x], v))
            else:
                self.subclassof[self.id2class[k]] = self.id2class[v]
        for cl in self.class2id.keys():
            self.dist_classes[cl] = self.leastDistance(self.subclassof, cl)
    
        self.dist_classes2id = {} 
        for k, values in self.dist_classes.items():
            self.dist_classes2id[self.class2id[k]] = {}
            for cl,dist in values.items():
                self.dist_classes2id[self.class2id[k]][self.class2id[cl]] = dist
        return self.dist_classes2id

    def get_appearing_classes(self):
        instype_v2 = copy.deepcopy(self.instype_all)
        for k,values in self.instype_all.items():
            instype_v2[k] = []
            for v in values:
                if v in self.class_mapping:
                    instype_v2[k].append(self.class_mapping[v])
        return instype_v2

    def build_mapping_classes(self):
        class_mapping = {}
        label2class = {}
        i=0
        unique_classes = set((list(np.unique(self.data["prototype"][:,0])) + list(np.unique(self.data["prototype"][:,2]))))
        with open(self.dir + "mapping_v"+str(self.proto)+".txt", 'w') as f:
            for c in unique_classes:
                class_mapping[c] = i
                label2class[self.id2class[c]] = i
                f.write('{}\t{}\n'.format(c, i))
                i+=1
        with open(self.dir + "pickle/mapping_v"+str(self.proto)+".pkl", 'wb') as f:
            pickle.dump(class_mapping, f)
        return class_mapping

    def map_prototype_classes(self):
        if self.weights == 0:
            df = pd.DataFrame(self.data["prototype"], columns = ["h","r","t"])
        else:
            df = pd.DataFrame(self.data["prototype"], columns = ['h','r','t','count','prop','rel_prop'])
        df.iloc[:,0] = df.iloc[:,0].map(self.class_mapping)
        df.iloc[:,0] = df.iloc[:,0].astype(int)
        df.iloc[:,2] = df.iloc[:,2].map(self.class_mapping)
        df.iloc[:,2] = df.iloc[:,2].astype(int)
        return df.to_numpy()
      
    def num_ent(self):
        return len(self.ent2id)
    
    def num_rel(self):
        return len(self.rel2id)

    def get_prototype_graph(self):
        proto_kg = self.build_prototype_graph()
        return proto_kg

    def build_prototype_graph(self):
        if self.proto == 1:
            try:
                proto_kg = pd.read_csv(self.dir + "prototype_v1.txt", sep='\t', header=None, names=['h','r','t'])
            except:
                print("Build prototype KG beforehand.")

        elif self.proto == 2:
            try:
                proto_kg = pd.read_csv(self.dir + "prototype_v2.txt", sep='\t', header=None, names=['h','r','t'])
            except:
                print("Build prototype KG beforehand.")

        proto_kg = proto_kg.to_numpy()
        return proto_kg

    def get_most_specific(self, e):
        all_classes = self.instype_all[e]
        all_classes2 = copy.deepcopy(all_classes)
        for c in all_classes:
            superclasses = list(self.dist_classes2id[c].keys())[1:]
            all_classes2 = list(set(all_classes2) - set(superclasses))
        return all_classes2


    # PROTOGRAPH-BASED SAMPLING

    def next_pos_batch_prototype(self, batch_size):
        if self.batch_index_proto + batch_size < len(self.data["prototype"]): 
            batch = self.data["prototype"][self.batch_index_proto: self.batch_index_proto+batch_size]
            self.batch_index_proto += batch_size
        else:
            batch = self.data["prototype"][self.batch_index_proto:]
            self.batch_index_proto = 0
        return batch  
    
    def next_pos_batch_prototype_inv(self, batch_size):
        if self.batch_index_proto_inv + batch_size < len(self.data["prototype_inv"]): 
            batch = self.data["prototype_inv"][self.batch_index_proto_inv: self.batch_index_proto_inv+batch_size]
            self.batch_index_proto_inv += batch_size
        else:
            batch = self.data["prototype_inv"][self.batch_index_proto_inv:]
            self.batch_index_proto_inv = 0
        return batch

    def random_negative_sampling_prototype(self, pos_batch, neg_ratio):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        for i in range(len(neg_batch)):
            if random.random() < 0.5:
                neg_batch[i][0] = self.rand_ent_except_prototype(neg_batch[i][0])
            else:
                neg_batch[i][2] = self.rand_ent_except_prototype(neg_batch[i][2])
        return neg_batch

    def next_batch_prototype(self, batch_size, neg_ratio, device):
        pos_batch = self.next_pos_batch_prototype(batch_size) 
        neg_batch = self.random_negative_sampling_prototype(pos_batch, neg_ratio)
        batch = np.append(pos_batch, neg_batch, axis=0)
        batch = torch.tensor(batch)
        return batch

    def rand_ent_except_prototype(self, ent):
        rand_ent = random.randint(0, len(self.proto_classes) - 1)
        while(rand_ent == ent):
            rand_ent = random.randint(0, len(self.proto_classes) - 1)
        return rand_ent


    # INSTANCE-BASED SAMPLING

    def next_pos_batch(self, batch_size):
        if self.batch_index_inst + batch_size < len(self.data["train"]): 
            batch = self.data["train"][self.batch_index_inst: self.batch_index_inst+batch_size]
            self.batch_index_inst += batch_size
        else:
            batch = self.data["train"][self.batch_index_inst:]
            self.batch_index_inst = 0
        return batch
    
    def next_pos_batch_inv(self, batch_size):
        if self.batch_index_inst_inv + batch_size < len(self.data["train_inv"]): 
            batch = self.data["train_inv"][self.batch_index_inst_inv: self.batch_index_inst_inv+batch_size]
            self.batch_index_inst_inv += batch_size
        else:
            batch = self.data["train_inv"][self.batch_index_inst_inv:]
            self.batch_index_inst_inv = 0
        return batch

    def random_negative_sampling(self, pos_batch, neg_ratio, side='all'):
        if self.neg_ratio==1:
            neg_ratio=2
        neg_batch = np.array([[0,0,0,0,0]])
        for i in range(self.neg_ratio):
            tmp_neg_batch = np.repeat(np.copy(pos_batch), 1, axis=0)
            M = tmp_neg_batch.shape[0]
            corr = np.random.randint(self.num_ent() - 1, size=M)
            e_idxs = np.random.choice([0, 2], size=M)
            tmp_neg_batch[np.arange(M), e_idxs] = corr
            tmp_neg_batch = np.column_stack((tmp_neg_batch, [i%len(pos_batch) for i in range(len(tmp_neg_batch))]))
            tmp_neg_batch = np.column_stack((tmp_neg_batch, e_idxs))
            neg_batch = np.concatenate((neg_batch,tmp_neg_batch),axis=0)
        neg_batch = neg_batch[1:]
        neg_batch = self.filtering(pos_batch, neg_batch)
        labels = np.array([int((self.r2id2dom2id[el[1]] in self.instype_all[el[0]]) and (self.r2id2range2id[el[1]] in self.instype_all[el[2]])) for el in neg_batch])
        neg_batch = np.column_stack((neg_batch, labels))
        neg_batch = neg_batch[neg_batch[:, 3].argsort()]
        e_idxs = neg_batch[:,4]
        return neg_batch, e_idxs

    def next_batch(self, batch_size, neg_ratio, device, cache):
        pos_batch = self.next_pos_batch(batch_size)
        neg_batch, corr_idxs = self.random_negative_sampling(pos_batch, neg_ratio)
        if cache != None:
            pos_batch = np.vstack([pos_batch, np.array(cache['pos_triples'].cpu())])
            neg_batch = np.vstack([neg_batch, np.array(cache['neg_triples'].cpu())])
            corr_idxs = np.concatenate((corr_idxs, np.array(cache['corr_idxs'].cpu())))
        return pos_batch, neg_batch, corr_idxs

    def was_last_batch_inst(self):
        return (self.batch_index_inst == 0)
    
    def was_last_batch_inst_inv(self):
        return (self.batch_index_inst_inv == 0)

    def was_last_batch_proto(self):
        return (self.batch_index_proto == 0)
    
    def was_last_batch_proto_inv(self):
        return (self.batch_index_proto_inv == 0)

    def filtering(self, pos_batch, neg_batch):
        nbt = tuple(map(tuple, neg_batch))
        both = ((set(nbt).intersection((self.data['dft']))))
        dupl = [nbt.index(x) for x in both]
        nbt_rem = tuple(map(tuple, neg_batch[dupl]))
        nbt_ok = tuple(set(nbt) - set(nbt_rem))
        neg_batch = np.asarray(nbt_ok[:(len(nbt_ok)-(len(nbt_ok)%pos_batch.shape[0]))])
        if self.neg_ratio == 1:
            neg_batch = neg_batch[:pos_batch.shape[0]]
        return neg_batch

    def leastDistance(self, graph, source):
        Q = Queue()
        distance = {k: 100 for k in self.class2id.keys()}
        visited_vertices = set()
        Q.put(source)
        visited_vertices.update({source})
        while not Q.empty():
            vertex = Q.get()
            if vertex == source:
                distance[vertex] = 0
            if vertex != 'owl:Thing':
                if self.name in ['FB15K237-ET', 'DB93K', 'JF17K']:
                    if vertex in graph:
                        if graph[vertex] not in visited_vertices:
                            if distance[graph[vertex]] > distance[vertex] + 1:
                                distance[graph[vertex]] = distance[vertex] + 1
                            Q.put(graph[vertex])
                            visited_vertices.update({graph[vertex]})
                elif self.name in ['YAGO4-19K']:
                    for u in graph[vertex]:
                        if u not in visited_vertices:
                            # update the distance
                            if distance[u] > distance[vertex] + 1:
                                distance[u] = distance[vertex] + 1
                            Q.put(u)
                            visited_vertices.update({u})
        sorted_distances = sorted(distance.items(), key=lambda x:x[1])
        converted_dict = dict(sorted_distances)
        distances = {k:v for k,v in converted_dict.items() if v != 100}
        return distances