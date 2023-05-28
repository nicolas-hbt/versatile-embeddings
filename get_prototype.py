import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import argparse

def read_pickle(file):
    try:
        with open(path + "pickle/" + file + ".pkl", 'rb') as f:
            pckl = pickle.load(f)
            return pckl
    except:
        print(file + ".pkl not found.")

def build_prototype_1():
    # information strictly contained in relations' domains & ranges
    with open(path + "prototype_v1.txt", 'w') as f:        
        for r in tqdm(rel2id):
            id_rel = rel2id[r]
            if id_rel in r2id2dom2id and id_rel in r2id2range2id:
                f.write('{}\t{}\t{}\n'.format(r2id2dom2id[id_rel], id_rel, r2id2range2id[id_rel]))

def build_prototype_2():
    with open(path + "prototype_v2.txt", 'w') as f:        
        for r in tqdm(rel2id):
            id_rel = rel2id[r]
            if id_rel in r2id2dom2id and id_rel in r2id2range2id:
                f.write('{}\t{}\t{}\n'.format(r2id2dom2id[id_rel], id_rel, r2id2range2id[id_rel]))                           
                if r2id2dom2id[id_rel] in class2subclasses2id:
                    dom_sbs = class2subclasses2id[r2id2dom2id[id_rel]]
                    for sb_dom in dom_sbs:
                        f.write('{}\t{}\t{}\n'.format(sb_dom, id_rel, r2id2range2id[id_rel]))
                if r2id2range2id[id_rel] in class2subclasses2id:
                     range_sbs = class2subclasses2id[r2id2range2id[id_rel]]
                     for sb_range in range_sbs:
                        f.write('{}\t{}\t{}\n'.format(r2id2dom2id[id_rel], id_rel, sb_range))
    proto_v2 = pd.read_csv(path + "prototype_v2.txt", sep='\t', header=None, names=['h','r','t'])
    proto_v2.drop_duplicates(inplace=True)
    np.savetxt(path + "prototype_v2.txt", proto_v2, fmt='%i', delimiter='\t')

def build_mapping_classes(proto_version):
    proto_kg = pd.read_csv(path + "prototype_v"+str(proto_version)+".txt", sep='\t', header=None, names=['h','r','t']).to_numpy()
    class_mapping = {}
    label2class = {}
    i=0
    unique_classes = set((list(np.unique(proto_kg[:,0])) + list(np.unique(proto_kg[:,2]))))
    with open(path + "mapping_v"+str(proto_version)+".txt", 'w') as f:
        for c in unique_classes:
            class_mapping[c] = i
            label2class[id2class[c]] = i
            f.write('{}\t{}\n'.format(c, i))
            i+=1
    with open(path + "pickle/mapping_v"+str(proto_version)+".pkl", 'wb') as f:
        pickle.dump(class_mapping, f)
    return class_mapping

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='YAGO14K', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parameter()
    dataset = args.dataset
    path = "datasets/" + dataset + "/"
    ent2id, rel2id = read_pickle('ent2id'), read_pickle('rel2id')
    class2id, class2id2ent2id, subclassof2id = read_pickle('class2id'), read_pickle('class2id2ent2id'), read_pickle('subclassof2id')
    r2id2dom2id, r2id2range2id = read_pickle('r2id2dom2id'), read_pickle('r2id2range2id')

    id2ent = {v:k for k,v in ent2id.items()}
    id2rel = {v:k for k,v in rel2id.items()}
    id2class = {v:k for k,v in class2id.items()}

    class2subclasses2id = {}
    for key, v in subclassof2id.items():
        if isinstance(v, list):
            for v2 in v:
                class2subclasses2id.setdefault(v2, []).append(key)
        else:
            class2subclasses2id.setdefault(v, []).append(key)


    build_prototype_1()
    build_prototype_2()

    build_mapping_classes(1)
    build_mapping_classes(2)

