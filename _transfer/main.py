from joint_trainer import JointTrainer
from tester import Tester
from dataset import Dataset
from models import *
import numpy as np
import pandas as pd
import argparse
import time
import os
import torch
import pickle
import json
from datetime import datetime

date_today = datetime.today().strftime('%d-%m-%Y')

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-use_proto', default=1, type=int, help="whether to use the proto KG")
    parser.add_argument('-ne', default=400, type=int, help="")
    parser.add_argument('-validate_every', default=20, type=int, help="validate every k epochs")
    parser.add_argument('-burn_in', default=0, type=int, help="epoch after which we can track validation results for early stopping")
    parser.add_argument('-patience', default=2, type=int, help="after how many validation steps of decreasing MRR training should stop")
    parser.add_argument('-lr', default=0.005, type=float, help="learning rate")
    parser.add_argument('-reg', default=0.0, type=float, help="l2 regularization parameter")
    parser.add_argument('-margin', default=2.0, type=float, help="margin")
    parser.add_argument('-dataset', default="YAGO14K", type=str, help="dataset")
    parser.add_argument('-model', default="TransE", type=str, help="knowledge graph embedding model")
    parser.add_argument('-dim', default=100, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=1, type=int, help="neg ratio")
    parser.add_argument('-batch_size', default=2048, type=int, help="batch size")
    parser.add_argument('-pipeline', default="both", type=str, help="(train|test|both)")
    parser.add_argument('-device', default="cuda:0", type=str, help="(cpu|cuda:0)")
    parser.add_argument('-setting', default="both", type=str, help="CWA|OWA|both")
    parser.add_argument('-sem', default="both", type=str, help="schema|extensional|both")
    parser.add_argument('-resume_training', default=False, type=bool)
    parser.add_argument('-resume_epoch', default=0, type=int, help='epoch at which resuming training (0 means: last epoch the model was saved)')
    parser.add_argument('-test_one_epoch', default=False, type=bool)
    parser.add_argument('-monitor_metrics', default=0, type=int)
    parser.add_argument('-proto_version', default=1, type=int)
    parser.add_argument('-aggreg_operator', default="mean", type=str, help="random|mean|frequency|weighted-frequency")
    parser.add_argument('-aggreg_option', default="most_specific", type=str, help="")
    parser.add_argument('-distribution', default="switch", type=str, help="")
    parser.add_argument('-hard_reset_proto', default=1, type=int, help="whether to re-initialize class embeddings")
    parser.add_argument('-erase_proto', default=0, type=int, help="whether to replace ent embeddings")
    parser.add_argument('-erase_inst', default=0, type=int, help="whether to replace ent embeddings")
    parser.add_argument('-switch', default=200, type=int, help="switching from proto to triple KG")
    parser.add_argument('-weights', default=0, type=int, help="")
    parser.add_argument('-discard', default=0, type=int, help="")
    parser.add_argument('-collab', default=1, type=int, help="")
    parser.add_argument('-factor', default=0.001, type=float, help="")

    # ConvE
    parser.add_argument('-input_drop', default=0.2, type=float)
    parser.add_argument('-hidden_drop', default=0.3, type=float)
    parser.add_argument('-feat_drop', default=0.3, type=float)
    parser.add_argument('-hidden_size', default=9728, type=int)
    parser.add_argument('-embedding_shape1', default=20, type=int)

    # TuckER
    parser.add_argument('-dim_e', default=100, type=int)
    parser.add_argument('-dim_r', default=100, type=int)
    parser.add_argument('-input_dropout', default=0.3, type=float)
    parser.add_argument('-hidden_dropout1', default=0.4, type=float)
    parser.add_argument('-hidden_dropout2', default=0.5, type=float)
    parser.add_argument('-label_smoothing', default=0.0, type=float)

    args = parser.parse_args()
    np.random.seed(7)
    torch.manual_seed(7)
    return args

if __name__ == '__main__':
    args = get_parameter()
    dataset = Dataset(args.dataset, args)
    model = args.model
    valid_test = True
    if args.monitor_metrics == 1:
        valid_test = False

    if args.pipeline == 'both' or args.pipeline == 'train':
        if not args.resume_training:
            print("------- Training -------")
            start = time.time()
            joint = JointTrainer(dataset, model, model, args=args)
            joint.train()

    if args.pipeline == 'both' or args.pipeline == 'test':
        if args.resume_training == False:
            epochs2test = [str(int(args.validate_every * (i + 1))) for i in range(args.ne // args.validate_every)]
        else:
            if args.resume_epoch == 0:
                # get last .pt file
                resume_epoch = max([int(f[-11:].split('=')[-1].split('.')[0]) for f in os.listdir("models/" + str(dataset.name) + "/" + str(model) + "/")]) - args.ne
            else:
                resume_epoch = args.resume_epoch
            print('Resuming at epoch ' + str(resume_epoch))
            epochs2test = [str(int(resume_epoch + args.validate_every * (i + 1))) for i in range(args.ne // args.validate_every)]
        if args.test_one_epoch == True:
            resume_epoch = args.resume_epoch
            print(resume_epoch)
            epochs2test = [str(resume_epoch)]

        dataset = Dataset(args.dataset, args)
        
        best_mrr = -1.0
        results = {}
        best_epoch = "0"
        directory = "models/" + dataset.name + "/" + model + "/instance/"
        if not os.path.exists('results/' + dataset.name + "/" + args.model + '/'):
            os.makedirs('results/' + dataset.name + "/" + args.model + '/')
        for epoch in epochs2test:
            print("Epoch n°", epoch)
            if model == 'ConvE':
                model_path = directory + "proto"+str(args.proto_version)+"_instance__dim=" +str(args.dim) +\
        "_lr="+str(args.lr) + "_input_drop="+str(args.input_drop) + "_hid_dropout="+str(args.hidden_drop) + \
        "_feat_drop="+str(args.feat_drop) + "_bs="+str(args.batch_size) + "__epoch="+str(epoch) + ".pt"
                model_inst = ConvE(dataset.num_ent(), dataset.num_rel(), args, args.device)

            elif model == 'TuckER':
                model_path = directory + "proto"+str(args.proto_version)+"_instance__dim="+str(args.dim_e) + \
        "_lr="+str(args.lr) + "_input_drop="+str(args.input_dropout) + "_hid_dropout1="+str(args.hidden_dropout1) + \
        "_hid_dropout2="+str(args.hidden_dropout2) + "_bs="+str(args.batch_size) + \
        "__epoch="+str(epoch) + ".pt"
                model_inst = TuckER(dataset.num_ent(), dataset.num_rel(), args, args.device)

            else:
                model_path = directory + "proto"+str(args.proto_version)+"_instance__dim="+str(args.dim) + \
        "_lr="+str(args.lr) + "_reg="+str(args.reg) + "_bs="+str(args.batch_size) + "__epoch="+str(epoch) + ".pt"
                if model == 'TransE':
                    model_inst = TransE(dataset.num_ent(), dataset.num_rel(), args.dim, args.device)
                elif model == 'DistMult':
                    model_inst = DistMult(dataset.num_ent(), dataset.num_rel(), args.dim, args.device)
                elif model == 'ComplEx':
                    model_inst = ComplEx(dataset.num_ent(), dataset.num_rel(), args.dim, args.device)           

            model_inst.load_state_dict(torch.load(model_path))

            if model in ['ConvE', 'TuckER']:
                tester = Tester(dataset, args, model_inst, "valid_inv")
            else:
                tester = Tester(dataset, args, model_inst, "valid")
            start = time.time()
            if True:
                if True:
                    filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10, filt_mrr_h, filt_mrr_t, \
                    filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, ext_CWA = tester.calc_valid_mrr()

                    if True:
                        if True:
                            results[int(epoch)] = {
                            "MRR": filt_mrr, "H@1": filtered_hits_at_1, "H@3": filtered_hits_at_3, "H@10": filtered_hits_at_10, \
                            "CWA_Sem@1": schema_CWA['sem1'], "CWA_Sem@3": schema_CWA['sem3'], "CWA_Sem@10": schema_CWA['sem10'], \
                            "Ext_Sem@1": ext_CWA['sem1'], "Ext_Sem@3": ext_CWA['sem3'], "Ext_Sem@10": ext_CWA['sem10']}

                    df_valid = pd.DataFrame.from_dict(results) 
                    if model == 'ConvE':
                        df_valid.to_csv ('results/' + dataset.name + "/" + args.model + '/' + '['+ date_today + ']'+ 'proto'+str(args.proto_version)+'-'+ args.model + '-valid_results-' + \
                            args.dataset + '-' + '-dim=' + \
                str(args.dim) + '-lr=' + str(args.lr) + "-input_drop="+str(args.input_drop) + "-hid_dropout="+str(args.hidden_drop) + \
                "-feat_drop="+str(args.feat_drop) + '-bs=' + str(args.batch_size) + '.csv')
                        with open('results/' + dataset.name + "/" + args.model + '/' + '['+ date_today + ']' + 'proto'+str(args.proto_version)+'-'+ args.model + '-valid_results-' + \
                            args.dataset + '-' + model + '-dim=' + \
                str(args.dim) + '-lr=' + str(args.lr) + "-input_drop="+str(args.input_drop) + "-hid_dropout="+str(args.hidden_drop) + \
                "-feat_drop="+str(args.feat_drop) + '-bs=' + str(args.batch_size) + '.json', 'w') as fp:
                            json.dump(results, fp)

                    elif model == 'TuckER':
                        df_valid.to_csv ('results/' + dataset.name + "/" + args.model + '/' + '['+ date_today + ']'+ 'proto'+str(args.proto_version)+'-'+ args.model + '-valid_results-' + \
                            args.dataset + '-' + model + "-dim_e="+str(args.dim_e) +  "-dim_r="+str(args.dim_r) + \
                '-lr=' + str(args.lr) + "-input_drop="+str(args.input_dropout) + "-hid_dropout1="+str(args.hidden_dropout1) + \
                "-hid_dropout2="+str(args.hidden_dropout2) + '-bs=' + str(args.batch_size) + '.csv')
                        with open('results/' + dataset.name + "/" + args.model + '/' + '['+ date_today + ']'+ 'proto'+str(args.proto_version)+'-'+ args.model + '-valid_results-' + \
                            args.dataset + '-' + model + "-dim_e="+str(args.dim_e) +  "-dim_r="+str(args.dim_r) + \
                '-lr=' + str(args.lr) + "-input_drop="+str(args.input_dropout) + "-hid_dropout1="+str(args.hidden_dropout1) + \
                "-hid_dropout2="+str(args.hidden_dropout2) + '-bs=' + str(args.batch_size) + '.json', 'w') as fp:
                            json.dump(results, fp)

                    else:
                        df_valid.to_csv ('results/' + dataset.name + "/" + args.model + '/' + '['+ date_today + ']'+ 'proto'+str(args.proto_version)+'-'+ args.model + '-valid_results-' + args.dataset + '-' + model + '-dim=' + \
                str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-bs=' + str(args.batch_size) + '.csv')
                        with open('results/' + dataset.name + "/" + args.model + '/' '['+ date_today + ']'+ 'proto'+str(args.proto_version)+'-'+ args.model + '-valid_results-' + args.dataset + '-' + model + '-dim=' + \
                    str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-bs=' + str(args.batch_size) + '.json', 'w') as fp:
                            json.dump(results, fp)

                if filt_mrr > best_mrr:
                    best_mrr = filt_mrr
                    best_epoch = epoch
            print(time.time() - start)
        print("Best epoch: " + best_epoch)

        print("------- Testing on the best epoch -------")

        if model == 'ConvE':
            best_model_path = directory + "proto"+str(args.proto_version)+"_instance__dim=" +str(args.dim) +\
    "_lr="+str(args.lr) + "_input_drop="+str(args.input_drop) + "_hid_dropout="+str(args.hidden_drop) + \
    "_feat_drop="+str(args.feat_drop) + "_bs="+str(args.batch_size) + "__epoch="+str(best_epoch) + ".pt"
            model_inst = ConvE(dataset.num_ent(), dataset.num_rel(), args, args.device)

        elif model == 'TuckER':
            best_model_path = directory + "proto"+str(args.proto_version)+"_instance__dim="+str(args.dim_e) + \
    "_lr="+str(args.lr) + "_input_drop="+str(args.input_dropout) + "_hid_dropout1="+str(args.hidden_dropout1) + \
    "_hid_dropout2="+str(args.hidden_dropout2) + "_bs="+str(args.batch_size) + \
    "__epoch="+str(best_epoch) + ".pt"
            model_inst = TuckER(dataset.num_ent(), dataset.num_rel(), args, args.device)

        else:
            best_model_path = directory + "proto"+str(args.proto_version)+"_instance__dim="+str(args.dim) + \
    "_lr="+str(args.lr) + "_reg="+str(args.reg) + "_bs="+str(args.batch_size) + "__epoch="+str(best_epoch) + ".pt"
            if model == 'TransE':
                model_inst = TransE(dataset.num_ent(), dataset.num_rel(), args.dim, args.device)
            elif model == 'DistMult':
                model_inst = DistMult(dataset.num_ent(), dataset.num_rel(), args.dim, args.device)
            elif model == 'ComplEx':
                model_inst = ComplEx(dataset.num_ent(), dataset.num_rel(), args.dim, args.device)           

        model_inst.load_state_dict(torch.load(best_model_path))
        model_inst.eval()

        if model in ['ConvE', 'TuckER']:
            tester = Tester(dataset, args, model_inst, "test_inv")
        else:
            tester = Tester(dataset, args, model_inst, "test")
        
        filt_mrr, filtered_hits_at_1, filtered_hits_at_3, filtered_hits_at_5, filtered_hits_at_10,\
        filt_mrr_h, filt_mrr_t, filt_h1_h, filt_h1_t, filt_h3_h, filt_h3_t, filt_h5_h, filt_h5_t, filt_h10_h, filt_h10_t, schema_CWA, ext_CWA = tester.test()

        best_ep_results = {
        "Epoch": int(best_epoch), "MRR": filt_mrr, "H@1": filtered_hits_at_1, "H@3": filtered_hits_at_3, "H@10": filtered_hits_at_10, \
        "CWA_Sem@1": schema_CWA['sem1'], "CWA_Sem@3": schema_CWA['sem3'], "CWA_Sem@10": schema_CWA['sem10'], \
        "Ext_Sem@1": ext_CWA['sem1'], "Ext_Sem@3": ext_CWA['sem3'], "Ext_Sem@10": ext_CWA['sem10']}

        df_test = pd.DataFrame.from_dict([best_ep_results])
        if model == 'ConvE':
            df_test.to_csv ('results/' + dataset.name + "/" + args.model + '/' + '['+ date_today + ']'+ 'proto'+str(args.proto_version)+'-'+ args.model + '-best-epoch_results-' + \
                args.dataset + '-' + '-dim=' + \
        str(args.dim) + '-lr=' + str(args.lr) + "-input_drop="+str(args.input_drop) + "-hid_dropout="+str(args.hidden_drop) + \
        "-feat_drop="+str(args.feat_drop) + '-bs=' + str(args.batch_size) + '.csv')
            with open('results/' + dataset.name + "/" + args.model + '/' + '['+ date_today + ']'+ 'proto'+str(args.proto_version)+'-'+ args.model + '-best-epoch_results-' + \
                args.dataset + '-' + model + '-dim=' + \
        str(args.dim) + '-lr=' + str(args.lr) + "-input_drop="+str(args.input_drop) + "-hid_dropout="+str(args.hidden_drop) + \
        "-feat_drop="+str(args.feat_drop) + '-bs=' + str(args.batch_size) + '.json', 'w') as fp:
                json.dump(best_ep_results, fp)

        elif model == 'TuckER':
            df_test.to_csv ('results/' + dataset.name + "/" + args.model + '/' + '['+ date_today + ']'+ 'proto'+str(args.proto_version)+'-'+ args.model + '-best-epoch_results-' + \
                args.dataset + '-' + model + "-dim_e="+str(args.dim_e) +  "-dim_r="+str(args.dim_r) + \
        '-lr=' + str(args.lr) + "-input_drop="+str(args.input_dropout) + "-hid_dropout1="+str(args.hidden_dropout1) + \
        "-hid_dropout2="+str(args.hidden_dropout2) + '-bs=' + str(args.batch_size) + '.csv')
            with open('results/' + dataset.name + "/" + args.model + '/' + '['+ date_today + ']'+ 'proto'+str(args.proto_version)+'-'+ args.model + '-best-epoch_results-' + \
                args.dataset + '-' + model + "-dim_e="+str(args.dim_e) +  "-dim_r="+str(args.dim_r) + \
        '-lr=' + str(args.lr) + "-input_drop="+str(args.input_dropout) + "-hid_dropout1="+str(args.hidden_dropout1) + \
        "-hid_dropout2="+str(args.hidden_dropout2) + '-bs=' + str(args.batch_size) + '.json', 'w') as fp:
                json.dump(best_ep_results, fp)

        else:
            df_test.to_csv ('results/' + dataset.name + "/" + args.model + '/' + '['+ date_today + ']'+ 'proto'+str(args.proto_version)+'-'+ args.model + '-best-epoch_results-' + args.dataset + '-' + model + '-dim=' + \
        str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-bs=' + str(args.batch_size) + '.csv')
            with open('results/' + dataset.name + "/" + args.model + '/' '['+ date_today + ']'+ 'proto'+str(args.proto_version)+'-'+ args.model + '-best-epoch_results-' + args.dataset + '-' + model + '-dim=' + \
        str(args.dim) + '-lr=' + str(args.lr) + "-reg="+str(args.reg) + '-bs=' + str(args.batch_size) + '.json', 'w') as fp:
                json.dump(best_ep_results, fp)