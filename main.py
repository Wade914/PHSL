import argparse
import copy
import time
import numpy as np
import torch
import pickle
import torch.nn.functional as F
import PROSE_HGNN.HGNN_utils.hypergraph_utils as hgut
from data_loader import load_data, new_load_data
from model import GCL, HGNN_Classifer
from graph_learners import *
from PROSE_HGNN.utils import *
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.cluster import KMeans
from PROSE_HGNN.config import get_config
from PROSE_HGNN.datasets.visual_data import load_feature_construct_H
from PROSE_HGNN.datasets.data_preprocess import data_process
import random
from PROSE_HGNN.models import HGNN
from PROSE_HGNN.models.ConvNet import ConvNet
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PROSE_HGNN.layers import HGNNConv_dense
from sklearn.preprocessing import MinMaxScaler, scale, MaxAbsScaler, normalize
import pandas as pd
import pickle

# Function to count model parameters and display them in a nice format
def get_model_summary(model):
    """
    Returns a summary of the model architecture and parameter counts similar to torchsummary
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Estimate model size in MB (assuming float32/4 bytes per parameter)
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    # Format parameter counts with commas for readability
    def format_count(count):
        return f"{count:,}"
    
    # Create layer-by-layer parameter information
    layer_info = []
    max_name_length = 0
    for name, module in model.named_modules():
        if list(module.children()):  # Skip container modules
            continue
        
        params = sum(p.numel() for p in module.parameters())
        if params > 0:  # Only include layers with parameters
            max_name_length = max(max_name_length, len(name))
            layer_info.append((name, params))
    
    layer_summary = ""
    if layer_info:
        layer_summary = "\nLayer-wise Parameter Information:\n"
        layer_summary += f"{'Layer Name'.ljust(max_name_length+2)}{'Parameters'}\n"
        layer_summary += f"{'-'*(max_name_length+2)}{'-'*15}\n"
        
        for name, params in layer_info:
            layer_summary += f"{name.ljust(max_name_length+2)}{format_count(params)}\n"
    
    # Create summary string
    summary = (
        f"\n{'='*50}\n"
        f"Total params: {format_count(total_params)}\n"
        f"Trainable params: {format_count(trainable_params)}\n"
        f"Non-trainable params: {format_count(non_trainable_params)}\n"
        f"Model size: {model_size_mb:.2f} MB\n"
        f"{'='*50}\n"
        f"{layer_summary}"
    )
    
    return summary

EOS = 1e-10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def save_as_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
def extract_values(matrix):
    result = []
    num_rows, num_cols = matrix.shape

    for col in range(num_cols):
        column_indices = np.where(matrix[:, col] != 0)[0]
        result.append(column_indices.tolist())

    return result

def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c='blue')
    ax1.set_ylabel('Loss', color='blue', fontsize=14)
    ax1.tick_params(axis='y', colors='blue', labelsize=12)

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c='red')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('Accuracy', color='red', fontsize=14)
    ax2.tick_params(axis='y', colors='red', labelsize=12)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)


    def loss_cls(self, model, mask, features, adj, labels):
        G = hgut.generate_G_from_H(adj)
        G = G.to(device)
        logits = model(features, G)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def loss_cls_auc(self, model, mask, features, adj, labels):
        # labels = torch.argmax(labels, dim=1)
        # print(labels)
        # print("mask",mask.shape)
        G = hgut.generate_G_from_H(adj).cuda()
        # G = hgut.generate_G_from_H(adj)
        # G = G.to(device)
        train_out = features[mask]
        features = features.to(device)
        # print(features.shape)
        # print(G.shape)

        logits = model(features, G)
        labels = labels.to(torch.float)
        # print(labels.shape)
        # print(logits.shape)
        # logp = F.log_softmax(logits, 1)
        logp = torch.tensor(logits)
        # labels = labels.to('cpu')
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss_fn(logits[mask], labels[mask])
        accu = accuracy(logp[mask], labels[mask],epoch=0)
        return loss, accu
    def loss_cls_auc_val(self, model, mask, features, adj, labels,epoch):

        G = hgut.generate_G_from_H(adj).cuda()
        G = G.to(device)
        features = features.to(device)
        logits = model(features, G)
        labels = labels.to(torch.float)
        logp = torch.tensor(logits)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss_fn(logits[mask], labels[mask])
        accu = accuracy(logp[mask], labels[mask],epoch)
        return loss, accu
    def loss_cls_auc_test(self, model, mask, features, adj, labels):

        G = hgut.generate_G_from_H(adj).cuda()
        G = G.to(device)
        features = features.to(device)
        logits = model(features, G)
        labels = labels.to(torch.float)
        logp = torch.tensor(logits)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss_fn(logits[mask], labels[mask])
        accu = accuracy(logp[mask], labels[mask],epoch=0)

        return loss, accu

    def deep_conv(self,features,cfg,conv):
        features = conv(features)

        anchor_adj = hgut.construct_H_with_KNN_conv(features, K_neigs=cfg['K_neigs'],
                                               split_diff_scale=False,
                                               is_probH=cfg['is_probH'], m_prob=cfg['m_prob']).cuda()
        return features,anchor_adj

    def gen_auc_mima(self, logits, label):
        # preds = torch.argmax(logits, dim=1)
        preds = np.where(F.sigmoid(logits.cpu()) > 0.5, 1, 0)
        preds = torch.tensor(preds).to('cpu')
        test_f1_macro = f1_score(label.cpu(), preds, average='macro')
        test_f1_micro = f1_score(label.cpu(), preds, average='micro')

        best_proba = F.softmax(logits, dim=1)
        if logits.shape[1] != 2:
            auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
                                                    y_score=best_proba.detach().cpu().numpy(),
                                                    multi_class='ovr'
                                                    )
        else:
            auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
                                                    y_score=best_proba[:,1].detach().cpu().numpy()
                                                    )
        return test_f1_macro, test_f1_micro, auc

    # def loss_gcl(self, model, graph_learner, features, anchor_adj, pos_infos, n_indices, mi_type = 'learn'):
    def loss_gcl(self, model, graph_learner, features, anchor_adj, G, mi_type='learn'):
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)
        z1, _ = model(features_v1, G, 'anchor')
        #
        #
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)
        #
        # learned_adj, embeddings_ = graph_learner(features, anchor_adj)
        learned_adj, prediction_adj, adj_, new_features = graph_learner(features, anchor_adj)
        torch.diagonal(prediction_adj).fill_(1)
        prediction_adj_G = hgut.generate_G_from_H(prediction_adj).to(device)
        if mi_type == 'learn':
            z2, _ = model(features_v2, learned_adj, 'learner')
        elif mi_type == 'final':
            z2, _ = model(features_v2, prediction_adj_G, 'learner')
            # z2, _ = model(features, prediction_adj, 'learner')

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)
        return loss, learned_adj, prediction_adj, adj_, new_features
        # return learned_adj, embeddings_, train_mask, test_mask, val_mask



    def Normlization(self, features):
        features_min = torch.min(features)
        features_max = torch.max(features)
        features_normalized = (features - features_min) / (features_max - features_min + 1e-8)
        return features_normalized


    def train(self, args):


        cfg = get_config('config/config.yaml')
        sample_length = 1024
        file_path = ['data/data/normal.csv', 'data/data/fault1.csv', 'data/data/fault2.csv', 'data/data/fault3.csv',
                     'data/data/2021_59.csv', 'data/data/2021_60.csv', 'data/data/2021_61.csv']

        print(args)

        # torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        X, Y,  idx, number_list = \
            data_process(file_path, sample_length)
        features, labels, nclasses, train_mask, test_mask, val_mask, adj_original = \
            load_feature_construct_H(

                X,
                Y,
                idx,
                m_prob=cfg['m_prob'],
                K_neigs=cfg['K_neigs'],
                is_probH=cfg['is_probH'],)

        corresponding_test_aucc = []
        test_accuracies = []
        validation_accuracies = []
        test_aucs = []
        validation_aucs = []
        test_f1_macros = []
        validation_f1_macros = []
        # a = np.array(features)
        time_history = []
        for trial in range(args.ntrials):

            print('seed = ', args.seeds[trial])
            self.setup_seed(args.seeds[trial])

            loss_history = []
            val_acc_history = []
            if args.sparse:
                anchor_adj_raw = adj_original
            else:
                anchor_adj_raw = torch.from_numpy(adj_original)


            anchor_adj = anchor_adj_raw
            nfeats = args.pred_len
            # nfeats = 1024
            X = torch.tensor(X).to(device)
            train_mask = train_mask.to(device)
            val_mask = val_mask.to(device)
            test_mask = test_mask.to(device)
            features = features.to(device)
            # HG = HG.to(X.device)
            labels = labels.to(device)

            if not args.sparse:
                anchor_adj = torch.tensor(anchor_adj).to(device)

            graph_learner = Stage_GNN_learner(2, nfeats, args.graph_learner_hidden_dim, args.k, args.sim_function, args.sparse,
                                              args.activation_learner, args.internal_type, args.stage_ks, args.share_up_gnn,
                                              args.fusion_ratio, args.stage_fusion_ratio, args.epsilon,
                                              args.add_vertical_position, args.v_pos_dim, args.dropout_v_pos,
                                              args.up_gnn_nlayers, args.dropout_up_gnn, args.add_embedding)

            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                         emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                         dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)
            convnet = ConvNet(enc_in=args.enc_in, seq_len=args.seq_len, pred_len=args.pred_len, patch_len=args.patch_len, stride=args.stride,
                              mixer_kernel_size=args.mixer_kernel_size, d_model=args.d_model, dropout=args.dropout_Conv, head_dropout=args.dropout_head, e_layers=args.e_layers)
            classifier = HGNN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses,
                                        dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, sparse=args.sparse, batch_norm=args.bn_cls)

            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr_cl, weight_decay=args.w_decay_cl)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr_gsl, weight_decay=args.w_decay_gsl)
            optimizer_classifer = torch.optim.Adamax(classifier.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)
            optimizer_conv = torch.optim.Adam(convnet.parameters(), lr=args.lr_conv, weight_decay=args.w_decay_conv)

            model = model.to(device)
            graph_learner = graph_learner.to(device)
            classifier = classifier.to(device)
            convnet = convnet.to(device)
            
            # Print model summaries with parameter counts
            print("\n=== ConvNet Model Summary ===")
            print(convnet)
            print(get_model_summary(convnet))
            
            # Test forward pass with sample input
            batch_size = 2  # Small batch size for tracing
            sample_input = torch.zeros((batch_size, args.enc_in, args.seq_len)).to(device)
            print(f"Input shape: {sample_input.shape}")
            with torch.no_grad():
                try:
                    sample_output = convnet(sample_input)
                    print(f"Output shape: {sample_output.shape}")
                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    print("Trying alternative input format...")
                    sample_input = torch.zeros((batch_size, args.seq_len, args.enc_in)).to(device)
                    print(f"Alternative input shape: {sample_input.shape}")
                    try:
                        sample_output = convnet(sample_input)
                        print(f"Output shape: {sample_output.shape}")
                    except Exception as e2:
                        print(f"Error with alternative format: {e2}")

            print("\n=== GCL Model Summary ===")
            print(model)
            print(get_model_summary(model))

            print("\n=== Graph Learner Model Summary ===")
            print(graph_learner)
            print(get_model_summary(graph_learner))

            print("\n=== Classifier Model Summary ===")
            print(classifier)
            print(get_model_summary(classifier))
            
            # Print additional information
            print(f"Classifier input dimensions: features - {nfeats}, output classes - {nclasses}")
            print(f"Classifier hidden dimension: {args.hidden_dim_cls}")
            
            # Optionally test forward passes
            with torch.no_grad():
                try:
                    print("\nTesting GCL model forward pass:")
                    dummy_features = torch.zeros((features.shape[0], nfeats)).to(device)
                    dummy_adj = torch.zeros((features.shape[0], features.shape[0])).to(device)
                    z, _ = model(dummy_features, dummy_adj, 'anchor')
                    print(f"GCL output shape: {z.shape}")
                except Exception as e:
                    print(f"Error during GCL forward pass: {e}")
                
                try:
                    print("\nTesting Classifier model forward pass:")
                    dummy_features = torch.zeros((features.shape[0], nfeats)).to(device)
                    dummy_G = torch.zeros((features.shape[0], features.shape[0])).to(device)
                    logits = classifier(dummy_features, dummy_G)
                    print(f"Classifier output shape: {logits.shape}")
                except Exception as e:
                    print(f"Error during Classifier forward pass: {e}")

            best_features = None
            best_loss = 10000
            best_val = 0
            best_auc = 0
            best_macro_f1 = 0
            best_test = 0
            corr_test = 0
            best_epch = 0
            bad_counter = 0
            best_adj = None
            best_classifier = None
            start_time = time.time()

            for epoch in range(1, args.epochs + 1):
                # schedular.step()
                model.train()
                graph_learner.train()
                classifier.train()
                convnet.train()
                features,anchor_adj = self.deep_conv(X,cfg,convnet)
                G = hgut.generate_G_from_H(anchor_adj).cuda()
                mi_loss, Adj, prediction_adj, adj_, new_features = self.loss_gcl(model, graph_learner, features, anchor_adj, G, args.head_tail_mi_type)

                semi_loss, train_accu = self.loss_cls_auc(classifier, train_mask, features, prediction_adj, labels)
                if args.head_tail_mi:
                    final_loss = semi_loss + mi_loss * args.mi_ratio
                else:
                    final_loss = semi_loss

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                optimizer_classifer.zero_grad()
                optimizer_conv.zero_grad()

                final_loss.backward()
                loss_history.append(final_loss.item())

                optimizer_cl.step()
                optimizer_learner.step()
                optimizer_conv.step()
                optimizer_classifer.step()
                if epoch % args.eval_freq == 0:
                    classifier.eval()
                    with torch.no_grad():
                        val_loss, val_accu = self.loss_cls_auc_val(classifier, val_mask, features, prediction_adj, labels,epoch)
                        test_loss, test_accu = self.loss_cls_auc_val(classifier, test_mask, features, prediction_adj, labels, epoch)

                    if test_accu >= best_test:
                        best_test = test_accu
                    if val_accu >= best_val:

                        best_val = val_accu
                        corr_test = test_accu
                        best_model = copy.deepcopy(classifier)
                    if best_loss >= val_loss and val_accu >= best_val:
                        # print('--------------- update!----------------')
                        bad_counter = 0
                        best_epoch = epoch
                        best_loss = val_loss
                        best_features = features
                        best_adj = prediction_adj

                        best_classifier = copy.deepcopy(classifier.state_dict())
                    else:
                        bad_counter += 1

                    if bad_counter >= args.patience_cls:
                        break
                    val_acc_history.append(val_accu)
                    print("Epoch {:05d} |MI Loss {:.4f} | Eval Loss {:.4f} | Eval ACC {:.4f} | Test ACC {:.4f}".format(
                        epoch, mi_loss.item(), val_loss.item(), val_accu.item(), test_accu.item()))

            best_model.eval()
            classifier.load_state_dict(best_classifier)
            classifier.eval()
            test_loss, test_accu = self.loss_cls_auc_test(classifier, test_mask, best_features, best_adj, labels,epoch)
            print("Best Epoch {:05d} ".format(best_epoch))
            print("test",test_accu)
            torch.cuda.empty_cache()
            end_time = time.time()
            training_time = end_time - start_time
            time_history.append(training_time)
            validation_accuracies.append(best_val.item())
            test_accuracies.append(test_accu.item())
            corresponding_test_aucc.append(corr_test.item())

            print("Trial: ", trial + 1)
            print("Best val ACC: ", best_val.item())
            print("Best test ACC: ", best_test.item())
            print("Corresponding test ACC: ", test_accu.item())
            print("REAL Corresponding test ACC: ", test_accu.item())
            print("Training time: {:.2f} seconds".format(training_time))
            plot_loss_with_acc(loss_history, val_acc_history)

        if trial != 0:
            print('---------------------------results as follows------------------------------')
            self.print_results(validation_accuracies, test_accuracies, corresponding_test_aucc, validation_aucs, test_aucs, validation_f1_macros, test_f1_macros,time_history)


    def print_results(self, validation_accu, test_accu, corr_accu, validation_aucs, test_aucs, validation_f1_macros, test_f1_macros,time_history):
        s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu),np.std(test_accu))
        corresponding_test = "Corresponding Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu), np.std(corr_accu))
        correspond_test = "Corresponding Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(corr_accu), np.std(corr_accu))
        time = "Time Cost: {:.4f} +/- {:.4f}".format(np.mean(time_history), np.std(time_history))
        print(s_val)
        print(s_test)
        print(corresponding_test)
        print(correspond_test)
        print(time)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='citeseer',
                        choices=['cora', 'citeseer', 'polblogs', 'wine', 'cancer', 'digits'])
    parser.add_argument('-ntrials', type=int, default=1)
    parser.add_argument('-seeds', nargs='+', type=list, default=[0])
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-eval_freq', type=int, default=1)
    parser.add_argument('-epochs', type=int, default=300)
    parser.add_argument('-gpu', type=int, default=0)

    parser.add_argument('-preprocess', type=int, default=0)

    # GSL Module
    parser.add_argument('-graph_learner_hidden_dim', type=int, default=256)
    parser.add_argument('-internal_type', type=str, default='mlp', choices=['gnn', 'mlp'])
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'weight_cosine'])
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    parser.add_argument('-epsilon', type=float, default=0.3)
    parser.add_argument('-k', type=int, default=0)
    parser.add_argument('-fusion_ratio', type=float, default=0.1)
    parser.add_argument('-stage_fusion_ratio', type=float, default=0.05)

    parser.add_argument('-stage_ks', nargs='+', type=list, default=[])
    parser.add_argument('-split_deep', type=int, default=3)
    parser.add_argument('-split_prop', type=float, default=0.7)

    parser.add_argument('-share_up_gnn', type=int, default=1)
    parser.add_argument('-up_gnn_nlayers', type=int, default=2)
    parser.add_argument('-dropout_up_gnn', type=float, default=0.5)

    parser.add_argument('-add_vertical_position', type=int, default=1)
    parser.add_argument('-v_pos_dim', type=int, default=64)
    parser.add_argument('-dropout_v_pos', type=float, default=0.5)

    parser.add_argument('-add_embedding', type=int, default=1)

    parser.add_argument('-lr_gsl', type=float, default=0.001)
    parser.add_argument('-w_decay_gsl', type=float, default=0.0005)

    # GCL Module
    parser.add_argument('-head_tail_mi', type=int, default=1)
    parser.add_argument('-mi_ratio', type=float, default=0.1)
    parser.add_argument('-head_tail_mi_type', type=str, default='final', choices=['learn', 'final'])

    # GCL Module - Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.5)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.5)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)

    # GCL Module - Framework
    parser.add_argument('-lr_cl', type=float, default=0.01)
    parser.add_argument('-w_decay_cl', type=float, default=0.0005)
    parser.add_argument('-hidden_dim', type=int, default=32)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=16)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)

    # ConvNet
    parser.add_argument('-enc_in', type=int, default=1)
    parser.add_argument('-seq_len', type=int, default=1024)
    parser.add_argument('-pred_len', type=int, default=256)
    parser.add_argument('-patch_len', type=int, default=8)
    parser.add_argument('-stride', type=int, default=4)
    parser.add_argument('-mixer_kernel_size', type=int, default=7)
    parser.add_argument('-d_model', type=int, default=8)
    parser.add_argument('-dropout_Conv', type=float, default=0.5)
    parser.add_argument('-dropout_head', type=float, default=0.5)
    parser.add_argument('-e_layers', type=int, default=1)
    parser.add_argument('-lr_conv', type=float, default=0.01)
    parser.add_argument('-w_decay_conv', type=float, default=0.0005)

    # Evaluation Network (Classification)
    parser.add_argument('-bn_cls', type=int, default=None)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=5e-4)
    parser.add_argument('-hidden_dim_cls', type=int, default=64)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.5)
    parser.add_argument('-nlayers_cls', type=int, default=1)
    parser.add_argument('-patience_cls', type=int, default=500)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=0.9999)
    parser.add_argument('-c', type=int, default=0)

    args = parser.parse_args()

    if args.split_deep>0 and args.split_prop>0:
        args.stage_ks = [args.split_prop] * args.split_deep

        
    experiment = Experiment()
    experiment.train(args)
