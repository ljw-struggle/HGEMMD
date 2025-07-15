import torch
import torch.nn as nn
import torch.nn.functional as F
import os, argparse, math, numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from model import HGEMMD
from data import load_feature_and_hyperedge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/BRCA/', help='The data dir.')
    parser.add_argument('-o', '--output_dir', default='./result/HGEMMD/BRCA/', help='The output dir.')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed.')
    args = parser.parse_args()
    if 'BRCA' in args.data_dir: 
        hidden_dim = [500]; num_epoch = 2500; lr = 1e-4; step_size = 500; num_class = 5; lambda_1 = 0.8; lambda_2 = 0.8; k = 300
    if 'ROSMAP' in args.data_dir: 
        hidden_dim = [300]; num_epoch = 2500; lr = 1e-4; step_size = 500; num_class = 2; lambda_1 = 0.5; lambda_2 = 1.0; k = 50
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_list, label, data_train_indices, data_test_indices, hyperedge_multi_omics, pre_calc_G = load_feature_and_hyperedge(args.data_dir, sigma=args.sigma, k_list=[k], is_prob=True, m_prob=1)
    dim_list = [data.shape[1] for data in data_list]
    data_list = [torch.FloatTensor(data).to(device) for data in data_list]
    
    label = torch.LongTensor(label).to(device)
    pre_calc_G = torch.FloatTensor(pre_calc_G).to(device)
    model = HGEMMD(dim_list, hidden_dim, num_class, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    for epoch in range(1, num_epoch + 1):
        model.train()
        optimizer.zero_grad()
        loss = model.forward_criterion(data_list=data_list, labeled_indices=data_train_indices, unlabeled_indices=data_test_indices, label=label, pre_calc_G=pre_calc_G, lambda_1=lambda_1, lambda_2=lambda_2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 50 == 0:
            print('Training Epoch {:d}: Loss={:.5f}'.format(epoch, loss.cpu().detach().numpy()))
            model.eval()
            with torch.no_grad():
                logit, _, _, _, _, _, _ = model.forward(data_list, pre_calc_G)
                prob = F.softmax(logit, dim=1).data.cpu().numpy()
                label_test = label[data_train_indices]
                prob_test = prob[data_train_indices]
            if 'ROSMAP' in args.data_dir or 'LGG' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
                f1 = f1_score(label_test.cpu().numpy(), prob_test.argmax(1))
                auc = roc_auc_score(label_test.cpu().numpy(), prob_test[:,1])
                print('Training Epoch {:d}: Train ACC={:.5f}, F1={:.5f}, AUC={:.5f}'.format(epoch, acc, f1, auc))
            if 'BRCA' in args.data_dir or 'KIPAN' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
                f1_weighted = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='weighted')
                f1_macro = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='macro')
                print('Training Epoch {:d}: Train ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(epoch, acc, f1_weighted, f1_macro))
            with torch.no_grad():
                logit, _, _, _, _, _, _ = model.forward(data_list, pre_calc_G)
                prob = F.softmax(logit, dim=1).data.cpu().numpy()
                label_test = label[data_test_indices]
                prob_test = prob[data_test_indices]
            if 'ROSMAP' in args.data_dir or 'LGG' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
                f1 = f1_score(label_test.cpu().numpy(), prob_test.argmax(1))
                auc = roc_auc_score(label_test.cpu().numpy(), prob_test[:,1])
                print('Training Epoch {:d}: Test ACC={:.5f}, F1={:.5f}, AUC={:.5f}'.format(epoch, acc, f1, auc))
            if 'BRCA' in args.data_dir or 'KIPAN' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
                f1_weighted = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='weighted')
                f1_macro = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='macro')
                print('Training Epoch {:d}: Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(epoch, acc, f1_weighted, f1_macro))
                
    torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint.pt"))
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint.pt'))
    model.load_state_dict(best_checkpoint)
    model.eval()
    with torch.no_grad():
        logit = model.forward(data_list, pre_calc_G)[0]
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
        label_test = label[data_test_indices]
        prob_test = prob[data_test_indices]
    if 'ROSMAP' in args.data_dir or 'LGG' in args.data_dir:
        acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
        f1 = f1_score(label_test.cpu().numpy(), prob_test.argmax(1))
        auc = roc_auc_score(label_test.cpu().numpy(), prob_test[:,1])
        print('Test ACC={:.5f}, F1={:.5f}, AUC={:.5f}'.format(acc, f1, auc))
    if 'BRCA' in args.data_dir or 'KIPAN' in args.data_dir:
        acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
        f1_weighted = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='weighted')
        f1_macro = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='macro')
        print('Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(acc, f1_weighted, f1_macro))
        