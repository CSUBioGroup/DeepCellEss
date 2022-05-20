import warnings,os,random, pickle, torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from module import DeepEss, FocalLoss, initial_model
from utils import load_dataset, set_params, cal_metrics, print_metrics, best_acc_thr


args = set_params()

## random seed ##
random_seed = args.seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHUSHSEED'] = str(random_seed)



warnings.filterwarnings('ignore')
if torch.cuda.is_available():
    num_workers = 16
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    num_workers = 0
    device = torch.device("cpu")
    

    

def train_cv():
    seed, cell_line, seq_type, emb_type, max_len, epoch_num, batch_size, patience, threshold = \
         args.seed, args.cell_line, args.seq_type, args.emb_type, args.max_len, args.epoch_num, args.batch_size, args.patience, args.threshold
    
    max_len,  kernel_size, head_num, hidden_dim, layer_num, attn_drop, lstm_drop, linear_drop = \
        args.max_len,  args.kernel_size, args.head_num, args.hidden_dim, args.layer_num, args.attn_drop, args.lstm_drop, args.linear_drop
        
    save_path = args.save_path
#     save_path = os.path.join(r'../saved_models',cell_line)
#     if os.path.exists(save_path) == False:
#         os.makedirs(save_path)
    
    
    print(f'===================================New Training===================================')
    print("Device: ", device)
    print("Seed: ", seed)
    print("Cell line: ", cell_line)
    
    # Load datasets
    train_dataset, test_dataset = load_dataset(seq_type, emb_type, cell_line, max_len, seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(seed))

    # Model
    model = DeepEss(max_len, train_dataset.emb_dim, kernel_size, head_num, hidden_dim, layer_num, attn_drop, lstm_drop, linear_drop)
    model = model.to(device)
    
    # Optimizer
    optimizer =torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    
    # Loss function
    pos_weight = float(train_dataset.num_non / train_dataset.num_ess)
    loss = FocalLoss(gamma=0, pos_weight=pos_weight, logits=False, reduction='sum')
    # loss = nn.BCELoss()

    # Train and validation using 5-fold cross validation
    val_auprs, test_auprs = [], []
    val_aucs, test_aucs = [], []
    test_trues, kfold_test_scores = [], []
    kfold = 5
    skf = StratifiedKFold(n_splits=kfold, random_state=seed, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(train_dataset.features, train_dataset.labels)):
        print(f'\nStart training CV fold {i+1}:')
        train_sampler, val_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(seed))
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed(seed))
        
        # Train model
        initial_model(model)
        count = 0
        best_val_aupr, best_test_aupr = .0, .0
        best_val_auc, best_test_auc = .0, .0
        best_test_scores = []
        best_model = model
        for epoch in range(epoch_num):
            print(f'\nEpoch [{epoch+1}/{epoch_num}]')
            
            # Calculate prediction results and losses 
            train_trues, train_scores, train_loss = cal_by_epoch(mode='train', model=model, loader=train_loader, loss=loss, optimizer=optimizer)
            val_trues, val_scores, val_loss = cal_by_epoch(mode='val', model=model, loader=val_loader, loss=loss)
            test_trues, test_scores, test_loss = cal_by_epoch(mode='test', model=model, loader=test_loader, loss=loss)
            
            # Calculate evaluation meteics
            train_metrics = cal_metrics(train_trues, train_scores, threshold)[:]
            val_metrics = cal_metrics(val_trues, val_scores, threshold)[:]
            test_metrics = cal_metrics(test_trues, test_scores, threshold)[:]
            
            train_auc, train_aupr  = train_metrics[-2], train_metrics[-1]
            val_auc, val_aupr  = val_metrics[-2], val_metrics[-1]
            test_auc, test_aupr  = test_metrics[-2], test_metrics[-1]
            
            # Print evaluation result
            print_metrics('train', train_loss, train_metrics)
            print_metrics('valid', val_loss, val_metrics)
            # print_metrics('test', test_loss, test_metrics)

            # Sava the model by auc
            if val_auc > best_val_auc:
                count = 0
                best_model = model
                best_val_auc = val_auc
                best_val_aupr = val_aupr
                
                best_test_auc = test_auc
                best_test_aupr = test_aupr
                
                best_test_scores = test_scores
                
                print("!!!Get better model with valid AUC:{:.6f}. ".format(val_auc))
                
            else:
                count += 1
                if count >= patience:
                    torch.save(best_model, os.path.join(save_path, 'model_{}_{:.3f}_{:.3f}.pkl'.format(i+1, best_test_auc, best_test_aupr)))
                    print(f'Fold {i+1} training done!!!\n')
                    break
                
        val_auprs.append(best_val_aupr)
        test_auprs.append(best_test_aupr)
        val_aucs.append(best_val_auc)
        test_aucs.append(best_test_auc)
        kfold_test_scores.append(best_test_scores)
        
    print(f'Cell line {cell_line} model training done!!!\n')
    for i, (test_auc, test_aupr) in enumerate(zip(test_aucs, test_auprs)):
        print('Fold {}: test AUC:{:.6f}   test AUPR:{:.6f}.'.format(i+1, test_auc, test_aupr))
        
    # Average 5 models' results
    final_test_scores = np.sum(np.array(kfold_test_scores), axis=0)/kfold
    
    # Cal the best threshold
    best_acc_threshold, best_acc = best_acc_thr(test_trues, final_test_scores)
    print('The best acc threshold is {:.2f} with the best acc({:.3f}).'.format(best_acc_threshold, best_acc))

    
    # Select the best threshold by acc
    final_test_metrics = cal_metrics(test_trues, final_test_scores, best_acc_threshold)[:]
    print_metrics('Final test', test_loss, final_test_metrics)


def cal_by_epoch(mode, model, loader, loss, optimizer=None):
    # Model on train mode
    model.train() if mode == 'train' else model.eval()
    all_trues, all_scores= [],[]
    losses, sample_num = 0.0, 0
    for iter_idx, (X, y) in enumerate(loader):
        sample_num += y.size(0)
        
        # Create vaiables
        with torch.no_grad():
            X_var = torch.autograd.Variable(X.to(device).float())
            y_var = torch.autograd.Variable(y.to(device).float())
            
        # compute output
        model = model.to(device)
        output = model(X_var).view(-1)
        
        # calculate and record loss
        loss_batch = loss(output, y_var)
        losses += loss_batch.item()
        
        # compute gradient and do SGD step when training
        if mode == 'train':
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            
        all_trues.append(y_var.data.cpu().numpy())
        all_scores.append(output.data.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    return all_trues, all_scores, losses/sample_num
    


if __name__ == '__main__':
        train_cv()
