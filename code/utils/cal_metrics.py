from sklearn import metrics


def cal_metrics(all_trues, all_scores, threshold):
    """ Calculate the evaluation metrics """
    all_preds = (all_scores >= threshold)
    acc = metrics.accuracy_score(all_trues, all_preds)
    pre = metrics.precision_score(all_trues, all_preds)
    rec = metrics.recall_score(all_trues, all_preds)
    f1 = metrics.f1_score(all_trues, all_preds)
    mcc = metrics.matthews_corrcoef(all_trues, all_preds)
    fpr, tpr, _ = metrics.roc_curve(all_trues, all_scores)
    AUC = metrics.auc(fpr, tpr)
    AUPR = metrics.average_precision_score(all_trues, all_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(all_trues, all_preds, labels=[0, 1]).ravel()
    return tp, tn, fp, fn, acc, f1, pre, rec, mcc, AUC, AUPR

        
def print_metrics(data_type, loss, metrics):
    """ Print the evaluation results """
    tp, tn, fp, fn, acc, f1, pre, rec, mcc, auc, aupr = metrics
    res = '\t'.join([
        '%s:' % data_type,
        'TP=%-5d' % tp,
        'TN=%-5d' % tn,
        'FP=%-5d' % fp,
        'FN=%-5d' % fn,
        'loss:%0.5f' % loss,
        'acc:%0.3f' % acc,
        'f1:%0.3f' % f1,
        'pre:%0.3f' % pre,
        'rec:%0.3f' % rec,
        'mcc:%0.3f' % mcc,
        'auc:%0.3f' % auc,
        'aupr:%0.3f' % aupr
    ])
    print(res)
    
    
def best_f1_thr(y_true, y_score):
    """ Calculate the best threshold  with f1 """
    best_thr = 0.5
    best_f1 = 0
    for thr in range(1,100):
        thr /= 100
        tp, tn, fp, fn, acc, f1, pre, rec, mcc, AUC, AUPR = cal_metrics(y_true, y_score, thr)
        if f1>best_f1:
            best_f1 = f1
            best_thr = thr 
    return best_thr, best_f1


def best_acc_thr(y_true, y_score):
    """ Calculate the best threshold with acc """
    best_thr = 0.5
    best_acc = 0
    for thr in range(1,100):
        thr /= 100
        tp, tn, fp, fn, acc, f1, pre, rec, mcc, AUC, AUPR = cal_metrics(y_true, y_score, thr)
        if acc>best_acc:
            best_acc = acc
            best_thr = thr 
    return best_thr, best_acc
