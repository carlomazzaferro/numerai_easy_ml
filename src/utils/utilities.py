import os
import pandas
import glob
import json
from src.utils import definitions
import matplotlib.pyplot as plt
from sklearn import metrics


def upload_predictions(df):
    pass


def scoring(preds, model_type, targs):

    preds = [i[0] for i in preds]
    targs = [i[0] for i in targs]

    as_df = pandas.DataFrame([preds, targs], index=['preds', 'targs']).T
    fpr, tpr, roc_auc = return_auc_metric(as_df)

    plot = make_plot(fpr, tpr, roc_auc, model_type)

    return as_df, fpr, tpr, roc_auc, plot


def r2_score(df):
    return metrics.r2_score(df['preds'], df['targs'])


def mean_sq_error(df):
    return metrics.mean_squared_error(df['preds'], df['targs'])


def abs_error(df):
    return metrics.mean_absolute_error(df['preds'], df['targs'])


def return_auc_metric(df):

    preds = df['preds'].values
    targs = df['targs'].values
    fpr, tpr, _ = metrics.roc_curve(targs, preds)
    roc_auc = metrics.auc(fpr, tpr)

    return fpr, tpr, roc_auc


def make_plot(fpr, tpr, roc_auc, name_):

    lw = 2

    fig = plt.figure()
    plt.plot(fpr, tpr, color='deeppink', linestyle=':', linewidth=lw, label='%s Regressor (area = %0.2f)' % (name_,
                                                                                                             roc_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve %s ' % name_)
    plt.legend(loc="lower right")

    return fig


def save_results_to_file(params, df, plot, fpr, tpr, joined):

    res_dir = definitions.MODEL_OUTPUTS
    final_res_dir = res_dir + params['run_id']
    os.makedirs(final_res_dir)

    with open(final_res_dir + '/params_dump.json', 'w') as p:
        json.dump(params, p)

    df.to_csv(final_res_dir + '/preds_targs.csv')
    joined.to_csv(final_res_dir + '/submission.csv')
    plot.savefig(final_res_dir + '/roc_curve.png', orientation='portrait', bbox_inches='tight')
    fp_tp_df = pandas.DataFrame([fpr, tpr], index=['FP', 'TP']).T
    fp_tp_df.to_csv(final_res_dir + '/FP_TP.csv')

    print('Results written to %s' % os.path.realpath(final_res_dir))


def assign_id(params, res_dir):

    res_files = glob.glob(res_dir + '%s*' % (params['model_type']))
    num_files = len(res_files)
    return params['model_type'] + '_' + str(num_files)

