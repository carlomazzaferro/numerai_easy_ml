import matplotlib.pyplot as plt
from utils import utilities


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

