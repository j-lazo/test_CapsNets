from sklearn.metrics import roc_curve
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os
import general_functions as gf


def main(predicted, real):

    y_results, names = gf.load_predictions(predicted)
    y_2test, names_test = gf.load_labels(real)
    y_test = []
    y_pred = []
    print(len(y_results), len(names), 'predicted')
    print(len(y_2test), len(names_test), 'reals')

    for i, name in enumerate(names):
        for j, other_name in enumerate(names_test):
            if name == other_name:
                y_pred.append(float(y_results[i]))
                y_test.append(float(y_2test[j]))

    print(len(y_pred))
    print(len(y_test))
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)

    auc_keras = auc(fpr_keras, tpr_keras)

    name = real[-7:]
    plt.plot([0, 1], [0, 1], 'k--')
    label = ''.join(['(area = {:.3f})'])
    plt.plot(fpr_keras, tpr_keras, label=label.format(auc_keras))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))


    # Zoom in view of the upper left corner.
    """plt.figure()
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')"""


if __name__ == "__main__":
    plt.figure()

    predicted = 'results_VGG/predictions_rgb_VGG_dgan.csv'
    real = 'real_values/Real_values_case4_rgb.csv'

    #predicted = 'results/predictions_keras3__0_0.008_.csv'
    #real = 'real_values/Real_values_validation_plus_no_case4.csv'

    main(predicted, real)
    """
    predicted = 'results/predictions_keras2_0.0_.csv'
    real = 'real_values/Real_values_case4_rgb.csv'
    main(predicted, real)

    predicted = 'results/predictions_rgb_keras2_2_0.01_.csv'
    real = 'real_values/Real_values_case4_rgb.csv'
    main(predicted, real)

    predicted = 'results/predictions_rgb_keras2_3_0.01_.csv'
    real =  'real_values/Real_values_case4_rgb.csv'
    main(predicted, real)"""

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

