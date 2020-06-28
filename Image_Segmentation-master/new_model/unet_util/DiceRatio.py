import numpy as np

def dice_ratio(predict,label):
    """
    Dice loss 只在二元分类中使用，输入的张量只能含0，1
    :param predict:
    :param label:
    :return:
    """
    return np.sum(predict[label==1])*2.0/(np.sum(predict)+np.sum(label))