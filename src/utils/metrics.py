import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

def dice_loss(y_true, y_pred, smooth=1e-5):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.3, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = intersection / (union + tf.keras.backend.epsilon())
    return iou

def calculate_iou(y_true, y_pred):
    """
    Tính toán IoU (Intersection over Union) cho ảnh nhị phân.

    Parameters:
    - y_true: Mảng mask thật (ground truth)
    - y_pred: Mảng mask dự đoán

    Returns:
    - iou: Giá trị IoU
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = intersection / union if union != 0 else 1.0
    return iou

def calculate_dice(y_true, y_pred):
    """
    Tính toán Dice Coefficient cho ảnh nhị phân.

    Parameters:
    - y_true: Mảng mask thật (ground truth)
    - y_pred: Mảng mask dự đoán

    Returns:
    - dice: Giá trị Dice Coefficient
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    intersection = np.logical_and(y_true, y_pred).sum()
    dice = (2 * intersection) / (y_true.sum() + y_pred.sum()) if (y_true.sum() + y_pred.sum()) != 0 else 1.0
    return dice

def calculate_precision(y_true, y_pred):
    """
    Tính toán precision cho mô hình phân đoạn.
    
    Parameters:
    - y_true: Danh sách các ảnh mask thật (ground truth)
    - y_pred: Danh sách các ảnh mask dự đoán
    
    Returns:
    - precision: Giá trị precision
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    precision = precision_score(y_true, y_pred, average='binary')
    return precision

def calculate_recall(y_true, y_pred):
    """
    Tính toán recall cho mô hình phân đoạn.
    
    Parameters:
    - y_true: Danh sách các ảnh mask thật (ground truth)
    - y_pred: Danh sách các ảnh mask dự đoán
    
    Returns:
    - recall: Giá trị recall
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    recall = recall_score(y_true, y_pred, average='binary')
    return recall

def calculate_f1_score(y_true, y_pred):
    """
    Tính toán F1-score cho mô hình phân đoạn.
    
    Parameters:
    - y_true: Danh sách các ảnh mask thật (ground truth)
    - y_pred: Danh sách các ảnh mask dự đoán
    
    Returns:
    - f1_score: Giá trị F1-score
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    f1 = f1_score(y_true, y_pred, average='binary')
    return f1
