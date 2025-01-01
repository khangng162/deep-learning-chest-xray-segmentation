import os
import random
import tensorflow as tf
import numpy as np
from utils.data_loader import load_image_and_mask
from utils.metrics import calculate_precision, calculate_recall, calculate_f1_score, calculate_iou, calculate_dice
from utils.visualization import plot_segmentation_results

def evaluate_models(models, images_dir, masks_dir, threshold=0.5):
    """
    Đánh giá nhiều mô hình trên một ảnh cụ thể hoặc một ảnh ngẫu nhiên từ thư mục, sau đó hiển thị kết quả dự đoán.

    Parameters:
    - models: Dictionary chứa các mô hình TensorFlow đã load
    - images_dir: Đường dẫn tới thư mục chứa ảnh hoặc một ảnh cụ thể
    - masks_dir: Đường dẫn tới thư mục chứa mask hoặc một mask cụ thể
    - threshold: Ngưỡng để chuyển đổi kết quả dự đoán thành nhị phân

    Returns:
    - None
    """
    # Kiểm tra nếu images_dir là tệp ảnh cụ thể
    if os.path.isfile(images_dir):
        test_image_path = images_dir
        true_mask_path = masks_dir  # Giả định masks_dir cũng là đường dẫn cụ thể
    elif os.path.isdir(images_dir):
        # Lấy danh sách ảnh và chọn ngẫu nhiên
        image_files = os.listdir(images_dir)
        random_image_file = random.choice(image_files)

        # Đường dẫn ảnh và mask tương ứng
        test_image_path = os.path.join(images_dir, random_image_file)
        true_mask_path = os.path.join(masks_dir, random_image_file)  # Giả sử tên file mask giống với ảnh
    else:
        raise ValueError("images_dir phải là đường dẫn tệp ảnh hoặc thư mục chứa ảnh!")
    
    print(f"Đang đánh giá trên ảnh: {os.path.basename(test_image_path)}")

    # Tải và chuẩn bị ảnh test và mask
    test_image_array, true_mask_array = load_image_and_mask(test_image_path, true_mask_path)

    # Mở rộng chiều cho ảnh đầu vào để phù hợp với mô hình
    test_image_array_expanded = tf.expand_dims(test_image_array, axis=0)

    # Dự đoán với từng mô hình và lưu kết quả vào dictionary
    predicted_masks = {}
    for model_name, model in models.items():
        prediction = model.predict(test_image_array_expanded)
        predicted_mask = (prediction[0] > threshold).astype(np.uint8)
        predicted_masks[model_name] = predicted_mask

    # Đảm bảo true_mask_array là mảng nhị phân
    true_mask_array = tf.cast(true_mask_array > 0, tf.uint8)

    if isinstance(test_image_array, tf.Tensor):
        test_image_array = test_image_array.numpy()

    if isinstance(true_mask_array, tf.Tensor):
        true_mask_array = true_mask_array.numpy()

    # Tính toán và hiển thị chỉ số đánh giá cho từng mô hình
    for model_name, predicted_mask in predicted_masks.items():
        if np.sum(true_mask_array) == 0:  # Kiểm tra nếu không có vùng bệnh
            if np.sum(predicted_mask) == 0:  # Cả true mask và predicted mask đều không có bệnh
                precision, recall, f1, iou, dice = 1.0, 1.0, 1.0, 1.0, 1.0  # Đặt chỉ số thành 1
            else:  # true mask không có bệnh, nhưng predicted mask có bệnh
                precision, recall, f1, iou, dice = 0.0, 0.0, 0.0, 0.0, 0.0  # Đặt chỉ số thành 0
        else:
            precision = calculate_precision(true_mask_array, predicted_mask)
            recall = calculate_recall(true_mask_array, predicted_mask)
            f1 = calculate_f1_score(true_mask_array, predicted_mask)
            iou = calculate_iou(true_mask_array, predicted_mask)
            dice = calculate_dice(true_mask_array, predicted_mask)
        
        print(f"{model_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}")

    # Trực quan hóa kết quả
    plot_segmentation_results(np.squeeze(test_image_array), true_mask_array, predicted_masks)
