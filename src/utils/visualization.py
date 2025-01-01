import matplotlib.pyplot as plt

def plot_segmentation_results(image, true_mask, predicted_masks, title="Segmentation Results"):
    """
    Trực quan hóa kết quả phân đoạn cho các mô hình.
    
    Parameters:
    - image: Ảnh đầu vào
    - true_mask: Mask thật (ground truth)
    - predicted_masks: Dictionary chứa các mask dự đoán từ các mô hình
    - title: Tiêu đề cho đồ thị
    
    Returns:
    - None
    """
    # Sử dụng subplots để dễ dàng kiểm soát không gian giữa các ảnh
    fig, axes = plt.subplots(1, len(predicted_masks) + 2, figsize=(15, 10))

    # Vẽ tiêu đề và điều chỉnh để tiêu đề gần ảnh
    fig.suptitle(title, fontsize=16, y=0.7)  # Điều chỉnh y để đưa tiêu đề gần phần trên của figure

    # Vẽ ảnh đầu vào
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Vẽ mask thật
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title("True Mask")
    axes[1].axis('off')

    # Vẽ các kết quả dự đoán
    for i, (model_name, mask_pred) in enumerate(predicted_masks.items()):
        axes[i + 2].imshow(mask_pred, cmap='gray')
        axes[i + 2].set_title(f"Pred: {model_name}")
        axes[i + 2].axis('off')

    # Điều chỉnh không gian giữa các subplot
    plt.subplots_adjust(wspace=0.2, hspace=0)  # Điều chỉnh khoảng cách ngang và dọc

    # Tự động điều chỉnh layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Điều chỉnh để không có chồng lấn giữa suptitle và subplot

    # Hiển thị hình ảnh
    plt.show()
