import os
import random
import tensorflow as tf

def load_image_and_mask(image_path, mask_path):
    # Đọc hình ảnh
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # Giải mã hình ảnh
    image = tf.cast(image, tf.float32) / 255.0  # Chuẩn hóa giá trị pixel cho hình ảnh

    # Kiểm tra giá trị NaN hoặc Infinity trong hình ảnh
    image = tf.debugging.check_numerics(image, "Image tensor contains NaN or Inf")
    image = tf.where(tf.math.is_finite(image), image, tf.zeros_like(image))

    # Đọc mặt nạ
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)  # Giải mã mặt nạ
    mask = tf.cast(mask, tf.uint8)  # Cast mask to uint8 first
    mask = tf.cast(mask > 0, tf.float32)  # Now perform the comparison

    # Kiểm tra giá trị NaN hoặc Infinity trong mặt nạ
    mask = tf.debugging.check_numerics(mask, "Mask tensor contains NaN or Inf")
    mask = tf.where(tf.math.is_finite(mask), mask, tf.zeros_like(mask))

    return image, mask

def create_dataset(images_path, masks_path, batch_size=6, num_samples=None, data_ratio=25):
    label_dict = {'normal': [], 'has_pneumo': []}

    # Tải đường dẫn hình ảnh
    for label in label_dict.keys():
        train_path = os.path.join(images_path, label, 'train')
        for root, _, files in os.walk(train_path):
            for filename in files:
                if filename.endswith('.png'):
                    label_dict[label].append(os.path.join(root, filename))

    selected_images = []

    data_ratio = 100 // data_ratio
    # Xác định số lượng hình ảnh cho mỗi nhãn
    count_label_0 = num_samples // data_ratio if num_samples else len(label_dict['normal'])
    count_label_1 = num_samples - count_label_0 if num_samples else len(label_dict['has_pneumo'])

    selected_images.extend(label_dict['has_pneumo'][:count_label_1])

    if count_label_0 > 0:
        selected_images.extend(random.sample(label_dict['normal'], min(count_label_0, len(label_dict['normal']))))

    random.shuffle(selected_images)

    # Tạo danh sách cặp (hình ảnh, mặt nạ) sau khi kiểm tra sự tồn tại của mặt nạ
    image_mask_pairs = []
    for img in selected_images:
        mask_path = os.path.join(masks_path, os.path.relpath(img, images_path)).replace('images', 'mask')
        if os.path.exists(mask_path):
            image_mask_pairs.append((img, mask_path))

    # Tạo Dataset từ danh sách
    dataset = tf.data.Dataset.from_tensor_slices(image_mask_pairs)

    # Sử dụng map để tải và xử lý dữ liệu
    dataset = dataset.map(lambda img_mask: load_image_and_mask(img_mask[0], img_mask[1]), num_parallel_calls=tf.data.AUTOTUNE)

    # Thay đổi kích thước batch và xáo trộn dữ liệu
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
