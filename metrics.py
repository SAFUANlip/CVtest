import numpy as np
import cv2

def get_borders_ver(img, labels):
    """
    Функция находит горизонтальные границы интересных зон для изображения

    :param img: изображение, для которого ищутся границы
    :param labels: аннотация к изображению
    :return: список пар, первое значение в которых отвечает верхней границе, а второе - высоте bb
    """

    res = []
    for i in range(len(labels)):
        y = int(float(labels[i].split(" ")[2]) * img.shape[0])
        height = int(float(labels[i].split(" ")[4]) * img.shape[0])

        res.append([y - height // 2, y + height // 2])

    return res


def get_borders_hor(img, labels):
    """
    Функция находит вертикальные границы интересных зон для изображения

    :param img: изображение, для которого ищутся границы
    :param labels: аннотация к изображению
    :return: список пар, первое значение в которых отвечает левой границе, а второе - ширине bb
    """

    res = []

    for i in range(len(labels)):
        x = int(float(labels[i].split(" ")[1]) * img.shape[1])
        width = int(float(labels[i].split(" ")[3]) * img.shape[1])

        res.append([x - width // 2, x + width // 2])

    return res


def compute_overlap(a: np.array, b: np.array) -> np.array:
    """
    Args
        a: (N, 4) ndarray of float [xmin, ymin, xmax, ymax]
        b: (K, 4) ndarray of float [xmin, ymin, xmax, ymax]

    Returns
        overlaps: (N, K) ndarray of overlap between boxes a and boxes b
    """
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    dx = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], axis=1), b[:, 0])
    dy = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], axis=1), b[:, 1])

    intersection = np.maximum(dx, 0) * np.maximum(dy, 0)
    union = np.expand_dims(a_area, axis=1) + b_area - intersection
    overlaps = intersection / union

    return overlaps

def iou_per_img(pred_path, label_path, img_path, name):
    predicts = []
    labels = []
    img = cv2.imread(f"{img_path}/{name}.png")

    with open(f'{pred_path}{name}.txt') as f:
        for line in f:
            predicts.append(line)
    with open(f'{label_path}{name}.txt') as f:
        for line in f:
            labels.append(line)

    hor_pred = get_borders_hor(img, predicts)
    ver_pred = get_borders_ver(img, predicts)
    hor_label = get_borders_hor(img, labels)
    ver_label = get_borders_ver(img, labels)

    pred_bb = []
    for i in range(len(hor_pred)):
        pred_bb.append([hor_pred[i][0], ver_pred[i][0], hor_pred[i][1], ver_pred[i][1]])

    label_bb = []
    for i in range(len(hor_label)):
        label_bb.append([hor_label[i][0], ver_label[i][0], hor_label[i][1], ver_label[i][1]])

    overlap = compute_overlap(np.array(pred_bb), np.array(label_bb))
    best_iou_idx = overlap.argmax(axis=1)
    iou_arr = []
    for i in range(len(best_iou_idx)):  # [0]
        if predicts[i].split()[0] == labels[best_iou_idx[i]][0]:
            iou_arr.append(overlap[i][best_iou_idx[i]])
            #print(predicts[i].split()[0], labels[best_iou_idx[i]][0], overlap[i][best_iou_idx[i]])
        else:
            iou_arr.append(0)
    # print(f"file_name: {name}, mean iou: {np.mean(iou_arr)}")
    return (f"{name}", np.mean(iou_arr))