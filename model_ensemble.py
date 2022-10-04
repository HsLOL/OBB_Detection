import jittor as jt
from jdet.ops import box_iou_rotated_v1
import argparse
from pathlib import Path
import os
from tqdm import tqdm
from collections import defaultdict
import csv
import numpy as np
from jdet.ops.nms_rotated import ml_nms_rotated, nms_rotated
import cv2

""" Calculate the IoU Example with Jittor method.
poly1 = jt.array([[0, 0, 2, 2, 3, 1, 1, -1], [0, 0, 0, 2, 2, 2, 2, 0]])
poly2 = jt.array([[0, 0, 0, 2, 2, 2, 2, 0]])

obb1 = poly2obb(poly1)
obb2 = poly2obb(poly2)
ious = box_iou_rotated_v1(obb1, obb2)
>>> ious: 
>>> jt.Var([[0.33333328]
        [1.        ]], dtype=float32)

"""

def poly2obb_(polys):

    polys_np = polys

    order = polys_np.shape[:-1]
    num_points = polys_np.shape[-1] // 2
    polys_np = polys_np.reshape(-1, num_points, 2)
    polys_np = polys_np.astype(np.float32)

    obboxes = []
    for poly in polys_np:
        (x, y), (w, h), angle = cv2.minAreaRect(poly)
        if w >= h:
            angle = -angle
        else:
            w, h = h, w
            angle = -90 - angle
        theta = angle / 180 * np.pi
        obboxes.append([x, y, w, h, theta])

    if not obboxes:
        obboxes = np.zeros((0, 5))
    else:
        obboxes = np.array(obboxes)

    obboxes = obboxes.reshape(*order, 5)
    return jt.array(obboxes)

label_idx = {'Airplane': 0, 'Ship': 1, 'Vehicle': 2, 'Basketball_Court': 3, 'Tennis_Court': 4, 
             'Football_Field': 5, 'Baseball_Field': 6, 'Intersection': 7,
             'Roundabout': 8, 'Bridge': 9}

idx_label = {0: 'Airplane', 1: 'Ship', 2: 'Vehicle', 3: 'Basketball_Court', 4: 'Tennis_Court', 
             5: 'Football_Field', 6: 'Baseball_Field', 7: 'Intersection',
             8: 'Roundabout', 9: 'Bridge'}

def get_dets_each_image(csv_file_lists):
    key_image_value_dets = defaultdict(list)
    for csv_file in csv_file_lists:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                image_name, cat= line[0], line[1]
                x1, y1, x2, y2, x3, y3, x4, y4, score = list(map(float, line[2:]))
                key_image_value_dets[image_name].append([cat, x1, y1, x2, y2, x3, y3, x4, y4, score])
    return key_image_value_dets


def generate_final_csv_file(output_path, post_process_results, filename):
    with open(os.path.join(output_path, f"model_ensemble_{filename}.csv"), "w") as f:
            for imageName, dets in tqdm(post_process_results.items(), desc='generating csv file.'):
                for single_box in dets:
                    temp_txt = '{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                        imageName, single_box[0],
                        float(single_box[1]),  float(single_box[2]), float(single_box[3]),  float(single_box[4]),
                        float(single_box[5]),  float(single_box[6]),  float(single_box[7]),  float(single_box[8]),
                        float(single_box[9]))
                    f.writelines(temp_txt)


def post_process_func(image_dets_dict, nms_method, nms_iou_thr):
    res_image_dict = defaultdict(list)
    for image_name, det_lists in image_dets_dict.items():
        labels = jt.array([label_idx[cat] for cat in list(np.array(det_lists)[:, 0])])
        scores = jt.array(np.array(det_lists)[:, -1].astype(np.float32))
        bboxes = np.array(det_lists)[:, 1:9].astype(np.float32)
        rbboxes = poly2obb_(bboxes)

        if nms_method == 0:
            keep_idx = nms_rotated(rbboxes, scores, nms_iou_thr)
        else:
            keep_idx = ml_nms_rotated(rbboxes, scores, labels, nms_iou_thr)

        for bbox, score, label in zip(bboxes[keep_idx], scores[keep_idx], labels[keep_idx]):
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            res_image_dict[image_name].append([idx_label[int(label)], x1, y1, x2, y2, x3, y3, x4, y4, float(score)])
    return res_image_dict

def main(args):
    csv_file_lists = [os.path.join(args.csv_file_root_path, f) for f in os.listdir(args.csv_file_root_path)]
    image_dets_dict = get_dets_each_image(csv_file_lists)
    post_process_results = post_process_func(image_dets_dict, args.nms_format, args.nms_iou_thr)
    generate_final_csv_file(args.output_path, post_process_results, f'method-{args.nms_format}_nmsiou-{args.nms_iou_thr}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_root_path', type=str, default='/home/fzh/Templates/JDET/csv_folder')
    parser.add_argument('--output_path', type=str, default='/home/fzh/Templates/JDET/demo')
    parser.add_argument('--nms_format', type=int, default=0, help='0: naive-nms, 1: class-nms')
    parser.add_argument('--nms_iou_thr', type=float, default=0.1)
    args = parser.parse_args()
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    main(args)

