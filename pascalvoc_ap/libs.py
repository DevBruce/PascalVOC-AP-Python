from collections import defaultdict


__all__ = [
    'get_preds', 'get_gts_map_with_img_and_len',
    'get_iou', 'get_all_ious_and_used_vals',
    'get_pr', 'get_voc_ap_with_pr',
]


def get_preds(preds_all, class_name):
    """
    Args:
        preds_all (list): list of [class_name, confidence, left, top, right, bottom, image_id]
        class_name (str): class name

    Returns:
        list: list of [confidence, left, top, right, bottom, image_id]
    """
    preds = list()
    for pred in preds_all:
        cls_name, confidence, *pts, img_id = pred
        if cls_name == class_name:
            preds.append([confidence, *pts, img_id])
    
    # Descending Ordering with Confidence
    preds = sorted(preds, key=lambda x : x[0], reverse=True)
    return preds


def get_gts_map_with_img_and_len(gts_all, class_name):
    """
    Args:
        gts_all (list): list of [class_name, left, top, right, bottom, image_id]
        class_name (str): class name

    Returns:
        tuple: (gts_map_with_img, gts_len)
    """
    gts_map_with_img = defaultdict(list)
    gts_len = 0
    for gt in gts_all:
        cls_name, *pts, img_id = gt
        if cls_name == class_name:
            gt_info = {'pts': pts, 'used': False}
            gts_map_with_img[img_id].append(gt_info)
            gts_len += 1
    return gts_map_with_img, gts_len


def get_iou(box1_pts, box2_pts):
    """Get IoU of box1 and box2

    Args:
        box1_pts (list): [left, top, right, bottom] (xyrb)
        box2_pts (list): [left, top, right, bottom] (xyrb)

    Returns:
        float: IoU of box1 and box2
    """
    box_intersection_pts = [
        max(box1_pts[0], box2_pts[0]),
        max(box1_pts[1], box2_pts[1]),
        min(box1_pts[2], box2_pts[2]),
        min(box1_pts[3], box2_pts[3]),
        ]
    intersection_width = box_intersection_pts[2] - box_intersection_pts[0] + 1
    intersection_height = box_intersection_pts[3] - box_intersection_pts[1] + 1
    intersection_area = intersection_width * intersection_height
    
    if intersection_width > 0 and intersection_height > 0:
        box1_width = box1_pts[2] - box1_pts[0] + 1
        box1_height = box1_pts[3] - box1_pts[1] + 1
        box2_width = box2_pts[2] - box2_pts[0] + 1
        box2_height = box2_pts[3] - box2_pts[1] + 1
        
        box1_area = box1_width * box1_height
        box2_area = box2_width * box2_height
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area
    else:
        iou = 0.0
    return iou


def get_all_ious_and_used_vals(preds, gts_map_with_img, iou_thr):
    """
    Args:
        preds (list): list of [confidence, left, top, right, bottom, image_id]. It can be get from get_preds function
        gts_map_with_img (dict): Key: image_id, Value: list of {'pts': [left, top, right, bottom], 'used': False}. It can be get from get_gts_map_with_img_and_len function
        iou_thr (float): Iou Threshold

    Returns:
        tuple: (ious, used_vals)
    """
    ious = list()
    used_vals = list()
    for pred in preds:
        _, *pred_pts, img_id = pred

        current_ious = list()
        gt_info_list = gts_map_with_img[img_id]
        for gt_info in gt_info_list:
            gt_pts, used = gt_info.get('pts'), gt_info.get('used')
            iou = get_iou(pred_pts, gt_pts)
            current_ious.append(iou)

        if current_ious:
            iou = max(current_ious)
            mapped_gt_idx = current_ious.index(iou)
            used = gt_info_list[mapped_gt_idx].get('used')
            if iou >= iou_thr and not used:
                gt_info_list[mapped_gt_idx]['used'] = True
        else:
            iou, used = 0.0, False

        ious.append(iou)
        used_vals.append(used)
    return ious, used_vals


def get_pr(ious, used_vals, gt_len, iou_thr):
    """
    Args:
        ious (list): IoU (float) list sorted by confidence. It can be get from get_all_ious_and_used_vals function
        used_vals (list): used_val (bool) list sorted by confidence. It can be get from get_all_ious_and_used_vals function
        gt_len (int): Number of gt boxes
        iou_thr (float): IoU Threshold

    Returns:
        tuple: (precisions, recalls)
    """
    TP = 0
    FP = 0
    precisions = list()
    recalls = list()

    for iou, used in zip(ious, used_vals):
        if iou >= iou_thr and not used:
            TP += 1
        else:
            FP += 1
            
        precision = TP / (TP + FP)
        recall = TP / gt_len
        precisions.append(precision)
        recalls.append(recall)
        
    return precisions, recalls


def get_voc_ap_with_pr(precisions, recalls):
    """
    Args:
        precisions (list): precisions from get_pr function
        recalls (list): recalls from get_pr function

    Returns:
        tuple: (ap, precisions, recalls)
    """
    precisions, recalls = precisions.copy(), recalls.copy()
    precisions.insert(0, 0.0)
    precisions.append(0.0)
    recalls.insert(0, 0.0)
    recalls.append(1.0)

    # Interpolation
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])

    recalls_changed_idx = list()
    for idx in range(1, len(recalls)):
        if recalls[idx] != recalls[idx-1]:
            recalls_changed_idx.append(idx)

    ap = 0.0
    for idx in recalls_changed_idx:
        recalls_diff = recalls[idx]-recalls[idx-1]
        ap += (recalls_diff * precisions[idx])
    return ap, precisions, recalls
