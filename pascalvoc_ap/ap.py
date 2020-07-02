from .libs import *


__all__ = ['get_ap']


def get_ap(preds_all, gts_all, classes, iou_thr=0.5):
    """Get PascalVOC AP Function

    Args:
        preds_all (list): list of [class_name, confidence, left, top, right, bottom, image_id]
        gts_all (list): list of [class_name, left, top, right, bottom, image_id]
        classes (list): class name list you want to get APs. It can be str if you want get one class.
        iou_thr (float): IoU Threshold (Default: 0.5)

    Returns:
        dict: {'class_name': AP, 'class_name2': AP2, . . ., 'mAP': mAP}
    """
    cls_ap = dict()
    if isinstance(classes, str):
        classes = [classes]
    
    for cls in classes:
        preds = get_preds(preds_all=preds_all, class_name=cls)
        gts_map_with_img, gts_len = get_gts_map_with_img_and_len(gts_all=gts_all, class_name=cls)

        ious, used_vals = get_all_ious_and_used_vals(preds=preds, gts_map_with_img=gts_map_with_img, iou_thr=iou_thr)
        precisions, recalls = get_pr(ious=ious, used_vals=used_vals, gt_len=gts_len, iou_thr=iou_thr)
        ap, precisions, recalls = get_voc_ap_with_pr(precisions, recalls)
        cls_ap[cls] = ap

    cls_ap['mAP'] = sum(cls_ap.values()) / len(classes)
    return cls_ap
