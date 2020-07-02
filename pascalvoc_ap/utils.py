import json


__all__ = ['get_gts_all_from_coco']


def get_gts_all_from_coco(coco_gt_path):
    coco_gt = json.load(open(coco_gt_path))

    cat_infos = coco_gt['categories']
    img_infos = coco_gt['images']
    annos = coco_gt['annotations']

    cls_map = dict()
    for cat_info in cat_infos:
        cls_num = cat_info['id']
        cls_name = cat_info['name']
        cls_map[cls_num] = cls_name

    img_map = dict()
    for img_info in img_infos:
        img_id = img_info['id']
        img_fname = img_info['file_name']
        img_map[img_id] = img_fname

    gts_all = list()
    for anno in annos:
        img_id = anno['image_id']
        img_fname = img_map[img_id]

        cat_id = anno['category_id']
        cls_name = cls_map[cat_id]

        x, y, w, h = anno.get('bbox')
        left, top, right, bottom = x, y, x+w, y+h

        gts_all.append([cls_name, left, top, right, bottom, img_fname])

    return gts_all
