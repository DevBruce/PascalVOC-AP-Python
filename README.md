# PascalVOC AP with Python

> Ref: <http://host.robots.ox.ac.uk/pascal/VOC/>

<br><br>

## Install

```bash
$ pip install pascalvoc-ap
```

<br><br>

## How to Use

```python
from pascalvoc_ap.ap import get_ap


# (some codes)...
ap_results = get_ap(preds_all=preds_all, gts_all=gts_all, classes=classes, iou_thr=0.5)
```

- `preds_all`: (list): list of `[class_name, confidence, left, top, right, bottom, image_id]`
- `gts_all`: (list): list of `[class_name, left, top, right, bottom, image_id]`
- `classes`: (list or str): Class name list you want to get APs. It can be str if you want get one class.
- `iou_thr`: (float): IoU Threshold (Default: 0.5)

<br>

### About return value

Return data type is dict.  
Key is class name, Value is AP.  
You can get AP like below.  

```python
car_AP = ap_results.get('car')
mAP = ap_results.get('mAP')
```
