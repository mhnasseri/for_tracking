from numba import jit
import numpy as np


@jit
def iou(bb_det, bb_trk):
    """
  Computes IOU (Intersection Over Union) between two bounding boxes in the form [x1,y1,x2,y2]
  """
    xx1 = np.maximum(bb_det[0], bb_trk[0])
    xx2 = np.minimum(bb_det[2], bb_trk[2])
    w = np.maximum(0., xx2 - xx1)
    if w == 0:
        return 0
    yy1 = np.maximum(bb_det[1], bb_trk[1])
    yy2 = np.minimum(bb_det[3], bb_trk[3])
    h = np.maximum(0., yy2 - yy1)
    if h == 0:
        return 0
    wh = w * h
    area_det = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
    area_trk = (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1])
    o = wh / (area_det + area_trk - wh)
    return o


def cal_iou_matrix(detections, trackers):
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    return iou_matrix


@jit
def niou(bb_det, bb_trk):
    iou_first = iou(bb_det, bb_trk)
    iou_final = iou_first
    if iou_first > 0:
        w_det = bb_det[2] - bb_det[0]
        h_det = bb_det[3] - bb_det[1]
        x_det = bb_det[0] + w_det/2
        y_det = bb_det[1] + h_det/2

        w_trk = bb_trk[2] - bb_trk[0]
        h_trk = bb_trk[3] - bb_trk[1]
        x_trk = bb_trk[0] + w_trk / 2
        y_trk = bb_trk[1] + h_trk / 2

        x_coef = np.absolute(x_det - x_trk) / w_det
        y_coef = np.absolute(y_det - y_trk) / h_det
        w_coef = np.absolute(w_det - w_trk) / w_det
        h_coef = np.absolute(h_det - h_trk) / h_det

        iou_final = iou_first - (x_coef + y_coef + w_coef + h_coef)/4
        # iou_final = iou_first - max([x_coef, y_coef, w_coef, h_coef])
        iou_final = max(0, iou_final)
    return iou_final


def cal_niou_matrix(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    niou_m = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if niou_m.size == 0:
        return niou_m

    for i in range(len(atlbrs)):
        for j in range(len(btlbrs)):
            niou_m[i, j] = niou(atlbrs[i], btlbrs[j])

    return niou_m


def giou(bb_det, bb_trk):
    """
  Computes IOU (Intersection Over Union) between two bounding boxes in the form [x1,y1,x2,y2]
  """
    xx1 = np.maximum(bb_det[0], bb_trk[0])
    xx2 = np.minimum(bb_det[2], bb_trk[2])
    w = np.maximum(0., xx2 - xx1)
    yy1 = np.maximum(bb_det[1], bb_trk[1])
    yy2 = np.minimum(bb_det[3], bb_trk[3])
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    xx3 = np.minimum(bb_det[0], bb_trk[0])
    xx4 = np.maximum(bb_det[2], bb_trk[2])
    wc = xx4-xx3
    yy3 = np.minimum(bb_det[1], bb_trk[1])
    yy4 = np.maximum(bb_det[3], bb_trk[3])
    hc = yy4-yy3
    area_c = wc*hc

    area_det = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
    area_trk = (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1])
    union = area_det + area_trk - wh
    iou = wh / (union)

    g_iou = iou - (area_c - union)/area_c
    g_iou = max(0, g_iou)
    return g_iou


def cal_giou_matrix(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    giou_m = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if giou_m.size == 0:
        return giou_m

    for i in range(len(atlbrs)):
        for j in range(len(btlbrs)):
            giou_m[i, j] = giou(atlbrs[i], btlbrs[j])

    return giou_m


#@jit
def iou_ext(bb_det, bb_trk, ext):
    """
    Computes extended IOU (Intersection Over Union) between two bounding boxes in the form [x1,y1,x2,y2]
    """
    trk_w = bb_trk[2] - bb_trk[0]
    trk_h = bb_trk[3] - bb_trk[1]
    xx1 = np.maximum(bb_det[0], bb_trk[0] - trk_w*ext/2)
    xx2 = np.minimum(bb_det[2], bb_trk[2] + trk_w*ext/2)
    w = np.maximum(0., xx2 - xx1)
    if w == 0:
        return 0
    yy1 = np.maximum(bb_det[1], bb_trk[1] - trk_h*ext/2)
    yy2 = np.minimum(bb_det[3], bb_trk[3] + trk_h*ext/2)
    h = np.maximum(0., yy2 - yy1)
    if h == 0:
        return 0
    wh = w * h
    area_det = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
    area_trk = (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1])
    o = wh / (area_det + area_trk - wh)
    return o


def iou_ext_sep(bb_det, bb_trk, ext_w, ext_h):
    """
    Computes extended IOU (Intersection Over Union) between two bounding boxes in the form [x1,y1,x2,y2]
    with separate extension coefficient
    """
    iou_final = 0
    trk_w = bb_trk[2] - bb_trk[0]
    trk_h = bb_trk[3] - bb_trk[1]
    xx1 = np.maximum(bb_det[0], bb_trk[0] - trk_w*ext_w/2)
    xx2 = np.minimum(bb_det[2], bb_trk[2] + trk_w*ext_w/2)
    w = np.maximum(0., xx2 - xx1)
    if w == 0:
        return 0
    yy1 = np.maximum(bb_det[1], bb_trk[1] - trk_h*ext_h/2)
    yy2 = np.minimum(bb_det[3], bb_trk[3] + trk_h*ext_h/2)
    h = np.maximum(0., yy2 - yy1)
    if h == 0:
        return 0
    wh = w * h
    area_det = (bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1])
    area_trk = (bb_trk[2] - bb_trk[0]) * (bb_trk[3] - bb_trk[1])
    o = wh / (area_det + area_trk - wh)

    if o > 0:
        w_det = bb_det[2] - bb_det[0]
        h_det = bb_det[3] - bb_det[1]
        x_det = bb_det[0] + w_det/2
        y_det = bb_det[1] + h_det/2

        w_trk = bb_trk[2] - bb_trk[0]
        h_trk = bb_trk[3] - bb_trk[1]
        x_trk = bb_trk[0] + w_trk / 2
        y_trk = bb_trk[1] + h_trk / 2

        x_coef = np.absolute(x_det - x_trk) / w_det
        y_coef = np.absolute(y_det - y_trk) / h_det
        w_coef = np.absolute(w_det - w_trk) / w_det
        h_coef = np.absolute(h_det - h_trk) / h_det

        iou_final = max(0, o - (x_coef + y_coef + w_coef + h_coef)/4)
    return iou_final


@jit
def ios(bb_first, bb_second):
    """
  Computes IOS (Intersection Over Second Bounding Box) between two bounding boxes in the form [x1,y1,x2,y2]
  """
    xx1 = np.maximum(bb_first[0], bb_second[0])
    yy1 = np.maximum(bb_first[1], bb_second[1])
    xx2 = np.minimum(bb_first[2], bb_second[2])
    yy2 = np.minimum(bb_first[3], bb_second[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_second[2] - bb_second[0]) * (bb_second[3] - bb_second[1]))
    return o


def cal_ios_matrix(trackers):
    ios_matrix = np.zeros((len(trackers), len(trackers)), dtype=np.float32)
    for t1, trk1 in enumerate(trackers):
        for t2, trk2 in enumerate(trackers):
            if t2 != t1:
                ios_matrix[t1, t2] = ios(trk1, trk2)
    return ios_matrix


def dist(bb_det, bb_trk):
        """
      Computes distance between borders of two bounding boxes in the form [x1,y1,x2,y2]
      (the sum of nearest part in x direction and nearest part in y direction)
      """
        xx1 = np.maximum(bb_det[0], bb_trk[0])
        xx2 = np.minimum(bb_det[2], bb_trk[2])
        dx = np.maximum(0., xx1 - xx2)
        yy1 = np.maximum(bb_det[1], bb_trk[1])
        yy2 = np.minimum(bb_det[3], bb_trk[3])
        dy = np.maximum(0., yy1 - yy2)
        distance = dx + dy
        return distance


def cal_dist(detections, trackers):
    dist_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    dtlbrs = [det.tlbr for det in detections]
    ttlbrs = [track.tlbr for track in trackers]
    for d, det in enumerate(dtlbrs):
        for t, trk in enumerate(ttlbrs):
            dist_matrix[d, t] = dist(det, trk)
    return dist_matrix


@jit
def outside(trk, img_s):
    """
    Computes how many percent of trk is placed outside of img_s
    """
    out_x = 0
    out_y = 0
    if trk[0] < 0:
        out_x = -trk[0]
    if trk[2] > img_s[0]:
        out_x = trk[2] - img_s[0]
    if trk[1] < 0:
        out_y = -trk[1]
    if trk[3] > img_s[1]:
        out_y = trk[3] - img_s[1]
    out_a = out_x * (trk[3] - trk[1]) + out_y * (trk[2] - trk[0])
    area = (trk[3] - trk[1]) * (trk[2] - trk[0])
    return out_a / area


def cal_outside(trackers, img_s):
    out = np.zeros(len(trackers), dtype=np.float32)
    for t, trk in enumerate(trackers):
        out[t] = outside(trk, img_s)
    return out