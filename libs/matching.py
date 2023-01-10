import numpy as np
import lap
from .basetrack import TrackState
from . import calculate_cost


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def cal_cost_matrix(atracks, btracks):
    """
    Compute cost based on Normalized IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    niou_matrix = calculate_cost.cal_niou_matrix(atlbrs, btlbrs)
    cost_matrix = 1 - niou_matrix

    return cost_matrix


def multi_matching(detections, stracks, det_thr=0.5, cost_thr=0.95, cost_thr_l=0.3, ext_matching=False):
    if len(stracks) == 0:
        return np.empty([0, 2]), np.empty(0), range(len(detections)), np.empty([0, len(detections)]), 0
    if len(detections) == 0:
        return np.empty([0, 2]), range(len(stracks)), np.empty(0), np.empty([len(stracks), 0]), 0

    dists = cal_cost_matrix(stracks, detections)

    dists_det = dists.copy()
    for i, det in enumerate(detections):
        if det.score < det_thr:
            dists_det[:, i] = 1

    matches, u_trk, u_det = linear_assignment(dists_det, cost_thr)

    dists_det = dists.copy()
    for i, det in enumerate(detections):
        if det.score >= det_thr:
            dists_det[:, i] = 1

    for m in matches:
        dists_det[m[0], :] = 1
        dists_det[:, m[1]] = 1
    matches_l, u_trk_l, u_det_l = linear_assignment(dists_det, cost_thr_l)
    if len(matches_l) > 0:
        if len(matches) > 0:
            matches = np.concatenate((matches, matches_l), axis=0)
        else:
            matches = matches_l

    if len(matches) > 0:
        u_track = np.setdiff1d(range(len(stracks)), matches[:, 0])
        u_detection = np.setdiff1d(range(len(detections)), matches[:, 1])
    else:
        u_track = range(len(stracks))
        u_detection = range(len(detections))
    extNum = 0

    # Extended Matching
    if ext_matching:
        if len(u_track) > 0 and len(u_detection) > 0:
            iou_matrix_ext = np.zeros((len(u_track), len(u_detection)), dtype=np.float32)
            for d, ud in enumerate(u_detection):
                det = detections[ud]
                if det.score > 0.1:
                    for t, ut in enumerate(u_track):
                        trk = stracks[ut]
                        if trk.state != TrackState.New:
                            iou_matrix_ext[t, d] = calculate_cost.iou_ext_sep(det.tlbr, trk.tlbr, 0.3, 0.2)
            cost_ext = 1 - iou_matrix_ext
            matches_ext, u_trk_ext, u_det_ext = linear_assignment(cost_ext, 0.3)
            if len(matches_ext) > 0:
                if len(matches) > 0:
                    for i, m in enumerate(matches_ext):
                        match = np.array([[u_track[m[0]], u_detection[m[1]]]])
                        matches = np.concatenate((matches, match), axis=0)
                        extNum += 1
                else:
                    matches = matches_ext
                u_track = np.setdiff1d(range(len(stracks)), matches[:, 0])
                u_detection = np.setdiff1d(range(len(detections)), matches[:, 1])

    return matches, u_track, u_detection, dists, extNum
