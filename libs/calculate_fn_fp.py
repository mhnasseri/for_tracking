import numpy as np
from scipy.optimize import linear_sum_assignment
from libs import visualization
from libs import calculate_cost



def cal_fn_fp(trks, gts, index):
    """
  Computes false positives and false negatives
  index should be 0 or 1 depending the columns in gts and trks
  """
    FN, FP = [], []
    if visualization.DisplayState.display_fn or visualization.DisplayState.display_fp:
        if len(trks) > 0:
            iou_matrix_1 = calculate_cost.cal_iou(gts[:, index:], trks[:, index:])
            # first column: ground truth indexes, second column: object indexes
            matched_indices_1 = linear_sum_assignment(-iou_matrix_1)
            matched_indices_1 = np.asarray(matched_indices_1)
            matched_indices_1 = np.transpose(matched_indices_1)

            # filter out matched with low IOU
            matches = []
            for m in matched_indices_1:
                if iou_matrix_1[m[0], m[1]] >= 0.45:
                    matches.append(m)
            if len(matches) > 0:
                matches = np.asarray(matches)
                fnIndx = ~np.isin(range(len(gts)), matches[:, 0])
                fpIndx = ~np.isin(range(len(trks)), matches[:, 1])
                FP = trks[fpIndx]
                FN = gts[fnIndx]
    return FN, FP
