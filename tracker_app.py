from __future__ import print_function
import os.path
import numpy as np
import time
from tqdm import tqdm

# new imports
from libs import visualization
from libs import for_tracker
from libs import calculate_cost
from scipy.optimize import linear_sum_assignment
from libs import calculate_fn_fp
from libs.basetrack import TrackState
import motmetrics as mm
from loguru import logger
import glob
from collections import OrderedDict
from pathlib import Path


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


if __name__ == '__main__':

    phase = 'train'
    dataset = 'MOT17'

    if dataset == 'MOT17':
        mot_path = 'C:/Users/mhnas/Courses/Thesis/MOT/MOT17'
        imgFolder = 'images/bytetracker_occ_new_mm_cm_op_sel_new_ex_0.3_0.2_0.3_new_test'
        outFolder = 'outputs/bytetracker_occ_new_mm_cm_op_sel_new_ex_0.3_0.2_0.3_new_test'
        if phase == 'train':
            sequences = ['MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP',
                         'MOT17-11-SDP', 'MOT17-13-SDP']
            # Parameters Set Selected for MOT17 train sequences:
            # score_limit = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            # score_limit_l = [0, 0, 0, 0, 0, 0, 0]
            score_limit = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
            score_limit_l = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
            cost_thr = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
            cost_thr_l = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
            min_conf = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
            oc_coef = [10, 10, 10, 10, 10, 10, 10]

        if phase == 'test':
            sequences = ['MOT17-01-SDP', 'MOT17-03-SDP', 'MOT17-06-SDP', 'MOT17-07-SDP', 'MOT17-08-SDP',
                         'MOT17-12-SDP', 'MOT17-14-SDP']
            # Parameters Set Selected for MOT17 test sequences:
            score_limit = [0.6, 0.7, 0.5, 0.6, 0.5, 0.7, 0.4]
            score_limit_l = [0.2, 0.4, 0.2, 0.2, 0.3, 0.3, 0.1]
            cost_thr = [0.95, 0.9, 0.95, 0.95, 0.95, 0.95, 0.95]
            cost_thr_l = [0.25, 0.4, 0.25, 0.4, 0.3, 0.2, 0.4]
            min_conf = [0.2, 0.15, 0.1, 0.1, 0.2, 0.15, 0.1]
            oc_coef = [10, 10, 20, 10, 20, 20, 10]

    if dataset == 'MOT20':
        mot_path = 'C:/Users/mhnas/Courses/Thesis/MOT/MOT20'
        imgFolder = 'images/bytetracker_occ_new_mm_cm_op_sel_new_ex_20_0.3_0.2_0.3_test'  # fs: fixed score/ sb: score based / op: optimized
        outFolder = 'outputs/bytetracker_occ_new_mm_cm_op_sel_new_ex_20_0.3_0.2_0.3_test'
        if phase == 'train':
            sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
            # Parameters Set Selected for MOT20 train sequences:
            score_limit = [0.6, 0.4, 0.6, 0.4]
            score_limit_l = [0.3, 0.3, 0.3, 0.3]
            cost_thr = [0.95, 0.9, 0.95, 0.9]
            cost_thr_l = [0.3, 0.4, 0.3, 0.4]
            min_conf = [0.1, 0.1, 0.1, 0.1]
            oc_coef = [20, 15, 20, 15]
        if phase == 'test':
            sequences = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08']
            # Parameters Set Selected for MOT20 train sequences:
            score_limit = [0.6, 0.4, 0.6, 0.4]
            score_limit_l = [0.3, 0.3, 0.3, 0.3]
            cost_thr = [0.95, 0.9, 0.95, 0.9]
            cost_thr_l = [0.3, 0.4, 0.3, 0.4]
            min_conf = [0.1, 0.1, 0.1, 0.1]
            oc_coef = [20, 15, 20, 15]

    total_time = 0.0
    total_frames = 0

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    occTot = 0
    extTot = 0
    fpTot = 0
    fnTot = 0
    fpSeq = 0
    fnSeq = 0
    evaluate_dets = False
    num_parts = 20
    dets_res_t = np.zeros((7, num_parts))
    dets_res_fp = np.zeros((7, num_parts))
    seq_num = -1

    for seq in sequences:
        seq_num += 1
        umt_ind = 200
        mot_tracker = for_tracker.FORTracker(dataset, det_thresh=score_limit[seq_num], det_thresh_l=score_limit_l[seq_num], cost_thr=cost_thr[seq_num], cost_thr_l=cost_thr_l[seq_num], min_conf=min_conf[seq_num], oc_coef=oc_coef[seq_num])
        common_path = ('%s/%s/%s' % (mot_path, phase, seq))
        seq_dets = np.loadtxt('%s/dety/det.txt' % common_path, delimiter=',')  # load detections
        seq_len = int(seq_dets[:, 0].max())
        frameFP = np.zeros(seq_len)
        frameFN = np.zeros(seq_len)
        if phase == 'train':
            seq_gts = np.loadtxt('%s/gt/gt.txt' % common_path, delimiter=',')  # load ground truth

        added = ''
        if evaluate_dets:
            added = '_dets'
        with open('%s/%s%s.txt' % (outFolder, seq, added), 'w') as out_file:
            print("\nProcessing %s." % seq)

            if visualization.DisplayState.display:
                if not os.path.exists('%s/%s' % (common_path, imgFolder)):
                    os.makedirs('%s/%s' % (common_path, imgFolder))

            for frame in tqdm(range(int(seq_dets[:, 0].max()))):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                gts = []
                gts_w_id = []   # ground truths with id
                if phase == 'train':
                    gts = seq_gts[np.logical_and(seq_gts[:, 0] == frame, seq_gts[:, 6] == 1), 2:7]
                    gts[:, 2:4] += gts[:, 0:2]  # convert [x1,y1,w,h] to [x1,y1,x2,y2]
                    gts_w_id = seq_gts[np.logical_and(seq_gts[:, 0] == frame, seq_gts[:, 6] == 1), 1:7]  # groundtruths with id
                    gts_w_id[:, 3:5] += gts_w_id[:, 1:3]  # convert [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                start_time = time.time()
                trackers_id, trackers_bbox = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                # find False Positives and False Negatives
                trks = trackers_bbox.copy()
                if len(trks) > 0:
                    trks[:, 2:4] += trks[:, 0:2]
                FN, FP = calculate_fn_fp.cal_fn_fp(trks, gts, 0)
                frameFP[frame - 1] = len(FP)
                frameFN[frame - 1] = len(FN)
                fpSeq += len(FP)
                fnSeq += len(FN)

                lost_tlbr = [track.tlbr for track in mot_tracker.stracks if track.state == TrackState.Lost]
                lost_tlbr = np.array(lost_tlbr)
                lost_id = [track.track_id for track in mot_tracker.stracks if track.state == TrackState.Lost]
                lost_id = np.array(lost_id)
                new_tlbr = [track.tlbr for track in mot_tracker.stracks if track.state == TrackState.New]
                new_tlbr = np.array(new_tlbr)
                new_id = [track.track_id for track in mot_tracker.stracks if track.state == TrackState.New]
                new_id = np.array(new_id)
                rep_tlbr = [track.tlbr for track in mot_tracker.stracks if (track.state == TrackState.Tracked or track.state == TrackState.Occluded)]
                rep_tlbr = np.array(rep_tlbr)
                rep_id = [track.track_id for track in mot_tracker.stracks if (track.state == TrackState.Tracked or track.state == TrackState.Occluded)]
                rep_id = np.array(rep_id)
                rep_iou = [track.IoU for track in mot_tracker.stracks if (track.state == TrackState.Tracked or track.state == TrackState.Occluded)]
                rep_iou = np.array(rep_iou)

                # ------------ find matched id from groundtruths ------------------
                not_removed_stracks = [track for track in mot_tracker.stracks if track.state != TrackState.Removed]
                not_removed_stracks_tlbr = [track.tlbr for track in not_removed_stracks]
                if len(not_removed_stracks) > 0 and len(gts_w_id) > 0:
                    iou_matrix_m = calculate_cost.cal_iou_matrix(gts_w_id[:, 1:], not_removed_stracks_tlbr)
                    # first column: ground truth indexes, second column: object indexes
                    matched_indices_m = linear_sum_assignment(-iou_matrix_m)
                    matched_indices_m = np.asarray(matched_indices_m)
                    matched_indices_m = np.transpose(matched_indices_m)

                    # filter out matched with low IOU
                    matches_m = []
                    for m in matched_indices_m:
                        if iou_matrix_m[m[0], m[1]] >= 0.45:
                            matches_m.append(m)

                # ----------------- match detections with groundtruthes
                if evaluate_dets:
                    dets_limited = np.empty([0, 5])
                    for det in dets:
                        if det[4] > score_limit[seq_num]:
                            dets_limited = np.r_[dets_limited, det.reshape(1, 5)]
                    dets = dets_limited
                    dets_xy = dets[:, 0:4].copy()
                    dets_xy[:, 2:4] += dets_xy[:, 0:2]  # convert [x1,y1,w,h] to [x1,y1,x2,y2]
                    gts[:, 2:4] += gts[:, 0:2]  # convert [x1,y1,w,h] to [x1,y1,x2,y2]
                    iou_matrix_d = calculate_cost.cal_iou(gts_w_id[:, 1:], dets_xy[:, 0:4])
                    # first column: ground truth indexes, second column: detection indexes
                    matched_indices_d = linear_sum_assignment(-iou_matrix_d)
                    matched_indices_d = np.asarray(matched_indices_d)
                    matched_indices_d = np.transpose(matched_indices_d)

                    # filter out matched with low IOU
                    matches_d = []
                    for m in matched_indices_d:
                        if iou_matrix_d[m[0], m[1]] >= 0.5:
                            matches_d.append(m)

                    # fill mIDs
                    if len(matches_d) > 0:
                        matches_d = np.asarray(matches_d)

                    for d, det in enumerate(dets):
                        ind = np.where(matches_d[:, 1] == d)
                        ind = np.asarray(ind)
                        score_ind = int(det[4] * num_parts)
                        if ind.size == 0:
                            dets_res_fp[seq_num, score_ind] += 1
                            umt_ind += 1
                        else:
                            g_ind = np.reshape(gts_w_id[matches_d[ind, 0], 0], 1)  # groundtruth ID
                            print('%d,%d,%d,%d,%d,%d,%.2f,-1,-1,-1' % (frame, g_ind, det[0], det[1], det[2], det[3], det[4]), file=out_file)
                            dets_res_t[seq_num, score_ind] += 1

                else:
                    # save trackers -> Frame number, ID, Left, Top, Width, Height
                    for (id, bbx) in zip(trackers_id, trackers_bbox):
                        print('%d,%d,%d,%d,%d,%d,1,-1,-1,-1' % (frame, id, bbx[0], bbx[1], bbx[2], bbx[3]),
                              file=out_file)
                    if trackers_bbox.size > 0:
                        if len(rep_tlbr) > 0:
                            rep = np.concatenate((rep_tlbr, rep_id.reshape((rep_id.size, 1))), axis=1)
                        else:
                            rep = np.empty([0, 5])
                        if len(lost_tlbr) > 0:
                            lost = np.concatenate((lost_tlbr, lost_id.reshape((lost_id.size, 1))), axis=1)
                        else:
                            lost = np.empty([0, 5])
                        if len(new_tlbr) > 0:
                            new = np.concatenate((new_tlbr, new_id.reshape((new_id.size, 1))), axis=1)
                        else:
                            new = np.empty([0, 5])
                        visualization.dispaly_details(common_path, imgFolder, frame, dets, gts_w_id, rep, FN, FP, lost, new, det_thr=score_limit[seq_num], det_thr_l=score_limit[seq_num]-score_limit_l[seq_num]) #, unmatched_trckr, unmatched_gts, mot_tracker.trackers) #np.concatenate((trackers_bbox, trackers_id.reshape((trackers_id.size, 1))), axis=1)

        print('Occluded Number: ', mot_tracker.occNum)
        print('Extended Number: ', mot_tracker.extNum)
        occTot += mot_tracker.occNum
        mot_tracker.occNum = 0
        extTot += mot_tracker.extNum
        mot_tracker.extNum = 0
        if visualization.DisplayState.display_fn or visualization.DisplayState.display_fp:
            print('False Positive: ', fpSeq, 'False Negative: ', fnSeq)
            fnTot += fnSeq
            fpTot += fpSeq
            fnSeq = 0
            fpSeq = 0
        # visualization.generate_video('%s/%s' % (common_path, imgFolder), '%s/%s/Video_%s.avi' % (common_path, imgFolder, seq), 10)

    # save detection evaluation results in file
    if evaluate_dets:
        with open('%s/dets_eval.txt' % (outFolder), 'w') as out_file:
            print("\nSaving detection evaluation results")
            print('TP Total: %d, FP Total: %d, Total det: %d' % (np.sum(dets_res_t), np.sum(dets_res_fp), np.sum(dets_res_t) + np.sum(dets_res_fp)), file=out_file)
            for s, seq in enumerate(sequences):
                print('TP Seq: %d, FP Seq: %d' % (np.sum(dets_res_t[s]), np.sum(dets_res_fp[s])), file=out_file)
                print('TP %d: ' % (s), file=out_file, end='')
                for n in range(num_parts):
                    print('%d,' % (dets_res_t[s, n]), file=out_file, end='')
                print('\nFP %d: ' % (s), file=out_file, end='')
                for n in range(num_parts):
                    print('%d,' % (dets_res_fp[s, n]), file=out_file, end='')
    # save run result in a file
    print('Total Occluded: ', occTot)
    print('Total Extended: ', extTot)
    print('False Positive Total: ', fpTot, 'False Negative Total: ', fnTot)
    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
    if visualization.DisplayState.display:
        print("Note: to get real runtime results run without the option: --display")

    # evaluate MOTA
    mm.lap.default_solver = 'lap'

    gtfiles = glob.glob(os.path.join(('%s/%s' % (mot_path, phase)), '*/gt/gt{}.txt'.format('')))
    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    tsfiles = [f for f in glob.glob(os.path.join(outFolder, '*.txt')) if
               (not os.path.basename(f).startswith('eval') and
                not os.path.basename(f).endswith('frame.txt') and
                not os.path.basename(f).endswith('error.txt') and
                not os.path.basename(f).endswith('full.txt') and
                not os.path.basename(f).startswith('dets') and
                not os.path.basename(f).startswith('summery'))]
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)
    logger.info('Running metrics')
    metrics = mm.metrics.motchallenge_metrics
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    resStr = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(resStr)
    ''' print the last line, the overall results'''
    # print(resStr[resStr.rfind('\n'):])
    logger.info('Completed')
    with open('%s/summery_%s.txt' % (outFolder, phase), 'w') as sum_file:
        print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time), file=sum_file)
        print(resStr, file=sum_file)


