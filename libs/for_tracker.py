import numpy as np
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from . import calculate_cost
from . import matching


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.mean_next, self.mean_next_temp = None, None
        self.state = TrackState.New

        self.score = score
        self.tracklet_len = 0

        # New Variables
        self.age = 0                    # Total frames from first detection of tracker
        self.time_since_observed = 0    # Number of frames after last time tracker is matched
        self.frames = np.empty(0)       # Frames a target detected
        self.CPs = np.empty(0)          # The maximum percent that is covered by other targets
        self.t_s_not_covered = 0        # Time since the covered_percent is less than a threshold
        self.confTot = 0                # Total confidence of matched detections
        self.cov_per = 0                # covered percent
        self.cov_area_ratio = 0         # The area of tracker, divided by the area of covering tracker
        self.cov_by = 0                 # The index of covering tracker
        self.covering_state = 0         # The state of covering tracker
        self.IoU = 0                    # The IoU with the matched detection
        self.area_ratio = 0             # the ratio of target area to average area of all targets
        self.occOkFrame = 0             # The number of frame that it is possible to target become occluded
        self.occConNum = 0              # number of successive frames the coverd percent is incresed or remained in 1

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self.age += 1

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            ''' Calculate the prediction for current frame'''
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                stracks[i].age += 1

            ''' Calculate the prediction for the next frame'''
            multi_mean_next = multi_mean.copy()
            multi_covariance_next = multi_covariance.copy()
            multi_mean_next, multi_covariance_next = STrack.shared_kalman.multi_predict(multi_mean_next, multi_covariance_next)
            for i, (mean, cov) in enumerate(zip(multi_mean_next, multi_covariance_next)):
                stracks[i].mean_next_temp = mean
                if stracks[i].mean_next is None:
                    stracks[i].mean_next = mean


    def activate(self, kalman_filter, frame_id, MinConfTot):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.New
        if frame_id == 1:
            self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.confTot = self.score
        if self.confTot > MinConfTot:
            self.state = TrackState.Tracked
        self.fill_arrays()

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh), self.cov_per
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.t_s_not_covered = 0
        self.fill_arrays()

    def add_conf(self, new_track, frame_id, MinConfTot):
        # N_CP_TH = 0.7       # New Covered Percent Threshold
        self.frame_id = frame_id
        self.score = new_track.score
        self.confTot += new_track.score
        # if self.cov_per < N_CP_TH:
        #     self.confTot += new_track.score
        # else:
        #     self.confTot += new_track.score/2
        if self.confTot > MinConfTot:
            self.state = TrackState.Tracked
        self.fill_arrays()

    def sub_conf(self, frame_id):
        self.confTot -= 0.3
        if self.confTot < 0:
            self.state = TrackState.Removed
        else:
            self.frame_id = frame_id
            self.score = 0
            self.fill_arrays()

    def fill_arrays(self, frame_id=None):
        if frame_id is None:
            self.frames = np.concatenate((self.frames, np.array([self.frame_id])), axis=0)
        else:
            self.frames = np.concatenate((self.frames, np.array([frame_id])), axis=0)
        self.CPs = np.concatenate((self.CPs, np.array([self.cov_per])), axis=0)


    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh), self.cov_per)
        self.state = TrackState.Tracked

        self.score = new_track.score
        self.t_s_not_covered = 0
        self.fill_arrays()

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class FORTracker(object):
    def __init__(self, dataset='MOT17', track_thresh=0.1, track_buffer=30, match_thresh=0.9, mot20=False, frame_rate=30, det_thresh=0.5, det_thresh_l=0.2, cost_thr=0.95, cost_thr_l=0.3, min_conf=0.2, oc_coef=20):
        self.dataset = dataset
        self.stracks = []  # type: list[STrack]
        STrack.reset_id()
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.match_thresh = match_thresh
        self.mot20 = mot20
        self.kalman_filter = KalmanFilter()
        self.occNum = 0
        self.area_avg = 0
        self.frame_parameters = np.empty([0, 15])
        self.extNum = 0
        self.detThr = det_thresh
        self.detThrL = det_thresh_l
        self.costThr = cost_thr
        self.costThrL = cost_thr_l
        self.minConfT = min_conf
        self.ocCoef = oc_coef


    def update(self, output_results):  #, img_info, img_size):
        MinConfTot = self.detThr + self.minConfT
        CP_TH = 0.3
        AR_TH = 1.02
        self.frame_id += 1

        scores = output_results[:, 4]
        bboxes = output_results[:, :4]

        remain_inds = scores > (self.detThr - self.detThrL)

        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(tlwh, s) for (tlwh, s) in zip(dets, scores_keep)]
        else:
            detections = []

        # ''' calculate distance of detections to each other '''
        # dist_matrix = calculate_cost.cal_dist(detections, detections)
        # dist_sorted = dist_matrix.copy()
        # dist_sorted.sort(axis=0)
        # if dist_sorted.size > 4:
        #     dist_sorted = dist_sorted[1:4, :]
        #     for d, det in enumerate(detections):
        #         det.nearestDists = dist_sorted[:, d].T

        ''' Add newly detected tracklets to tracked_stracks'''
        tracked_stracks = []  # type: list[STrack]
        for track in self.stracks:
            if track.state == TrackState.New or track.state == TrackState.Tracked or track.state == TrackState.Lost or track.state == TrackState.Occluded:
                tracked_stracks.append(track)

        ''' Predict the current location with KF '''
        STrack.multi_predict(tracked_stracks)

        ''' Calculate the average area of targets in current frame '''
        area_sum = 0
        for t, strk in enumerate(tracked_stracks):
            area_sum = area_sum + strk.tlwh[2] * strk.tlwh[3]
        area_avg = 0
        if len(tracked_stracks) > 0:
            area_avg = area_sum / len(tracked_stracks)

        ''' Calculate Covered Percent and the ratio of area of covered target to covering target 
            and ratio of area of target to average area'''
        stracks_bbox = [trk.tlbr for trk in tracked_stracks]
        ios_matrix = calculate_cost.cal_ios_matrix(stracks_bbox)

        if len(ios_matrix) > 0:
            trks_cov_per = np.amax(ios_matrix, axis=0)      # tracks covered percent
            trks_cov_by = np.argmax(ios_matrix, axis=0)     # tracks covered by which other track
            for t in range(len(tracked_stracks)):
                tracked_stracks[t].cov_per = trks_cov_per[t]
                if tracked_stracks[t].cov_per > 0:
                    area_t = tracked_stracks[t].tlwh[2] * tracked_stracks[t].tlwh[3]
                    tracked_stracks[t].cov_by = trks_cov_by[t]
                    tracked_stracks[t].covering_state = tracked_stracks[tracked_stracks[t].cov_by].state
                    covering_ind = trks_cov_by[t]
                    area_covering = tracked_stracks[covering_ind].tlwh[2] * tracked_stracks[covering_ind].tlwh[3]
                    tracked_stracks[t].cov_area_ratio = area_covering / area_t
                    if area_avg > 0:
                        tracked_stracks[t].area_ratio = area_t/area_avg
                    else:
                        tracked_stracks[t].area_ratio = 0
                if len(tracked_stracks[t].CPs) > 0:
                    if trks_cov_per[t] > (tracked_stracks[t].CPs[-1] + 0.05) or trks_cov_per[t] == 1:
                        tracked_stracks[t].occConNum += 1
                        if tracked_stracks[t].occConNum > 3 and tracked_stracks[t].state == TrackState.Tracked:
                            tracked_stracks[t].occOkFrame = self.frame_id
                    else:
                        tracked_stracks[t].occConNum = 0

        ''' Update time since not covered according to trks_cov_per and cov_area_ratio'''
        for i in range(len(tracked_stracks)):
            if tracked_stracks[i].cov_per < CP_TH or tracked_stracks[i].cov_area_ratio < AR_TH or \
               tracked_stracks[i].covering_state == TrackState.Lost or \
               tracked_stracks[i].covering_state == TrackState.Occluded:
                tracked_stracks[i].t_s_not_covered += 1

        ''' Step 2: First association, with high score detection boxes'''
        matches0, u_track0, u_detection0, dists0, extNum0 = matching.multi_matching(detections, tracked_stracks, det_thr=self.detThr, cost_thr=self.costThr, cost_thr_l=self.costThrL)
        # Do not apply camera motion removal for MOT20 dataset, because it is negligible
        if self.dataset == 'MOT20':
            matches = matches0
            u_track = u_track0
            u_detection = u_detection0
            dists = dists0
        avgErr = np.array([0., 0., 0., 0.])
        avgIoU = 0
        trkNum = 0
        for itracked, idet in matches0:
            track = tracked_stracks[itracked]
            det = detections[idet]
            if track.state != TrackState.New:
                err = track.tlwh_to_xyah(det.tlwh) - track.mean[0:4]
                avgErr += err
                avgIoU += (1 - dists0[itracked, idet])
                trkNum += 1
        if trkNum > 0:
            avgErr = avgErr / trkNum
            avgIoU = avgIoU / trkNum
        sumErr = np.sum(np.absolute(avgErr))

        for i, trk in enumerate(tracked_stracks):
            trk.mean[0:2] += avgErr[0:2]
            trk.mean_next[0:2] += avgErr[0:2]

        avgErr1 = np.array([0., 0., 0., 0.])
        avgIoU1 = 0
        trkNum1 = 0
        sumErr1 = 0
        if self.dataset == 'MOT17':
            matches, u_track, u_detection, dists, extNum = matching.multi_matching(detections, tracked_stracks, det_thr=self.detThr, cost_thr=self.costThr, cost_thr_l=self.costThrL, ext_matching=True)
            self.extNum += extNum
            for itracked, idet in matches:
                track = tracked_stracks[itracked]
                det = detections[idet]
                if track.state != TrackState.New:
                    err = track.tlwh_to_xyah(det.tlwh) - track.mean[0:4]
                    avgErr1 += err
                    avgIoU1 += (1 - dists[itracked, idet])
                    trkNum1 += 1
            if trkNum1 > 0:
                avgErr1 = avgErr1 / trkNum1
                avgIoU1 = avgIoU1 / trkNum1
            sumErr1 = np.sum(np.absolute(avgErr1))

        # update next frame prediction
        for trk in tracked_stracks:
            trk.mean_next = trk.mean_next_temp

        param = np.append(self.frame_id, avgErr)
        param = np.append(param, avgErr1)
        param = np.append(param, [avgIoU, avgIoU1, len(u_track0), len(u_track), len(u_detection0), len(u_detection)])
        self.frame_parameters = np.r_[self.frame_parameters, param.reshape((1, 15))]

        # if applying camera motion increases the error, return to state before applying camera motion
        if sumErr1/(sumErr + 0.1) > 2:  # or avgIoU1 < avgIoU or len(u_track) < len(u_track0):
            for i, trk in enumerate(tracked_stracks):
                trk.mean[0:2] -= avgErr[0:2]
            matches = matches0
            u_track = u_track0
            u_detection = u_detection0

        for itracked, idet in matches:
            track = tracked_stracks[itracked]
            det = detections[idet]
            track.time_since_observed = 0
            track.IoU = 1-dists[itracked, idet]
            if track.state == TrackState.Tracked or track.state == TrackState.Occluded:
                track.update(detections[idet], self.frame_id)
            elif track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, new_id=False)
            elif track.state == TrackState.New:
                track.add_conf(detections[idet], self.frame_id, MinConfTot)
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)

        ''' Step 3: mark unmatched trackers as occluded or lost'''
        for it in u_track:
            track = tracked_stracks[it]
            track.time_since_observed += 1
            track.score = 0
            track.IoU = 0
            ut_area = track.tlwh[2] * track.tlwh[3]
            confidence = min(1, track.age / (track.time_since_observed * self.ocCoef) * (ut_area / area_avg))
            if track.state == TrackState.Tracked or track.state == TrackState.Occluded:
                if (tracked_stracks[it].cov_per > 0.7 and confidence > 0.9 and tracked_stracks[it].covering_state == TrackState.Tracked) or (self.frame_id - tracked_stracks[it].occOkFrame < 4):
                    track.mark_occluded()
                    track.frame_id = self.frame_id
                    track.fill_arrays()
                    self.occNum += 1
                else:
                    track.mark_lost()
                    track.fill_arrays(frame_id=self.frame_id)
            elif track.state == TrackState.New:
                track.sub_conf(self.frame_id)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if area_avg > 0:
                track.area_ratio = track.tlwh[2] * track.tlwh[3] / area_avg
            if track.score >= self.detThr:
                track.activate(self.kalman_filter, self.frame_id, MinConfTot)
                self.stracks.append(track)

        """ Step 5: Remove lost stracks"""
        for track in self.stracks:
            if track.state == TrackState.Lost:
                if ((track.t_s_not_covered > (np.minimum(3, track.age/10)) and track.time_since_observed > 2) or (track.age - track.time_since_observed) < track.time_since_observed):  # or track.time_since_observed > 10:
                    track.mark_removed()

        bbox_stracks = [track.tlwh for track in self.stracks if track.state == TrackState.Tracked or track.state == TrackState.Occluded]
        bbox_stracks = np.array(bbox_stracks)
        id_stracks = [track.track_id for track in self.stracks if track.state == TrackState.Tracked or track.state == TrackState.Occluded]
        id_stracks = np.array(id_stracks)

        return id_stracks, bbox_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
