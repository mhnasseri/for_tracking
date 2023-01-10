from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Occluded = 2
    Lost = 3
    Removed = 4


class BaseTrack(object):
    _count = 0

    track_id = 0
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_id():
        BaseTrack._count = 0

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def mark_occluded(self):
        self.state = TrackState.Occluded
