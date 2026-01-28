# ------------------------------------------------------------------------
# Code to calculate mAP and mRecall for action detection. Modified from official EPIC-Kitchens action detection evaluation code.
# See: https://github.com/epic-kitchens/C2-Action-Detection/blob/master/EvaluationCode/evaluate_detection_json_ek100.py 
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

import os
import json
import polars as pl
import numpy as np
from joblib import Parallel, delayed
from typing import List
from typing import Tuple
from typing import Dict


def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate / very short annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e, l = event['segment'][0], event['segment'][1], event['label_id']
        if (e - s) >= tol:
            valid = True
        else:
            valid = False
        for p_event in valid_events:
            if ((abs(s-p_event['segment'][0]) <= tol)
                and (abs(e-p_event['segment'][1]) <= tol)
                and (l == p_event['label_id'])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events


def load_gt_seg_from_json(json_anno, split=None, label='label_id', label_offset=0):
    # load json file
    with open(json_anno, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels = [], [], [], []
    for k, v in json_db.items():

        # filter based on split
        if (split is not None) and v['subset'].lower() != split:
            continue
        # remove duplicated instances
        ants = remove_duplicate_annotations(v['annotations'])
        # video id
        vids += [k] * len(ants)
        # for each event, grab the start/end time and label
        for event in ants:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]

    # move to pl dataframe
    gt_base = pl.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels
    })

    return gt_base


def load_pred_seg_from_json(json_anno, label='label_id', label_offset=0):
    # load json file
    with open(json_anno, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels, scores = [], [], [], [], []
    for k, v, in json_db.items():
        # video id
        vids += [k] * len(v)
        # for each event
        for event in v:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]
            scores += [float(event['scores'])]

    # move to pl dataframe
    pred_base = pl.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels,
        'score': scores
    })

    return pred_base


class ANETdetection(object):
    """Adapted from https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_detection.py"""

    def __init__(
        self,
        ant_file,
        split=None,
        tiou_thresholds=np.linspace(0.1, 0.5, 5),
        top_k=(1, 5),
        label='label_id',
        label_offset=0,
        num_workers=8,
        dataset_name=None,
    ):

        self.tiou_thresholds = tiou_thresholds
        self.top_k = top_k
        self.ap = None
        self.num_workers = num_workers
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = os.path.basename(ant_file).replace('.json', '')

        # Import ground truth and predictions
        self.split = split
        self.ground_truth = load_gt_seg_from_json(
            ant_file, split=self.split, label=label, label_offset=label_offset)

        # remove labels that does not exists in gt
        self.activity_index = {j: i for i, j in enumerate(sorted(self.ground_truth['label'].unique().to_list()))}
        self.ground_truth = self.ground_truth.with_columns(pl.col('label').replace(self.activity_index))

    def _get_predictions_with_label(self, preds, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        res = preds.filter(pl.col('label') == cidx)
        if res.is_empty():
            return pl.DataFrame()
        return res

    def wrapper_compute_average_precision(self, preds):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_average_precision_detection)(
                ground_truth=self.ground_truth.filter(pl.col('label') == cidx),
                prediction=self._get_predictions_with_label(preds, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def wrapper_compute_topkx_recall(self, preds):
        """Computes Top-kx recall for each class in the subset.
        """
        recall = np.zeros((len(self.tiou_thresholds), len(self.top_k), len(self.activity_index)))

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_topkx_recall_detection)(
                ground_truth=self.ground_truth.filter(pl.col('label') == cidx),
                prediction=self._get_predictions_with_label(preds, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
                top_k=self.top_k,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            recall[...,cidx] = results[i]

        return recall

    def evaluate(self, preds):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        preds can be (1) a pl.DataFrame; or (2) a json file where the data will be loaded;
        or (3) a python dict item with numpy arrays as the values
        """

        if isinstance(preds, pl.DataFrame):
            assert 'label' in preds.columns
        elif isinstance(preds, str) and os.path.isfile(preds):
            preds = load_pred_seg_from_json(preds)
        elif isinstance(preds, Dict):
            # move to pl dataframe
            # did not check dtype here, can accept both numpy / pytorch tensors
            try:
                preds = pl.DataFrame({
                'video-id' : preds['video-id'],
                't-start' : preds['t-start'].tolist(),
                't-end': preds['t-end'].tolist(),
                'label': preds['label'].tolist(),
                'score': preds['score'].tolist()
                })
            except:
                preds = pl.DataFrame(schema={'video-id': pl.Utf8, 't-start': pl.Float64, 't-end': pl.Float64, 'label': pl.Int64, 'score': pl.Float64})

        # always reset ap
        self.ap = None

        # make the label ids consistent
        preds = preds.with_columns(pl.col('label').replace(self.activity_index))

        # compute mAP
        self.ap = self.wrapper_compute_average_precision(preds)
        self.recall = self.wrapper_compute_topkx_recall(preds)
        mAP = self.ap.mean(axis=1)
        mRecall = self.recall.mean(axis=2)

        # return the results
        return mAP, mRecall


def compute_average_precision_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.is_empty():
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].to_numpy().argsort()[::-1]
    prediction = prediction[sort_idx.tolist()]

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Add original row index to ground truth for lock_gt tracking
    ground_truth = ground_truth.with_row_index("_orig_idx")

    # Assigning true positive to truly ground truth instances.
    for idx in range(len(prediction)):
        pred_row = prediction.row(idx, named=True)

        # Check if there is at least one ground truth in the video associated.
        ground_truth_videoid = ground_truth.filter(pl.col('video-id') == pred_row['video-id'])
        if ground_truth_videoid.is_empty():
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid
        pred_segment = np.array([pred_row['t-start'], pred_row['t-end']])
        gt_segments = this_gt.select(['t-start', 't-end']).to_numpy()
        tiou_arr = segment_iou(pred_segment, gt_segments)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                orig_gt_idx = this_gt['_orig_idx'][jdx]
                if lock_gt[tidx, orig_gt_idx] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, orig_gt_idx] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap


def compute_topkx_recall_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5),
    top_k=(1, 5),
):
    """Compute recall (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    top_k: tuple, optional
        Top-kx results of a action category where x stands for the number of 
        instances for the action category in the video.
    Outputs
    -------
    recall : float
        Recall score.
    """
    if prediction.is_empty():
        return np.zeros((len(tiou_thresholds), len(top_k)))

    # Initialize true positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(top_k)))
    n_gts = 0

    # Get unique video ids from ground truth
    gt_video_ids = ground_truth['video-id'].unique().to_list()

    for videoid in gt_video_ids:
        ground_truth_videoid = ground_truth.filter(pl.col('video-id') == videoid)
        n_gts += len(ground_truth_videoid)
        prediction_videoid = prediction.filter(pl.col('video-id') == videoid)
        if prediction_videoid.is_empty():
            continue

        this_gt = ground_truth_videoid
        this_pred = prediction_videoid

        # Sort predictions by decreasing score order.
        score_sort_idx = this_pred['score'].to_numpy().argsort()[::-1]
        top_kx_idx = score_sort_idx[:max(top_k) * len(this_gt)]
        tiou_arr = k_segment_iou(this_pred.select(['t-start', 't-end']).to_numpy()[top_kx_idx],
                                 this_gt.select(['t-start', 't-end']).to_numpy())
            
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for kidx, k in enumerate(top_k):
                tiou = tiou_arr[:k * len(this_gt)]
                tp[tidx, kidx] += ((tiou >= tiou_thr).sum(axis=0) > 0).sum()

    recall = tp / n_gts

    return recall


def k_segment_iou(target_segments, candidate_segments):
    return np.stack(
        [segment_iou(target_segment, candidate_segments) \
            for target_segment in target_segments]
    )


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap
