from numba import njit
import numpy as np


@njit
def _xyxy2xywh(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1
    return bbox_xywh


@njit
def process_mmdet_results(mmdet_results):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 0 for human)
    :return: a list of detected bounding boxes
    """

    persons = mmdet_results
    bboxes = _xyxy2xywh(persons[:, :4])
    confs = persons[:, 4]
    ids = np.zeros(len(persons), dtype=np.int64)

    return bboxes, confs, ids


def process_track2pose(track_result):
    pass


def mmtracking_output(deepsort_output):
    # confidense = np.full(len(deepsort_output), 0.99)
    # deepsort_output = np.c_[deepsort_output, confidense]
    person_results = []
    for track_person in deepsort_output:
        person = {}
        person['track_id'] = int(track_person[-1])
        person['bbox'] = np.append(track_person[:-1], np.array([0.99]))
        person_results.append(person)

    return person_results
