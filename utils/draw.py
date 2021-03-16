import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
kpt_num = 17
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def draw_bbox_keypoints(img, person_results):
    identities = np.array([i['track_id'] for i in person_results])
    bboxes = np.array([i['bbox'][:4] for i in person_results])
    keypoints = np.array([i['keypoints'][:, :2] for i in person_results])
    img = _draw_bbox_keypoints(img, bboxes, keypoints, identities)

    return img


def _draw_bbox_keypoints(img, bboxes, keypoints, identities):
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = np.int32(bboxes[i])
        id = int(identities[i])
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        img = draw_keypoints(img, keypoints[i], color)
    return img


def draw_keypoints(img, keypoint, color):
    for skelet in skeleton:
        cv2.line(img, tuple(keypoint[skelet[0] - 1]), tuple(keypoint[skelet[1] - 1]), color, 2)
    return img


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
