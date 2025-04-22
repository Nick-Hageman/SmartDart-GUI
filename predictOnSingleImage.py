import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import cv2
import numpy as np
from dataset.annotate import draw, get_dart_scores


def bboxes_to_xy(bboxes, max_darts=3):
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    for cls in range(5):
        if cls == 0:
            dart_xys = bboxes[bboxes[:, 4] == 0, :2][:max_darts]
            xy[4:4 + len(dart_xys), :2] = dart_xys
        else:
            cal = bboxes[bboxes[:, 4] == cls, :2]
            if len(cal):
                xy[cls - 1, :2] = cal[0]
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
    if np.sum(xy[:4, -1]) == 4:
        return xy
    else:
        xy = est_cal_pts(xy)
    return xy


def est_cal_pts(xy):
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
                xy[0, 2] = 1
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
                xy[1, 2] = 1
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
                xy[2, 2] = 1
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
                xy[3, 2] = 1
            xy[:, :2] += center
    else:
        print('Missed more than 1 calibration point')
    return xy


def predict_single_image(yolo, cfg, image_path):
    # Read and process image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get model predictions
    bboxes = yolo.predict(img)
    predictions = bboxes_to_xy(bboxes)
    
    # Calculate dart scores
    scores = get_dart_scores(predictions[:, :2], cfg, numeric=True)
    total_score = sum(scores)
    
    # Draw predictions on image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    annotated_img = draw(img, predictions[predictions[:, -1] == 1, :2], cfg, circles=False, score=True)
    
    # Save annotated image
    cv2.imwrite('test_prediction.jpg', annotated_img)
    
    return predictions, scores, total_score


if __name__ == '__main__':
    from train import build_model
    
    # Setup configuration
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', 'deepdarts_utrecht.yaml'))
    cfg.model.name = 'deepdarts_utrecht'

    # Build and load model
    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', cfg.model.name, 'weights'), cfg.model.weights_type)

    # Process single image
    image_path = 'picam2Test.jpg'
    try:
        predictions, dart_scores, total_score = predict_single_image(yolo, cfg, image_path)
        
        print("\nPredictions:")
        print("Calibration points (x, y):")
        for i, point in enumerate(predictions[:4]):
            if point[2] == 1:  # if point is visible
                print(f"Point {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
        
        print("\nDart locations (x, y):")
        for i, point in enumerate(predictions[4:]):
            if point[2] == 1:  # if dart is detected
                print(f"Dart {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
        
        print("\nDart Scores:", [int(score) for score in dart_scores if score > 0])
        print("Total Score:", int(total_score))
        print("\nAnnotated image saved as 'test_prediction.jpg'")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")