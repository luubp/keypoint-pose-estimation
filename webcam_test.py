import cv2
import torch
import argparse
import numpy as np

import colorsys
import random
from attributedict.collections import AttributeDict
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2

import network.generator.centernet_decode
import network.centernet

WINDOW_NAME = "CenterNet"

KEYPOINT_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

def random_colors(N):
    """ Random color generator.
    """
    N = N + 1
    hsv = [(i / N, 1.0, 1.0) for i in range(N)]
    colors = list(
        map(lambda c: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv)
    )
    random.shuffle(colors)
    return colors


def draw_rectangle(image, box, color, thickness=3):
    """ Draws a rectangle.
    Args:
        image: The image to draw on.
        box: A list of 4 elements (x1, y1, x2, y2).
        color: Rectangle color.
        thickness: Thickness of lines.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.
    Args:
        image: The image to draw on.
        box: A list of 4 elements (x1, y1, x2, y2).
        caption: String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
    )


def draw_circle(image, point):
    """ Draws a caption above the box in an image.
    Args:
        image: The image to draw on.
        point: A list of 4 elements (x, y).
    """
    cv2.circle(image, point, 7, (246, 250, 250), -1)
    cv2.circle(image, point, 2, (255, 209, 0), 2)


def draw_line(image, point1, point2):
    """ Draws a caption above the box in an image.
    Args:
        image: The image to draw on.
        point: A list of 4 elements (x, y).
    """
    cv2.line(image, point1, point2, (255, 209, 0), 5)

def get_output(model, image, cfg):
    
    with torch.no_grad():
        model.eval()
        model.to(cfg.MODEL.DEVICE)
        pred_dict = model((image, 1))
        torch.cuda.empty_cache()
        
        for gt in pred_dict:
            pred_dict[gt] = pred_dict[gt].to('cpu')

        decoded_prediction = network.generator.centernet_decode.PredictionDecoder.decode(pred_dict, cfg)
        #return decoded_prediction
        if len(decoded_prediction) == 0:
            return dict()
        boxes = decoded_prediction['boxes'][0]
        boxes_scores = decoded_prediction['boxes_scores']
        #top_val, top_ids = torch.topk(boxes_scores, 1)
        keypoints = decoded_prediction['keypoints']
        #keypoints_scores = decoded_prediction['keypoints_score']


        results = {
            'bbox': torch.stack([boxes]),
            'class_id': 0,
            'bbox_score':  boxes_scores[0],
            'keypoints': keypoints
        }

        return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videopath", help="File path of Videofile.", default="")
    parser.add_argument("--output", help="File path of result.", default="")
    args = parser.parse_args()

    #cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 512, 512)
    #cv2.moveWindow(WINDOW_NAME, 100, 200)


    y = yaml.safe_load(open('./config.yaml', 'r'))
    cfg = AttributeDict(y)
    img_transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=cfg.MODEL.PIXEL_MEAN, std=cfg.MODEL.PIXEL_STD),
                ToTensorV2()
                ])

    #test
    if args.videopath == "":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
    else:
        cap = cv2.VideoCapture(args.videopath)
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_writer = None
    if args.output != "":
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    #model
    model = network.centernet.CenterNet(cfg)
    checkpoint = torch.load('./weights/nt/FINALLYFIXEDcenternet_epoch8.pth')
    model.load_state_dict(checkpoint)

    while cap.isOpened():
        okay, frame = cap.read()
        if not okay:
            print('GG')
            break

        img = frame[:, :, ::-1]
        img = img.astype(np.float32)

        img = img_transform(image=img)['image']
        output = get_output(model, (img, ), cfg)

        if len(output) == 0:
            continue

        bbox = output['bbox'][0]
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin, xmax, ymax = int(xmin * 4), int(ymin * 4), int(xmax * 4), int(ymax * 4)
        draw_rectangle(frame, (xmin, ymin, xmax, ymax), (255, 0, 0))

        # keypoints = (output['keypoints'][0]).numpy().astype(np.int64)
        # keypoints_x = [int(i[1] * 4) for i in keypoints]
        # keypoints_y = [int(i[0] * 4) for i in keypoints]
        # for kp in zip(keypoints_x, keypoints_y):
        #     draw_circle(frame, (kp[1], kp[0]))
        # for keypoint_start, keypoint_end in KEYPOINT_EDGES:
        #     draw_line(frame,
        #               (keypoints_y[keypoint_start], keypoints_x[keypoint_start]),
        #               (keypoints_y[keypoint_end], keypoints_x[keypoint_end]),)

        if video_writer is not None:
            video_writer.write(frame)

        cv2.imshow(WINDOW_NAME, cv2.resize(frame, dsize=(512, 512)))
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()