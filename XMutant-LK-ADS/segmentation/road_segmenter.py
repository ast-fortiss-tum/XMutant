# knowledge based semantic segmentation for udacity simulation
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Segment:
    @staticmethod
    def color_mask(img, color):
        # print(img[159:,25:60,])
        # print(img[0])
        img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)

        if color in ["yellow", "y"]:
            lb = np.array([10, 100, 100])
            ub = np.array([25, 255, 255])
            mask = cv2.inRange(img_hsv, lb, ub)
            mask_blur = cv2.medianBlur(mask, 5)

        elif color in ["white", "w"]:
            lb = np.array([0, 0, 200])
            ub = np.array([180, 50, 255])
            mask = cv2.inRange(img_hsv, lb, ub)
            mask_blur = cv2.medianBlur(mask, 5)

        elif color in ["green", "g"]:
            lb = np.array([26, 50, 50])
            ub = np.array([38, 200, 175])
            mask = cv2.inRange(img_hsv, lb, ub)
            mask_blur = cv2.medianBlur(mask, 15)

        # result = cv2.bitwise_and(img, img_hsv, mask=mask)
        return mask_blur

    @staticmethod
    def right_lane(yellow_mask, green_mask):
        labels = ["grass_left", "lane_left", "lane_right", "grass_right"]
        road_mask = 255 - green_mask
        right_lane_mask = road_mask.copy()

        # if side in ["r", "right"]:
        #    pass
        # elif side in ["l", "left"]:

        last = last_last = None

        for row_id in range(yellow_mask.shape[0]):

            r_row = road_mask[row_id, :]
            y_row = yellow_mask[row_id, :]

            mid_line_indices = np.nonzero(y_row)[0]

            if len(mid_line_indices) == 0 and last_last is None:
                # No yellow line
                road_indices = np.nonzero(r_row)[0]
                if len(road_indices) == 0:
                    middle_index = -1
                elif r_row[-1] > 0 and r_row[-2] > 0:
                    middle_index = 256

                elif r_row[0] > 0 and r_row[1] > 0:
                    # yellow line is on the left
                    middle_index = -1
                else:

                    middle_index = road_indices[len(road_indices) // 2]
            else:
                if len(mid_line_indices) == 0 and last_last is not None:
                    middle_index = 2 * last - last_last
                else:
                    middle_index = mid_line_indices[len(mid_line_indices) // 2]

                last_last = last if last is not None else last_last
                last = middle_index

                # print(middle_index)

            for col_id in np.nonzero(r_row)[0]:
                # if middle_index > 0:
                if col_id < middle_index:
                    right_lane_mask[row_id, col_id] = 0

        return road_mask, right_lane_mask


if __name__ == "__main__":
    from utils.dataset_utils import preprocess
    from natsort import natsorted
    from glob import glob
    import os
    from PIL import Image

    print(os.getcwd())

    simulation_name = ".././simulations/24-01-30-12-39-XAI-seed=14-num-episodes=20-agent=supervised-num-control-nodes=8-max-angle=70/episode8"
    image_names_list = natsorted(glob(os.path.join(simulation_name, '*.jpg')))

    print(len(image_names_list))
    image_list = list(map(Image.open, natsorted(glob(os.path.join(simulation_name, '*.jpg')))))

    img = image_list[0]
    img = preprocess(np.asarray(img), if_yuv = False)
    green_mask = Segment.color_mask(img, color="g")
    yellow_mask = Segment.color_mask(img, color="y")
    white_mask = Segment.color_mask(img, color="w")
    _, right_lane_mask = Segment.right_lane(yellow_mask, green_mask)

    f, ax = plt.subplots(nrows=1, ncols=5)

    ax[0].imshow(img)
    ax[0].set_title("original")
    ax[1].imshow(green_mask)
    ax[1].set_title("green")
    ax[2].imshow(yellow_mask)
    ax[2].set_title("yellow")
    ax[3].imshow(white_mask)
    ax[3].set_title("white")
    ax[4].imshow(right_lane_mask)
    ax[4].set_title("right lane")

    plt.show()
