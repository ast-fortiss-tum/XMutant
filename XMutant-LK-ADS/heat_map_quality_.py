import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
from natsort import natsorted
import pandas as pd
from datetime import datetime

from glob import glob
from PIL import Image
import os

#os.chdir("../")
from utils.dataset_utils import preprocess
from segmentation.road_segmenter import Segment
from attention_manager import AttentionManager

XAI_methods = ["ScoreCAM", "IntegratedGradients", "Faster-ScoreCAM",
                   "GradCAM", "GradCAM++", "VanillaSaliency", "SmoothGrad"]
INDEX_STEP = 5
DETAIL = False


def preprocess_images(images):
    processed_images = []
    processed_images_yuv = []

    for i, image in enumerate(images):
        img = preprocess(np.asarray(image), if_yuv=False)
        img_yuv = preprocess(np.asarray(image), if_yuv=True)
        processed_images.append(img)
        processed_images_yuv.append(img_yuv)

    processed_images = np.asarray(processed_images)
    processed_images_yuv = np.asarray(processed_images_yuv)
    return processed_images, processed_images_yuv


def average_intensity(heat_map, green_mask, yellow_mask, white_mask,
                      road_mask, right_lane_mask, detail=DETAIL):

    average_intensity_right_lane = np.mean(heat_map[right_lane_mask > 0]) if np.sum(right_lane_mask) > 0 else np.nan
    average_intensity_left_lane = np.mean(heat_map[road_mask - right_lane_mask > 0]) \
        if np.sum(road_mask - right_lane_mask) > 0 else np.nan
    average_intensity_yellow = np.mean(heat_map[yellow_mask > 0]) if np.sum(yellow_mask) > 0 else np.nan
    average_intensity_white = np.mean(heat_map[white_mask > 0]) if np.sum(white_mask) > 0 else np.nan
    average_intensity_green = np.mean(heat_map[green_mask > 0]) if np.sum(green_mask) > 0 else np.nan

    if detail:
        print(f"average_intensity_right_lane {average_intensity_right_lane}")
        print(f"average_intensity_left_lane {average_intensity_left_lane}")
        print(f"average_intensity_yellow {average_intensity_yellow}")
        print(f"average_intensity_white {average_intensity_white}")
        print(f"average_intensity_green {average_intensity_green}")

    return (average_intensity_right_lane, average_intensity_left_lane, average_intensity_yellow,
            average_intensity_white, average_intensity_green)


def check_one_episode(simulation_name, xai_method, index_step=INDEX_STEP):
    image_names_list = natsorted(glob(os.path.join(simulation_name, '*.jpg')))
    image_list = list(map(Image.open, image_names_list))
    images = image_list[::index_step]

    processed_images, processed_images_yuv = preprocess_images(images)

    attention_manager = AttentionManager(simulation_name=None)

    heat_map = attention_manager.single_attention_map(preprocessed_image=processed_images_yuv,
                                                      attention_type=xai_method)

    rl_list = np.array([])
    ll_list = np.array([])
    y_list = np.array([])
    w_list = np.array([])
    g_list = np.array([])

    re_ll_list = np.array([])
    re_y_list = np.array([])
    re_w_list = np.array([])
    re_g_list = np.array([])

    for idx in range(processed_images.shape[0]):
        img = processed_images[idx]

        green_mask = Segment.color_mask(img, color="g")
        yellow_mask = Segment.color_mask(img, color="y")
        white_mask = Segment.color_mask(img, color="w")
        road_mask, right_lane_mask = Segment.right_lane(yellow_mask, green_mask)

        av_rl, av_ll, av_y, av_w, av_g = average_intensity(heat_map[0], green_mask, yellow_mask,
                                                           white_mask, road_mask, right_lane_mask, detail=DETAIL)

        rl_list = np.append(rl_list, av_rl)
        ll_list = np.append(ll_list, av_ll)
        y_list = np.append(y_list, av_y)
        w_list = np.append(w_list, av_w)
        g_list = np.append(g_list, av_g)

        if av_rl > 0:
            re_ll = av_ll/av_rl
            re_y = av_y/av_rl
            re_w = av_w/av_rl
            re_g = av_g/av_rl

            re_ll_list = np.append(re_ll_list, re_ll)
            re_y_list = np.append(re_y_list, re_y)
            re_w_list = np.append(re_w_list, re_w)
            re_g_list = np.append(re_g_list, re_g)
        else:
            print(xai_method+"  zero intensity on right lane")
    print(xai_method)

    #print(f"mean of rl_list {np.nanmean(rl_list)}")
    #print(f"mean of ll_list {np.nanmean(ll_list)}")
    #print(f"mean of y_list {np.nanmean(y_list)}")
    #print(f"mean of w_list {np.nanmean(w_list)}")
    #print(f"mean of g_list {np.nanmean(g_list)}")

    print(f"mean of re_ll_list {np.nanmean(re_ll_list)} variance of re_ll_list {np.nanvar(re_ll_list)}")
    print(f"mean of re_y_list {np.nanmean(re_y_list)} variance of re_y_list {np.nanvar(re_y_list)}")
    print(f"mean of re_w_list {np.nanmean(re_w_list)} variance of re_w_list {np.nanvar(re_w_list)}")
    print(f"mean of re_g_list {np.nanmean(re_g_list)} variance of re_g_list {np.nanvar(re_g_list)}")

    mean_rl = np.nanmean(rl_list)
    mean_ll = np.nanmean(ll_list)
    mean_y = np.nanmean(y_list)
    mean_w = np.nanmean(w_list)
    mean_g = np.nanmean(g_list)

    var_rl = np.nanmean(rl_list)
    var_ll = np.nanvar(ll_list)
    var_y = np.nanvar(y_list)
    var_w = np.nanvar(w_list)
    var_g = np.nanvar(g_list)

    mean_re_ll = np.nanmean(re_ll_list)
    mean_re_y = np.nanmean(re_y_list)
    mean_re_w = np.nanmean(re_w_list)
    mean_re_g = np.nanmean(re_g_list)

    var_re_ll = np.nanvar(re_ll_list)
    var_re_y = np.nanvar(re_y_list)
    var_re_w = np.nanvar(re_w_list)
    var_re_g = np.nanvar(re_g_list)

    return ([mean_rl, mean_ll, mean_y, mean_w, mean_g], [var_rl, var_ll, var_y, var_w, var_g],
            [mean_re_ll, mean_re_y, mean_re_w, mean_re_g], [var_re_ll, var_re_y, var_re_w, var_re_g])


def enumerate_sim_folder(timestamp=None, detail: bool = DETAIL):
    collection_df = pd.DataFrame(columns=["xai_method",
                                          "mean_rl", "mean_ll", "mean_y", "mean_w", "mean_g",
                                          "var_rl", "var_ll", "var_y", "var_w", "var_g",
                                          "mean_re_ll", "mean_re_y", "mean_re_w", "mean_re_g",
                                          "var_re_ll", "var_re_y", "var_re_w", "var_re_g"])
    dir = "./simulations"

    folder_glob = os.path.join(dir, "24-*")
    folder_list = []
    folder_list.extend(glob(folder_glob))
    folder_list = [i[14:] for i in folder_list]
    # folder_list_all = os.listdir(dir)
    folder_list = sorted(folder_list, key=lambda x: x[:14])

    if timestamp is None:
        for folder_name in folder_list:
            print("Find folder " + folder_name)
            folder_path = os.path.join(dir, folder_name)
            df = check_one_sim(folder_path, detail=DETAIL)
            collection_df = pd.concat([collection_df, df])
        collection_df.to_csv("./heat_map_ranking_" + "all"+".csv", index=False)

    elif isinstance(timestamp, str):
        find_folder = False
        for folder_name in folder_list:
            if folder_name.startswith(timestamp) or folder_name.endswith(timestamp):
                find_folder = True
                print("Find folder " + folder_name)
                folder_path = os.path.join(dir, folder_name)
                df = check_one_sim(folder_path, detail=DETAIL)
                collection_df = pd.concat([collection_df, df])

        if not find_folder:
            print("Given time does not exist")

        collection_df.to_csv("./heat_map_ranking_" + timestamp +".csv", index=False)

    elif isinstance(timestamp, list):
        time_start = datetime.strptime(timestamp[0], '%y-%m-%d-%H-%M')
        time_end = datetime.strptime(timestamp[1], '%y-%m-%d-%H-%M')
        assert time_start < time_end, "time_start must be earlier than time_end"
        for folder_name in folder_list:
            time_str = datetime.strptime(folder_name[:14], '%y-%m-%d-%H-%M')
            if time_start <= time_str <= time_end:
                print("--------------------------------------------")
                print("Find folder " + folder_name)
                folder_path = os.path.join(dir, folder_name)
                df = check_one_sim(folder_path, detail=DETAIL)
                collection_df = pd.concat([collection_df, df])

        collection_df.to_csv("./segmentation/heat_map_ranking_" + timestamp[0] + "-" + timestamp[1] + ".csv", index=False)


def find_episode_folders(root_dir):
    found_folders = []
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if 'episode' in dir_name:
                folder_path = os.path.join(root, dir_name)
                found_folders.append(folder_path)
                print(folder_path)
    return found_folders


def check_one_sim(folder_path: str, detail: bool = DETAIL):
    summary_df = pd.DataFrame(columns=["xai_method",
                                       "mean_rl", "mean_ll", "mean_y", "mean_w", "mean_g",
                                       "var_rl", "var_ll", "var_y", "var_w", "var_g",
                                       "mean_re_ll", "mean_re_y", "mean_re_w", "mean_re_g",
                                       "var_re_ll", "var_re_y", "var_re_w", "var_re_g"])
    idx = 0
    episode_folders = find_episode_folders(folder_path)

    for episode_folder in episode_folders:

        for xai_method in XAI_methods:
            # [mean_rl, mean_ll, mean_y, mean_w, mean_g], [var_rl, var_ll, var_y, var_w, var_g],
            # [mean_re_ll, mean_re_y, mean_re_w, mean_re_g], [var_re_ll, var_re_y, var_re_w, var_re_g]
            mean_list, var_list, mean_re_list, var_re_list = check_one_episode(episode_folder, xai_method,
                                                                               index_step=INDEX_STEP)

            summary_df.loc[idx] = [xai_method,
                                   mean_list[0], mean_list[1], mean_list[2], mean_list[3], mean_list[4],
                                   var_list[0], var_list[1], var_list[2], var_list[3], var_list[4],
                                   mean_re_list[0], mean_re_list[1], mean_re_list[2], mean_re_list[3],
                                   var_re_list[0], var_re_list[1], var_re_list[2], var_re_list[3],
                                   ]
            idx += 1

    if detail:
        print(summary_df)
    return summary_df


def bar_plot(df_name: str):
    collection_df = pd.read_csv(df_name)
    xai_ranking_result = collection_df.groupby("xai_method").mean()
    print(xai_ranking_result)

    xai_ranking_result_selected = xai_ranking_result[["mean_rl", "mean_ll", "mean_y", "mean_w", "mean_g",
                                                      "var_rl", "var_ll", "var_y", "var_w", "var_g"]]
    
    
    xai_ranking_result_selected.plot(kind='bar', figsize=(10,6))

    plt.title('Average Intensity on different segments')
    plt.xlabel('XAI methods')
    plt.ylabel('average intensity')
    plt.xticks(rotation=10)

    plt.legend(["mean of intensity on right lane", 
                "mean of intensity on left lane", 
                "mean of intensity on yellow line",
                "mean of intensity on white line",
                "mean of intensity on grass",
                "variance of intensity on right lane", 
                "variance of intensity on left lane", 
                "variance of intensity on yellow line",
                "variance of intensity on white line",
                "variance of intensity on grass"])
    #plt.ylim([0,2])
    plt.savefig("./segmentation/bar_plot"+ df_name[:-4]+ ".png")

    xai_ranking_result_re = xai_ranking_result[["mean_re_ll", "mean_re_y", "mean_re_w", "mean_re_g",
                                                "var_re_ll", "var_re_y", "var_re_w", "var_re_g"]]
    

    xai_ranking_result_re.plot(kind='bar', figsize=(10,6))

    plt.title('Relative intensity to the right lane')
    plt.xlabel('XAI methods')
    plt.ylabel('relative intensity')
    plt.xticks(rotation=10)

    plt.legend(["mean of intensity on left lane", 
                "mean of intensity on yellow line",
                "mean of intensity on white line",
                "mean of intensity on grass",
                "variance of intensity on left lane", 
                "variance of intensity on yellow line",
                "variance of intensity on white line",
                "variance of intensity on grass"])
    plt.ylim([0,2])
    plt.savefig("./segmentation/bar_plot_relative"+ df_name[:-4]+ ".png")

def main():
    """simulation_name = ("simulations/24-01-30-12-39-XAI-seed=14-num-episodes=20-agent=supervised-num"
                       "-control-nodes=8-max-angle=70")

    check_one_sim(simulation_name, detail=True)"""
    enumerate_sim_folder("24-01-31-03-58")

if __name__ == "__main__":
    #main()
    #find_episode_folders("simulations/24-01-23-23-55-RANDOM-seed=21-num-episodes=20-agent=supervised-num-control-nodes=9-max-angle=70")
    bar_plot("heat_map_ranking_24-01-31-03-58.csv")

