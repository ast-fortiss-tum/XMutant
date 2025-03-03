import gc
import os
import glob
import shutil
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
"""
collect generated digits from raw data in 'run' folder
"""
def find_num(string):
    return int(''.join(filter(str.isdigit, string)))


def get_mutants(scr_folder, dst_folder, xai_type, record_file_name):
    assert os.path.exists(scr_folder), f"scr_folder path {scr_folder} does not exist"
    folder_list = []
    folder_list.extend(glob.glob(os.path.join(scr_folder, "*" + xai_type)))



    for log in folder_list:

        df_stat = pd.read_csv(os.path.join(log, 'stats.csv'), skiprows=3)
        # filter misbehaviour
        df_stat = df_stat[df_stat['misbehaviour'] == True]

        # digit_ids = df_stat['id'].
        #
        # dst = os.path.join(log, 'individual_logs', 'data')
        # csv_glob = glob.glob(os.path.join(dst, "*stats.csv"))
        # csv_list = []
        # csv_list.extend(csv_glob)
        #
        # csv_list = sorted(csv_list, key=find_num)
        # print(csv_list)

        # enumerate df_stat list
        for row in df_stat.iterrows():
            id = row[1]['id']
            gen = int(row[1]['mutate_attempts'])

            npy_path = os.path.join(log, 'individual_logs', 'data', 'ID' + str(id), 'GEN' + str(gen) +'.npy') # remove "runs" in csv records
            csv_file = os.path.join(log, 'individual_logs', str(gen) +'.csv')
            assert os.path.exists(npy_path), f"npy_path {npy_path} does not exist"
            assert os.path.exists(csv_file), f"csv_file {csv_file} does not exist"
            # use PIL save

            img_array = np.load(npy_path).reshape(28,28)
            img = Image.fromarray(np.uint8(img_array*255))

            # plt.imshow(img_array, cmap='gray')
            # plt.axis("off")
            # plt.show()

            dst_path = os.path.join(dst_folder, xai_type)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            file_name = "digit_"+str(row[1]['expected_label'])+"_id_"+str(id)
            file_path = os.path.join(dst_path, file_name)
            data = {'method': xai_type,
                    'id': id,
                    'mutation_number': gen,
                    'expected_label': row[1]['expected_label'],
                    'predicted_label': row[1]['predicted_label'],
                    'original_folder': csv_file,
                    'image_path': file_path}
            img.save(file_path + ".png")
            plt.close()
            shutil.copy2(npy_path, os.path.join(dst_path, file_name+".npy"))

            record_path = os.path.join(dst_folder, record_file_name + ".csv")
            # check if the record csv already exists

            if os.path.exists(record_path):
                with open(record_path, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(list(data.values()))
            else:
                with open(record_path, 'w', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(list(data.keys()))
                    csv_writer.writerow(list(data.values()))
            gc.collect()
        del df_stat
        gc.collect()


if __name__ == "__main__":
    print(os.getcwd())
    #os.chdir("./data_analysis")
    #print(os.getcwd())
    
    scr_folder =  r"../runs/"  #"../../../dataset/final-test2"#
    dst_folder = "../result/digits"


    all_methods = ["_R_R", "_C_C"]
    print(all_methods)

    #digit = 3
    
    for method in all_methods:
        #for digit in range(10):
        get_mutants(scr_folder, dst_folder, method, "record" + method)
