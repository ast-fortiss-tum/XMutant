import gc
import os
import glob
import shutil
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
collect generated digits from raw data in 'run' folder
"""
def find_num(string):
    return int(''.join(filter(str.isdigit, string)))


def get_mutants(scr_folder, dst_folder, xai_type, record_file_name):
    assert os.path.exists(scr_folder), f"scr_folder path {scr_folder} does not exist"
    folder_list = []
    folder_list.extend(glob.glob(os.path.join(scr_folder, "*" + xai_type)))

    subfolder = "individual_logs"

    csv_list = []
    last_index = None
    last_row = None
    img_array = None
    target_stats =None

    for log in folder_list:

        df_stat = pd.read_csv(os.path.join(log, 'stats.csv'), skiprows=3)

        dst = os.path.join(log, subfolder)

        csv_glob = glob.glob(os.path.join(dst, "*.csv"))
        csv_list = []
        csv_list.extend(csv_glob)

        csv_list = sorted(csv_list, key=find_num)
        # print(csv_list)

        for id, csv_file in enumerate(csv_list):
            # only read last row for the last digit
            df = pd.read_csv(csv_file)
            # find last nonempty row
            last_index = df['mutation_id'].idxmax()
            last_row = df.iloc[last_index]

            del df

            print(csv_file)
            # find misclassified one
            if float(last_row[3]) < 0:
                np_path = last_row[2]
                npy_path = os.path.join(scr_folder, np_path[5:]) # remove "runs" in csv records
                img_array = np.load(npy_path).reshape(28,28)
                # XMutant-MNIST/runs/log03-07_17-08_0_S_R_sm/individual_logs/data/ID0/GEN482.npy
                plt.imshow(img_array, cmap='gray')
                plt.axis("off")
                # plt.show()
                target_stats = df_stat[df_stat['id'] == id]

                dst_path = os.path.join(dst_folder, xai_type)
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                file_name = "digit_"+str(df_stat['expected_label'][0])+"_id_"+str(id)
                file_path = os.path.join(dst_path, file_name)
                data = {'method': xai_type,
                        'id': id,
                        'mutation_number': target_stats['mutate_attempts'].values[0],
                        'expected_label': target_stats['expected_label'].values[0],
                        'predicted_label': target_stats['predicted_label'].values[0],
                        'original_folder': csv_file,
                        'image_path': file_path}
                plt.savefig(file_path + ".png")
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

    xais = ["_sm", "_GC", "_FSC", "_IG"]
    mutations = ["_S_R", "_C_R", "_C_C"]
    all_methods = []
    for xai in xais:
        for mutation in mutations:
            all_methods.append(mutation+xai)
    all_methods.sort(key=str.lower)
    all_methods.append("_R_R")
    print(all_methods)

    #digit = 3
    
    for method in all_methods[12:]:
        #for digit in range(10):
        get_mutants(scr_folder, dst_folder, method, "record" + method)
