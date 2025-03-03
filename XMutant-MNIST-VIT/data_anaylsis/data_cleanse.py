import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# if os.getcwd().split("/")[-1] != "XMutant-MNIST-VIT":
#     print("change directory")
#     os.chdir("../")

csv_folder = "../result/digits"
assert os.path.exists(csv_folder), f"scr_folder path {csv_folder} does not exist"

# -------------------------method string-------------------------
all_methods = [ "_R_R", "_C_C"]

print(all_methods)

# ---------------------------------------------------------------
""" Add column drop for mutation_number = 1 and mutation_number-1"""


def modify_record_csv():
    sum_csv_list = []
    sum_csv_list.extend(glob.glob(os.path.join(csv_folder, "record*")))

    print(sum_csv_list)
    for record_csv in sum_csv_list:
        record_df = pd.read_csv(record_csv)
        record_df['drop'] = record_df['mutation_number'] == 1
        record_df['mutation_number_real'] = record_df['mutation_number'] - 1
        record_df.to_csv(record_csv, index=False)

# modify_record_csv()
# --------------------------------------------------------------------
def check_validity_rate_over_iteration(record_name, digit=None):
    RESULTS_PATH = r"../result/digits/"
    df_record = pd.read_csv(record_name)

    if digit is not None:
        df_record = df_record[df_record["expected_label"] == digit]

    # ----------------------------------------------------------------
    # df_record['ID'] = df_record['ID/OOD'].apply(lambda x:1 if x.lower()== "id" else 0)
    df_record['one'] = 1
    df_record['mutation_number_real'] = df_record['mutation_number_real'].astype(int)
    df_record = df_record[df_record['mutation_number_real']!=0]
    df_cumulative = pd.DataFrame()
    df_cumulative["idx"] = df_record.groupby('mutation_number_real')['one'].sum().to_frame().index


    df_cumulative["pop_num"] = df_record.groupby('mutation_number_real')['one'].sum().to_list()
    df_cumulative["pop_cum_num"] = df_record.groupby('mutation_number_real')['one'].sum().cumsum().to_list()

    df_cumulative.to_csv(os.path.join(RESULTS_PATH, "cumulative_clear_validity_rate_" + record_name.split("/")[-1]))

# RESULTS_PATH = r"../result/digits"
# folders = glob.glob(os.path.join(RESULTS_PATH, "record*.csv"))
# print(folders)
# for csv_file in folders:
#     # main_xc(csv_file) #get loss
#     # new_threshold(record_name=csv_file) # compare all thresholds
#     check_validity_rate_over_iteration(record_name=csv_file, digit=None) # get cumulative validity rate
# --------------------------------------------------------------------

def cumulative_misclassified(if_save=False):
    name_list = glob.glob(os.path.join("../result", "csv_folder", "cumulative_clear_*.csv"))
    name_list.sort(key=str.lower)
    #change order of r_r
    to_be_moved = [i for i in name_list if "R_R" in i]
    name_list.remove(to_be_moved[0])
    name_list.append(to_be_moved[0])

    print(name_list)

    col_names = [("_").join(name.split("/")[-1][:-4].split("_")[5:]) for name in name_list]

    df = pd.DataFrame(columns=['idx'] + col_names)
    df.idx = np.linspace(1,1000,1000)
    plt.figure(figsize=(16, 10), dpi=100)
    for col, csv_file in zip(col_names, name_list):
        df_temp = pd.read_csv(csv_file)
        plt.plot(df_temp['idx'], df_temp['pop_cum_num'])
        # line_plot(df_temp[col], title='Line plot of Mutation iteration of digit '+col)
        for index, row in df.iterrows():
            #print(row['idx'])

            df_temp_idx = df_temp['idx'][df_temp['idx']<=row['idx']].max()

            df.at[index, col] = df_temp[df_temp['idx'] == df_temp_idx]['pop_cum_num'].values[0]

    #  '../result/csv_folder/cumulative_clear_validity_rate_record_R_R.csv'
    plt.legend(col_names)
    plt.show()

    df.to_csv(os.path.join("../result", 'csv_folder', "cumulative_misclassified_all.csv"), index=False)
# print(os.getcwd())
# cumulative_misclassified()
# --------------------------------------------------------------


def paper_table():
    df_cum = pd.read_csv(os.path.join("../result", 'csv_folder', "cumulative_misclassified_all.csv"))
    col_names = df_cum.columns

    df_cum_per_100 = pd.DataFrame(columns=df_cum.columns)
    df_cum_per_100.idx = np.linspace(10,100,10)

    for index, row in df_cum_per_100.iterrows():
        iteration = row.idx
        for col in col_names[1:]:
            number = df_cum[df_cum['idx'] == iteration][col].values[0]/(2000 -23) #
            efficiency = df_cum[col].cumsum()[iteration-1] / df_cum['R_R'].cumsum()[iteration-1]

            df_cum_per_100.at[index, col] = f"{number:.2f}-{efficiency:.2f}"

    df_cum_per_100.to_csv(os.path.join("../result", 'csv_folder', "paper_table.csv"), index=False)
paper_table()