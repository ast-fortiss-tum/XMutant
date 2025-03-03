import pandas as pd
import os
import glob
import numpy as np
from datetime import datetime
from natsort import natsorted

import matplotlib.pyplot as plt
import seaborn as sns

# import csv_logger
from stat_tests import run_wilcoxon_and_cohend, run_wilcoxon, cohend




"""global_log_old = pd.read_csv("simulations_simple/simulations_before_rearrange_cp/global_log.csv",
                         usecols=['mutation point', 'Time', "episode id", "seed",
                                  'mutation type', 'mutation method',
                                  'is success'])
global_log_old['mutation point'] = pd.to_numeric(global_log_old['mutation point'], errors='coerce')"""


def reset_two_dic():
    previous = {
            "cte_l1_m1": np.nan,
            "cte_l1": np.nan,
            "cte_l1_p1": np.nan,
            "cte_l2_m1": np.nan,
            "cte_l2": np.nan,
            "cte_l2_p1": np.nan,
            "cte_max_m1": np.nan,
            "cte_max": np.nan,
            "cte_max_p1": np.nan,
            "cte_der_m1": np.nan,
            "cte_der": np.nan,
            "cte_der_p1": np.nan,

            "steering_l1_m1": np.nan,
            "steering_l1": np.nan,
            "steering_l1_p1": np.nan,
            "steering_l2_m1": np.nan,
            "steering_l2": np.nan,
            "steering_l2_p1": np.nan,
            "steering_max_m1": np.nan,
            "steering_max": np.nan,
            "steering_max_p1": np.nan,
            "steering_der_m1": np.nan,
            "steering_der": np.nan,
            "steering_der_p1": np.nan
        }
    after = previous.copy()
    return previous, after

def reset_dq_dic_simple():
    dq_data = {
            "cte_max_before": np.nan,
            "cte_max_after": np.nan,
            "steering_l2_before": np.nan,
            "steering_l2_after": np.nan,
        }
    return dq_data


def get_dqd_data_as_csv(global_log: pd.DataFrame,
                        folder_path: str,
                        summary_path: str = 'results/summary_increment.csv'):
    # folder_path format "simulations/03-18-16-58-RANDOM-seed=1"
    log_time = folder_path.split('/')[-1][:11]


    csv_glob = os.path.join(folder_path, "episode*.csv")
    csv_name_list = []
    csv_name_list.extend(glob.glob(csv_glob))
    csv_name_list = natsorted(csv_name_list)

    idx = 0

    for idx, csv_name in enumerate(csv_name_list[:-1]):
        csv_name_after = csv_name_list[idx+1]
        episode_id = int(csv_name.split('/')[-1][7:-4])
        global_row = global_log[(global_log["Time"] == log_time) &
                                    (global_log["episode id"] == episode_id+1)]
        mutation_point = global_row['mutation point'].values[0]
        assert mutation_point != np.nan
        df_temp = pd.read_csv(csv_name)
        df_temp = df_temp[(df_temp['closest_cp'] == mutation_point-1) |
                          (df_temp['closest_cp'] == mutation_point) |
                          (df_temp['closest_cp'] == mutation_point + 1)]
        previous, after = reset_two_dic()
        for obj in ['cte', 'steering']:
            df_temp[obj + '_l2'] = df_temp[obj] ** 2
            df_temp[obj + '_l1'] = df_temp[obj].abs()
            df_temp[obj + '_der'] = df_temp[obj] - df_temp[obj].shift(1)
            df_temp[obj + '_der'] = df_temp[obj + '_der'].abs()

            if mutation_point in df_temp['closest_cp'].unique():
                previous[obj + '_l1'] = df_temp.groupby('closest_cp')[obj + '_l1'].mean()[mutation_point]
                previous[obj + '_l2'] = df_temp.groupby('closest_cp')[obj + '_l2'].mean()[mutation_point]
                previous[obj + '_der'] = df_temp.groupby('closest_cp')[obj + '_der'].mean()[mutation_point]
                previous[obj + '_max'] = df_temp.groupby('closest_cp')[obj + '_l1'].max()[mutation_point]
            if mutation_point-1 in df_temp['closest_cp'].unique():
                previous[obj + '_l1_m1'] = df_temp.groupby('closest_cp')[obj + '_l1'].mean()[mutation_point-1]
                previous[obj + '_l2_m1'] = df_temp.groupby('closest_cp')[obj + '_l2'].mean()[mutation_point-1]
                previous[obj + '_der_m1'] = df_temp.groupby('closest_cp')[obj + '_der'].mean()[mutation_point-1]
                previous[obj + '_max_m1'] = df_temp.groupby('closest_cp')[obj + '_l1'].max()[mutation_point-1]
            if mutation_point+1 in df_temp['closest_cp'].unique():
                previous[obj + '_l1_p1'] = df_temp.groupby('closest_cp')[obj + '_l1'].mean()[mutation_point+1]
                previous[obj + '_l2_p1'] = df_temp.groupby('closest_cp')[obj + '_l2'].mean()[mutation_point+1]
                previous[obj + '_der_p1'] = df_temp.groupby('closest_cp')[obj + '_der'].mean()[mutation_point+1]
                previous[obj + '_max_p1'] = df_temp.groupby('closest_cp')[obj + '_l1'].max()[mutation_point+1]
        del df_temp

        df_temp_after = pd.read_csv(csv_name_after)
        df_temp_after = df_temp_after[(df_temp_after['closest_cp'] == mutation_point-1) |
                          (df_temp_after['closest_cp'] == mutation_point) |
                          (df_temp_after['closest_cp'] == mutation_point + 1)]
        for obj in ['cte', 'steering']:
            df_temp_after[obj + '_l2'] = df_temp_after[obj] ** 2
            df_temp_after[obj + '_l1'] = df_temp_after[obj].abs()
            df_temp_after[obj + '_der'] = df_temp_after[obj] - df_temp_after[obj].shift(1)
            df_temp_after[obj + '_der'] = df_temp_after[obj + '_der'].abs()

            if mutation_point in df_temp_after['closest_cp'].unique():
                after[obj + '_l1'] = df_temp_after.groupby('closest_cp')[obj + '_l1'].mean()[mutation_point]
                after[obj + '_l2'] = df_temp_after.groupby('closest_cp')[obj + '_l2'].mean()[mutation_point]
                after[obj + '_der'] = df_temp_after.groupby('closest_cp')[obj + '_der'].mean()[mutation_point]
                after[obj + '_max'] = df_temp_after.groupby('closest_cp')[obj + '_l1'].max()[mutation_point]
            if mutation_point-1 in df_temp_after['closest_cp'].unique():
                after[obj + '_l1_m1'] = df_temp_after.groupby('closest_cp')[obj + '_l1'].mean()[mutation_point-1]
                after[obj + '_l2_m1'] = df_temp_after.groupby('closest_cp')[obj + '_l2'].mean()[mutation_point-1]
                after[obj + '_der_m1'] = df_temp_after.groupby('closest_cp')[obj + '_der'].mean()[mutation_point-1]
                after[obj + '_max_m1'] = df_temp_after.groupby('closest_cp')[obj + '_l1'].max()[mutation_point-1]
            if mutation_point+1 in df_temp_after['closest_cp'].unique():
                after[obj + '_l1_p1'] = df_temp_after.groupby('closest_cp')[obj + '_l1'].mean()[mutation_point+1]
                after[obj + '_l2_p1'] = df_temp_after.groupby('closest_cp')[obj + '_l2'].mean()[mutation_point+1]
                after[obj + '_der_p1'] = df_temp_after.groupby('closest_cp')[obj + '_der'].mean()[mutation_point+1]
                after[obj + '_max_p1'] = df_temp_after.groupby('closest_cp')[obj + '_l1'].max()[mutation_point+1]
        del df_temp_after

        increment = {key: after[key]-previous[key] for key in after}

        increment["time"] = log_time
        increment["mutation type"] = global_row['mutation type'].values[0]
        increment["mutation method"] = global_row['mutation method'].values[0]
        increment["episode"] = episode_id

        csv_logger.episode_logger(filepath=summary_path, log_info=increment)

def get_dqd_data_as_csv_simple(global_log: pd.DataFrame,
                        folder_path: str,
                        summary_path: str = 'results/summary_increment.csv'):
    # folder_path format "simulations/03-18-16-58-RANDOM-seed=1"
    log_time = folder_path.split('/')[-1][:11]


    csv_glob = os.path.join(folder_path, "episode*.csv")
    csv_name_list = []
    csv_name_list.extend(glob.glob(csv_glob))
    csv_name_list = natsorted(csv_name_list)

    idx = 0

    for idx, csv_name in enumerate(csv_name_list[:-1]):
        csv_name_after = csv_name_list[idx+1]
        episode_id = int(csv_name.split('/')[-1][7:-4])
        global_row = global_log[(global_log["Time"] == log_time) &
                                    (global_log["episode id"] == episode_id+1)]
        mutation_point = global_row['mutation point'].values[0]
        assert mutation_point != np.nan
        df_temp = pd.read_csv(csv_name)
        df_temp = df_temp[(df_temp['closest_cp'] == mutation_point)]

        dq_data = reset_dq_dic_simple()

        df_temp['steering_l2'] = df_temp['steering'] ** 2
        df_temp['cte_l1'] = df_temp['cte'].abs()

        if mutation_point in df_temp['closest_cp'].unique():
            dq_data['steering_l2_before'] = df_temp.groupby('closest_cp')['steering_l2'].mean()[mutation_point]
            dq_data['cte_max_before'] = df_temp.groupby('closest_cp')['cte_l1'].max()[mutation_point]

        del df_temp

        df_temp_after = pd.read_csv(csv_name_after)
        df_temp_after = df_temp_after[(df_temp_after['closest_cp'] == mutation_point)]

        df_temp_after['steering_l2'] = df_temp_after['steering'] ** 2
        df_temp_after['cte_l1'] = df_temp_after['cte'].abs()

        if mutation_point in df_temp_after['closest_cp'].unique():
            dq_data['steering_l2_after'] = df_temp_after.groupby('closest_cp')['steering_l2'].mean()[mutation_point]
            dq_data['cte_max_after'] = df_temp_after.groupby('closest_cp')['cte_l1'].max()[mutation_point]

        del df_temp_after

        dq_data["time"] = log_time
        dq_data["mutation type"] = global_row['mutation type'].values[0] + "_" + folder_path.split("/")[-2].split("_")[-1]
        dq_data["mutation method"] = global_row['mutation method'].values[0]
        dq_data["episode"] = episode_id

        csv_logger.episode_logger(filepath=summary_path, log_info=dq_data)

def walk_folder_dqd(timestamp=None,
                sim_dir: str = "./simulations",
                summary_path: str = '../results/csvs/summary_increment.csv'):

    global_log = pd.read_csv(os.path.join(sim_dir, "global_log.csv"), index_col=False)
    folder_glob = os.path.join(sim_dir, "03*")

    folder_list = []
    folder_list.extend(glob.glob(folder_glob))
    # folder_list = [i[14:] for i in folder_list]
    # folder_list_all = os.listdir(dir)
    folder_list = sorted(folder_list, key=lambda x: x.split('/')[-1][:11])
    if timestamp is None:
        for folder_name in folder_list:
            print("Find folder " + folder_name)

            get_dqd_data_as_csv(global_log=global_log, folder_path=folder_name, summary_path=summary_path)
    elif isinstance(timestamp, str):
        find_folder = False
        for folder_name in folder_list:
            if folder_name.startswith(timestamp) or folder_name.endswith(timestamp):
                find_folder = True
                print("Find folder " + folder_name)

                get_dqd_data_as_csv(global_log=global_log, folder_path=folder_name, summary_path=summary_path)
        if not find_folder:
            print("Given time does not exist")
    elif isinstance(timestamp, list):
        time_start = datetime.strptime(timestamp[0], '%y-%m-%d-%H-%M')
        time_end = datetime.strptime(timestamp[1], '%y-%m-%d-%H-%M')
        assert time_start<time_end, "time_start must be earlier than time_end"
        for folder_name in folder_list:
            time_str = datetime.strptime(folder_name[:14], '%y-%m-%d-%H-%M')
            if time_start <= time_str <= time_end:
                print("--------------------------------------------")
                print("Find folder " + folder_name)

                get_dqd_data_as_csv(global_log=global_log, folder_path=folder_name, summary_path=summary_path)

def walk_folder_dqd_simple(timestamp=None,
                sim_dir: str = "./simulations",
                summary_path: str = '../results/csvs/summary_increment.csv'):

    global_log = pd.read_csv(os.path.join(sim_dir, "global_log.csv"), index_col=False)
    folder_glob = os.path.join(sim_dir, "03*")

    folder_list = []
    folder_list.extend(glob.glob(folder_glob))
    # folder_list = [i[14:] for i in folder_list]
    # folder_list_all = os.listdir(dir)
    folder_list = sorted(folder_list, key=lambda x: x.split('/')[-1][:11])
    if timestamp is None:
        for folder_name in folder_list:
            print("Find folder " + folder_name)

            get_dqd_data_as_csv_simple(global_log=global_log, folder_path=folder_name, summary_path=summary_path)
    elif isinstance(timestamp, str):
        find_folder = False
        for folder_name in folder_list:
            if folder_name.startswith(timestamp) or folder_name.endswith(timestamp):
                find_folder = True
                print("Find folder " + folder_name)

                get_dqd_data_as_csv_simple(global_log=global_log, folder_path=folder_name, summary_path=summary_path)
        if not find_folder:
            print("Given time does not exist")
    elif isinstance(timestamp, list):
        time_start = datetime.strptime(timestamp[0], '%y-%m-%d-%H-%M')
        time_end = datetime.strptime(timestamp[1], '%y-%m-%d-%H-%M')
        assert time_start<time_end, "time_start must be earlier than time_end"
        for folder_name in folder_list:
            time_str = datetime.strptime(folder_name[:14], '%y-%m-%d-%H-%M')
            if time_start <= time_str <= time_end:
                print("--------------------------------------------")
                print("Find folder " + folder_name)

                get_dqd_data_as_csv_simple(global_log=global_log, folder_path=folder_name, summary_path=summary_path)

"""def dqd_estimation(timestamps, imgname: str =None):
    dir = "./simulations"

    folder_glob = os.path.join(dir, "24-*")
    folder_list = []
    folder_list.extend(glob.glob(folder_glob))
    folder_list = [i[14:] for i in folder_list]
    folder_list = sorted(folder_list, key=lambda x: x[:14])

    time_start = datetime.strptime(timestamps[0], '%y-%m-%d-%H-%M')
    time_end = datetime.strptime(timestamps[1], '%y-%m-%d-%H-%M')
    assert time_start < time_end, "time_start must be earlier than time_end"

    df_oot = pd.DataFrame(columns=["method combi", "oot"])
    df_dqd = pd.DataFrame(columns=["mse_increments", "max_increments", "method combi"])
    for folder_name in folder_list:
        time_str = datetime.strptime(folder_name[:14], '%y-%m-%d-%H-%M')

        if time_start <= time_str <= time_end:
            summary_csv = os.path.join(dir, folder_name, "summary.csv")
            df_sum = pd.read_csv(summary_csv)

            if "RANDOM" in folder_name:
                if len(df_sum) > 1:
                    df_oot.loc[len(df_oot)] = ["Random", len(df_sum)]

                    df_temp = df_sum[['mse_increments', 'max_increments']].copy()
                    df_temp.dropna(inplace=True)

                    df_temp["method combi"] = "Random"
                    #p#rint(df_temp)
                    df_dqd = pd.concat([df_dqd, df_temp], ignore_index=True)
                    del df_temp
            elif "XAI" in folder_name:
                if len(df_sum) > 1:
                    df_oot.loc[len(df_oot)] = ["XAI", len(df_sum)]

                    df_temp = df_sum[['mse_increments', 'max_increments']].copy()
                    df_temp.dropna(inplace=True)

                    df_temp["method combi"] = "XAI"
                    #print(df_temp)

                    df_dqd = pd.concat([df_dqd, df_temp], ignore_index=True)
                    del df_temp"""

def dqd_plot(dqd_path="results/summary_increment.csv",
             save_img: bool = False,
             img_path: bool = "result/vit_model/"):
    df_dqd = pd.read_csv(dqd_path)
    df_dqd["method combi"] = df_dqd["mutation type"].values + " " + df_dqd["mutation method"].values
    print(df_dqd["method combi"].unique())
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))

    previous, _ = reset_two_dic()
    for col_name in previous.keys():
        sns.boxplot(x="method combi",
                    y=col_name,
                    data=df_dqd,
                    showfliers=False, #don't show outliers
                    showmeans=True,
                    order=['RANDOM random', 'XAI random', 'XAI attention_opposite', 'XAI attention_same']
                    )

        plt.title('Driving Quality Degradation '+col_name)
        plt.ylabel('Increments')
        
        plt.tight_layout()
        plt.show()


def dqd_stat_simple(dqd_path="results/summary_increment.csv",
             save_img: bool = False,
             img_path: bool = "result/vit_model/"):
    df_dqd = pd.read_csv(dqd_path)
    # print(df_dqd["method combi"].unique())
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    keys_simple = [
            "cte_l1",
            "cte_l2",
            "cte_max",
            "cte_der",
            "steering_l1",
            "steering_l2",
            "steering_max",
            "steering_der",
        ]

    df_stat = pd.DataFrame(index=df_dqd['method combi'].unique())
    for col_name in keys_simple:
        means = df_dqd.groupby('method combi')[col_name].mean()
        means.name = col_name + " means"
        df_stat = pd.concat([df_stat, means], axis=1)

        stds = df_dqd.groupby('method combi')[col_name].std()
        stds.name = col_name + " stds"
        df_stat = pd.concat([df_stat, stds], axis=1)


        baseline = df_stat.at["RANDOM random", col_name + ' means']
        df_stat[col_name + ' improvements'] = df_stat[col_name + ' means'].apply(lambda x: (x-baseline)/baseline)

def dqd_plot_simple(dqd_path="results/summary_increment.csv"):
    df_dqd = pd.read_csv(dqd_path)
    # print(df_dqd["method combi"].unique())
    sns.set_theme(style="whitegrid")


    keys_simple = [
        "cte_max",
        "steering_l2"
    ]

    # df_stat = pd.DataFrame(index=df_dqd['method combi'].unique())
    # only need FSC and GC for plot
    df_dqd = df_dqd[df_dqd['method combi'].str.contains('RANDOM random') | df_dqd['method combi'].str.contains("FSC")
                    | df_dqd['method combi'].str.contains("GC")]
    df_dqd = df_dqd[keys_simple + ['method combi']]

    df_dqd_cte = df_dqd[["cte_max", "method combi"]]
    df_dqd_cte["type"] = "cte_max"
    df_dqd_cte = df_dqd_cte.rename(columns={'cte_max': 'value'})

    df_dqd_steer = df_dqd[["steering_l2", "method combi"]]
    df_dqd_steer["type"] = "steering_l2"
    df_dqd_steer = df_dqd_steer.rename(columns={'steering_l2': 'value'})

    df_dqd_plot = pd.concat([df_dqd_cte,df_dqd_steer], ignore_index=True)

    fig, ax1 = plt.subplots(figsize=(7.8, 5.51))
    ax2 = ax1.twinx()
    sns.set_theme(style="ticks", palette="pastel")

    v1 = sns.violinplot(ax=ax1,
                x="method combi",
                y= "value",
                hue=True,
                hue_order=[False, True],
                data=df_dqd_cte,
                split=True,
                inner="quart",
                #showfliers=False,  # don't show outliers
                #showmeans=True,
                palette = ["m"],
                #legend=['CTE']
                order=['RANDOM random', 'XAI random GC', 'XAI attention_same GC', 'XAI random FSC']
                )

    v2 = sns.violinplot(ax=ax2,
                   x="method combi",
                   y="value",
                   hue=True,
                   hue_order=[True, False],
                   data=df_dqd_steer,
                   split=True,
                   inner="quart",
                   # showfliers=False,  # don't show outliers
                   # showmeans=True,
                   palette = ["g"],
                   legend = None,
                   order=['RANDOM random', 'XAI random GC', 'XAI attention_same GC', 'XAI random FSC']
                   )

    p1 = sns.pointplot(ax=ax1, x="method combi", y="value",hue="type",
                       marker="_", markersize=20, markeredgewidth=3, linestyle="none",
                       order=['RANDOM random', 'XAI random GC', 'XAI attention_same GC', 'XAI random FSC'],
                    data=df_dqd_cte, estimator=np.mean, palette = ["darkred"],legend = None)
    p2 = sns.pointplot(ax=ax2, x="method combi", y="value", hue="type",
                       marker="_", markersize=20, markeredgewidth=3,linestyle="none",
                       order=['RANDOM random', 'XAI random GC', 'XAI attention_same GC', 'XAI random FSC'],
                    data=df_dqd_steer, estimator=np.mean,color = "seagreen",legend = None)

    #fig.legend([v1,v2,p1,p2 ], labels=[['CTE'], ['steering'], ['CTE mean'], ['steering mean']], loc="upper right")

    labs = ["cte","steer","1","2"]
    ax1.set_xticklabels(['Random', 'GradCAM++ random', 'GradCAM++ high', 'Faster-ScoreCAM random'], rotation=15)

    #ax1.legend(lns, labs, loc=0)
    ax1.legend(["cte"],loc='upper left')
    ax2.legend(["Steer"],loc='upper right')
    ax1.set_title('Driving Quality Degradations')

    ax1.set_xlabel('')
    ax1.set_ylabel('DQD of CTEs')
    ax2.set_ylabel('DQD of Steering Angles')
    fig.tight_layout()
    plt.show()


def dqd_plot_simple2(dqd_path="../results/csvs/summary_driving_quality.csv"):
    df_dq = pd.read_csv(dqd_path)
    # print(df_dqd["method combi"].unique())
    sns.set_theme(style="whitegrid")


    keys_simple = [
        "cte_max_before","cte_max_after","steering_l2_before","steering_l2_after","combi"
    ]

    # df_stat = pd.DataFrame(index=df_dqd['method combi'].unique())
    # only need FSC and GC for plot
    df_dq = df_dq[(df_dq['mutation type'].str.contains('RANDOM'))
                    | (df_dq['mutation type'].str.contains("FSC"))
                    | (df_dq['mutation type'].str.contains("GC"))]
    df_dq["combi"] = df_dq["mutation type"] + "_" +df_dq["mutation method"]


    df_dq = df_dq[keys_simple]

    df_dq = df_dq.dropna()

    df_dq["cte_max"] = df_dq["cte_max_after"] - df_dq["cte_max_before"]
    df_dq["steering_l2"] = df_dq["steering_l2_after"] - df_dq["steering_l2_before"]

    df_dqd_cte = df_dq[["combi","cte_max"]]
    df_dqd_cte["type"] = "cte_max"
    df_dqd_cte = df_dqd_cte.rename(columns={'cte_max': 'value'})

    df_dqd_steer = df_dq[["combi","steering_l2"]]

    df_dqd_steer = df_dqd_steer.rename(columns={'steering_l2': 'value'})
    df_dqd_steer["type"] = "steering_l2"
    df_dqd_plot = pd.concat([df_dqd_cte,df_dqd_steer], ignore_index=True)

    for combi in ['RANDOM_FSC_random', 'XAI_GC_random', 'XAI_GC_attention_same', 'XAI_FSC_random']:
        print("-------------" + combi + "-------------")
        print("---cte_max")
        run_wilcoxon(df_dq[df_dq['combi'] == combi]["cte_max"])
        cohensd = cohend(df_dq[df_dq['combi'] == combi]["cte_max_before"], df_dq[df_dq['combi'] == combi]["cte_max_after"])
        print(f"Cohen's D is: {cohensd}")
        print("---steering_l2")
        run_wilcoxon(df_dq[df_dq['combi'] == combi]["steering_l2"])
        cohensd = cohend(df_dq[df_dq['combi'] == combi]["steering_l2_before"], df_dq[df_dq['combi'] == combi]["steering_l2_after"])
        print(f"Cohen's D is: {cohensd}")
    # plot
    fig, ax1 = plt.subplots(figsize=(7.8, 5.51))
    ax2 = ax1.twinx()
    sns.set_theme(style="ticks", palette="pastel")

    v1 = sns.violinplot(ax=ax1,
                x="combi",
                y= "value",
                hue=True,
                hue_order=[False, True],
                data=df_dqd_cte,
                split=True,
                inner="quart",
                #showfliers=False,  # don't show outliers
                #showmeans=True,
                palette = ["m"],
                #legend=['CTE']
                order=['RANDOM_FSC_random', 'XAI_GC_random', 'XAI_GC_attention_same', 'XAI_FSC_random']
                )

    v2 = sns.violinplot(ax=ax2,
                   x="combi",
                   y="value",
                   hue=True,
                   hue_order=[True, False],
                   data=df_dqd_steer,
                   split=True,
                   inner="quart",
                   # showfliers=False,  # don't show outliers
                   # showmeans=True,
                   palette = ["g"],
                   legend = None,
                   order=['RANDOM_FSC_random', 'XAI_GC_random', 'XAI_GC_attention_same', 'XAI_FSC_random']
                   )

    p1 = sns.pointplot(ax=ax1, x="combi", y="value",hue="type",
                       marker="_", markersize=20, markeredgewidth=3, linestyle="none",
                       order=['RANDOM_FSC_random', 'XAI_GC_random', 'XAI_GC_attention_same', 'XAI_FSC_random'],
                    data=df_dqd_cte, estimator=np.mean, palette = ["darkred"],legend = None)
    p2 = sns.pointplot(ax=ax2, x="combi", y="value", hue="type",
                       marker="_", markersize=20, markeredgewidth=3,linestyle="none",
                       order=['RANDOM_FSC_random', 'XAI_GC_random', 'XAI_GC_attention_same', 'XAI_FSC_random'],
                    data=df_dqd_steer, estimator=np.mean,color = "seagreen",legend = None)

    #fig.legend([v1,v2,p1,p2 ], labels=[['CTE'], ['steering'], ['CTE mean'], ['steering mean']], loc="upper right")

    labs = ["cte","steer","1","2"]
    ax1.set_xticklabels(['Random', 'Grad-CAM++ random', 'Grad-CAM++ high', 'Score-CAM random'], rotation=15)

    #ax1.legend(lns, labs, loc=0)
    ax1.legend(["CTE"],loc='upper left')
    ax2.legend(["Steering"],loc='upper right')

    ax1.set_title('Driving Quality Degradations')

    ax1.set_xlabel('')
    ax1.set_ylabel('DQD of CTEs')
    ax2.set_ylabel('DQD of Steering Angles')
    fig.tight_layout()
    plt.show()

def dqd_plot_simple_horizontal(dqd_path="../results/csvs/summary_driving_quality.csv"):
    df_dq = pd.read_csv(dqd_path)
    # print(df_dqd["method combi"].unique())
    sns.set_theme(style="whitegrid")


    keys_simple = [
        "cte_max_before","cte_max_after","steering_l2_before","steering_l2_after","combi"
    ]

    # df_stat = pd.DataFrame(index=df_dqd['method combi'].unique())
    # only need FSC and GC for plot
    df_dq = df_dq[(df_dq['mutation type'].str.contains('RANDOM'))
                    #| (df_dq['mutation type'].str.contains("FSC"))
                    | (df_dq['mutation type'].str.contains("GC"))]
    df_dq["combi"] = df_dq["mutation type"] + "_" +df_dq["mutation method"]


    df_dq = df_dq[keys_simple]

    df_dq = df_dq.dropna()

    df_dq["cte_max"] = df_dq["cte_max_after"] - df_dq["cte_max_before"]
    df_dq["steering_l2"] = df_dq["steering_l2_after"] - df_dq["steering_l2_before"]

    df_dqd_cte = df_dq[["combi","cte_max"]]
    df_dqd_cte["type"] = "cte_max"
    df_dqd_cte = df_dqd_cte.rename(columns={'cte_max': 'value'})

    df_dqd_steer = df_dq[["combi","steering_l2"]]

    df_dqd_steer = df_dqd_steer.rename(columns={'steering_l2': 'value'})
    df_dqd_steer["type"] = "steering_l2"
    df_dqd_plot = pd.concat([df_dqd_cte,df_dqd_steer], ignore_index=True)

    """for combi in ['RANDOM_FSC_random', 'XAI_GC_random', 'XAI_GC_attention_same', 'XAI_FSC_random']:
        print("-------------" + combi + "-------------")
        print("---cte_max")
        run_wilcoxon(df_dq[df_dq['combi'] == combi]["cte_max"])
        cohensd = cohend(df_dq[df_dq['combi'] == combi]["cte_max_before"], df_dq[df_dq['combi'] == combi]["cte_max_after"])
        print(f"Cohen's D is: {cohensd}")
        print("---steering_l2")
        run_wilcoxon(df_dq[df_dq['combi'] == combi]["steering_l2"])
        cohensd = cohend(df_dq[df_dq['combi'] == combi]["steering_l2_before"], df_dq[df_dq['combi'] == combi]["steering_l2_after"])
        print(f"Cohen's D is: {cohensd}")"""
    # plot
    fig, ax1 = plt.subplots(figsize=(7.8, 5.51))
    ax2 = ax1.twiny()
    sns.set_theme(style="ticks", palette="pastel")

    v1 = sns.violinplot(ax=ax1,
                        y="combi", x="value",
                        hue=True,
                        hue_order=[True, False],
                        data=df_dqd_cte,
                        split=True,
                        inner="quart",
                        # showfliers=False,  # don't show outliers
                        # showmeans=True,
                        palette=["m"],
                        linewidth=0.5,
                        # legend=['CTE']
                        order=['RANDOM_FSC_random', 'XAI_GC_random', 'XAI_GC_attention_same']# , 'XAI_FSC_random'
                        )


    v2 = sns.violinplot(ax=ax2,
                        y="combi",x="value",

                        hue=True,
                        hue_order=[False, True],
                        data=df_dqd_steer,
                        split=True,
                        inner="quart",
                        # showfliers=False,  # don't show outliers
                        # showmeans=True,
                        palette = ["g"],
                        legend = None,
                        linewidth=0.5,
                        order=['RANDOM_FSC_random', 'XAI_GC_random', 'XAI_GC_attention_same']# , 'XAI_FSC_random'
                        )




    p1 = sns.pointplot(ax=ax1, y="combi", x="value",hue="type",
                       marker="|", markersize=20, markeredgewidth=3, linestyle="none",
                       order=['RANDOM_FSC_random', 'XAI_GC_random', 'XAI_GC_attention_same'],
                    data=df_dqd_cte, estimator=np.mean, palette = ["darkred"],legend = None)
    p2 = sns.pointplot(ax=ax2, y="combi", x="value", hue="type",
                       marker="|", markersize=20, markeredgewidth=3,linestyle="none",
                       order=['RANDOM_FSC_random', 'XAI_GC_random', 'XAI_GC_attention_same'],# , 'XAI_FSC_random'
                    data=df_dqd_steer, estimator=np.mean,color = "seagreen",legend = None)

    #fig.legend([v1,v2,p1,p2 ], labels=[['CTE'], ['steering'], ['CTE mean'], ['steering mean']], loc="upper right")

    labs = ["cte","steer","1","2"]
    ax1.set_yticklabels(['DeepJanus', 'Grad-CAM++ random', 'Grad-CAM++ high', 'Score-CAM random'],fontsize=14
                        # rotation=15
                        )

    #ax1.legend(lns, labs, loc=0)
    # Move the legend for ax1 to the right of the plot

    # ax1.legend(["CTE"],loc='lower right')
    # ax2.legend(["Steering"],loc='upper left')
    ax2.set_xlim([-0.4,1.2])
    ax1.set_xlim([-2,1.5])
    ax1.set_xticks(np.linspace(-1, 1.5, 6))
    ax2.set_xticks(np.linspace(-0.25, .75, 5))
    # ax1.set_title('Driving Quality Degradations')


    ax1.set_ylabel('')
    ax1.set_xlabel('DQD of CTEs')
    ax1.xaxis.set_label_coords(0.1, -0.025)
    ax2.set_xlabel('DQD of Steering Angles')
    ax2.xaxis.set_label_coords(0.85, 1.025)

    ax1.legend(["CTE"],bbox_to_anchor=(0.82, 0.1), loc='lower left', borderaxespad=0,fontsize=14)
    ax2.legend(["Steering"],bbox_to_anchor=(0.02, 0.9), loc='upper left', borderaxespad=0,fontsize=14)

    fig.tight_layout()
    plt.show()


def read_global_log(global_csv_path):
    # path : "simulations_simple/simulations_15_nodes_60_seeds/global_log.csv",
    global_log = pd.read_csv(global_csv_path, index_col=False)

    global_log = global_log[['Time', "episode id", "seed", 'mutation point',
                             'mutation type', 'mutation method',
                             'is success']]
    global_log['mutation point'] = pd.to_numeric(global_log['mutation point'], errors='coerce')
    return global_log


def get_episode_number(xai: str,
                       glog: pd.DataFrame,
                       exclude_limit=False,
                       save_img: bool = False):

    df_temp_failure = glog[glog["is success"] == 0]

    df_max_episode = glog.iloc[glog.groupby('Time')["episode id"].idxmax()]

    #print(df_max_episode)
    #df_max_episode = df_max_episode.drop(df_max_episode[df_max_episode["episode id"]==0].index)
    #print(df_max_episode.to_string())
    df_max_episode = df_max_episode.assign(method_combi = df_max_episode["mutation type"].values + " " + df_max_episode["mutation method"].values)
    df_max_episode = df_max_episode[['Time', 'episode id', 'is success', 'seed',  'method_combi']]
    df_max_episode['method_combi'] = df_max_episode['method_combi'].apply(lambda x: x if x == "RANDOM random" else x + ' ' + xai)

    csv_path = "../results/csvs/max_episode.csv"
    if os.path.exists(csv_path):
        max_episode_csv = pd.read_csv(csv_path)
        max_episode_csv = pd.concat([max_episode_csv, df_max_episode], ignore_index=True)
        max_episode_csv.to_csv(csv_path, index=False)
    else:
        df_max_episode.to_csv(csv_path, index=False)


def clean_data():
    """
    drop the seeds that it "once" failed at the first episode
    """
    csv_path = "../results/csvs/max_episode.csv"
    df_max_episode = pd.read_csv(csv_path)

    print(df_max_episode[df_max_episode['episode id'] == 0])
    seed_to_discard = df_max_episode[df_max_episode['episode id'] == 0]['seed'].unique()
    print(f"Seeds to be discarded are: {seed_to_discard}")

    df_max_episode_clean = df_max_episode[~df_max_episode['seed'].isin(seed_to_discard)]
    df_max_episode_clean = df_max_episode_clean.sort_values(by=['method_combi', "seed"])
    df_max_episode_clean.to_csv("../results/csvs/max_episode_clean.csv")


def do_some_statistic():
    # df_pivot_table = df_max_episode.pivot_table(index='seed', columns="method_combi", values='episode id')
    # print(df_pivot_table.to_string())

    df_table = pd.DataFrame()
    df_episode = pd.read_csv("../results/csvs/max_episode_clean.csv")

    means = df_episode.groupby('method_combi')['episode id'].mean()
    means.name = "means"
    print(f"mean values: {means}")
    df_table = pd.concat([df_table, means], axis=1)

    standard_deviations = df_episode.groupby('method_combi')['episode id'].std()
    standard_deviations.name = "standard_deviations"
    print(f"mean values: {standard_deviations}")
    df_table = pd.concat([df_table, standard_deviations], axis=1)

    DNFs = df_episode.groupby('method_combi')['is success'].sum()
    DNFs.name = "DNFs"
    print(f"mean values: {DNFs}")
    df_table = pd.concat([df_table, DNFs], axis=1)

    df_table['p_value'] = None
    df_table['Cohen D'] = None
    df_table['Cohen value'] = None
    for method in df_table.index:
        if method != 'RANDOM random':
            print(f"run wilcoxon and cohend for random and {method}")
            pvalue, cohensd = run_wilcoxon_and_cohend(df_episode[df_episode["method_combi"] == 'RANDOM random']["episode id"],
                                    df_episode[df_episode["method_combi"] == method]["episode id"])
            df_table.at[method, 'p_value'] = pvalue
            df_table.at[method, 'Cohen D'] = cohensd[0]
            df_table.at[method, 'Cohen value'] = cohensd[1]

    df_table.to_csv("../results/csvs/statistic.csv")

def cumulative_rate():
    df_episode = pd.read_csv("../results/csvs/max_episode_clean.csv")
    methods = df_episode['method_combi'].unique()
    print(methods)
    df_cumulative_failure = pd.DataFrame(index=np.linspace(1, 39, 39))
    for method in methods:
        df_temp = df_episode[df_episode["method_combi"] == method]
        # df_temp['failure'] = None
        df_temp['failure'] = df_temp['is success'].apply(lambda x: 1-x)
        cumulative_series = df_temp.groupby('episode id')['failure'].sum().cumsum()
        cumulative_series.name = method
        df_cumulative_failure = pd.concat([df_cumulative_failure, cumulative_series], axis=1)
    # fill NaN in dataframe
    df_cumulative_failure = df_cumulative_failure.fillna(method="ffill")
    print(df_cumulative_failure.to_string())
    df_cumulative_failure.to_csv("../results/csvs/cumulative_failure.csv")

    df_cumulative_failure.plot()
    plt.show()

    """
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    sns.boxplot(x="method_combi",
                y='episode id',
                data=df_episode,
                showfliers=False,  # don't show outliers
                showmeans=True,
                order=['RANDOM random', 'XAI random', 'XAI attention_opposite', 'XAI attention_same']
                )

    if save_img:
        plt.savefig("plots/boxplot_iteration_4_methods.png")
    else:
        plt.show()"""

def table_for_paper():
    df_cum = pd.read_csv("../results/csvs/cumulative_failure.csv", index_col=0)
    col_names = df_cum.columns

    df_cum_per = pd.DataFrame(columns=df_cum.columns, index=np.linspace(5, 40, 8))
    #df_cum_per.idx = np.linspace(5, 40, 8)

    for index, row in df_cum_per.iterrows():
        iteration = index - 1
        for col in col_names:
            number = df_cum[df_cum.index == iteration][col].values[0]/(60 - 5) #
            efficiency = df_cum[col].cumsum()[iteration-1] / df_cum['RANDOM random'].cumsum()[iteration-1]

            df_cum_per.at[index, col] = f"{number:.2f}-{efficiency:.2f}"

    df_cum_per.to_csv("../results/csvs/paper_table.csv")


if __name__ == "__main__":
    #timestamps = ["24-01-18-19-11", "24-01-18-21-30"]
    #enumerate_folder(timestamp=timestamps, detail=False)
    # get all subfolder in simulation_path
    """simulation_path = "./simulations_simple/simulations_15_nodes_60_seeds"
    subfolders = []
    for stuff in os.listdir(simulation_path):
        if os.path.isdir(os.path.join(simulation_path, stuff)):
            subfolders.append(stuff)
    print(subfolders)
    
    for subfolder in subfolders:
        get_dqd_data_as_csv(os.path.join(simulation_path, subfolder))"""
    # get_cte_mse("./simulations/03-18-17-47-RANDOM-seed=1")results/summary_increment_before_rearrange_cp.csv
    # get_episode_number(glog=global_log_old, exclude_limit=False)
    # dqd_plot(dqd_path = "results/summary_increment_before_rearrange_cp.csv")

    # get_episode_number(glog=global_log, exclude_limit=False)
    # dqd_plot(dqd_path = "results/summary_increment.csv")

    # get_episode_number(exclude_limit=False)

    # ----------------------------------- collect max episode -------------------------------------

    """global_log_FSC = read_global_log("../simulations_simple/simulations_15_nodes_60_seeds_FSC/global_log.csv")
    global_log_GC = read_global_log("../simulations_simple/simulations_15_nodes_60_seeds_GC/global_log.csv")
    global_log_IG = read_global_log("../simulations_simple/simulations_15_nodes_60_seeds_IG/global_log.csv")
    global_log_SM = read_global_log("../simulations_simple/simulations_15_nodes_60_seeds_SM/global_log.csv")"""

    # get_episode_number(xai="SM", glog=global_log_SM, exclude_limit=False)

    # ----------------------------------- clean data -------------------------------------

    # clean_data()

    # ----------------------------------- statistic over the max episode -------------------------------------
    #do_some_statistic()

    # ----------------------------------- cumulative rate -------------------------------------
    # cumulative_rate()

    # ----------------------------------- table for paper -------------------------------------
    # table_for_paper()

    # ----------------------------------- driving quality -------------------------------------
    """for xai in ['FSC', 'SM', 'IG', 'GC']:
        walk_folder_dqd(timestamp=None,
                    sim_dir="../simulations_simple/simulations_15_nodes_60_seeds_" + xai,
                    summary_path='../results/csvs/summary_increment_' + xai + '.csv')"""
    """for xai in ['FSC', 'SM', 'IG', 'GC']:
        walk_folder_dqd_simple(timestamp=None,
                        sim_dir="../simulations_simple/simulations_15_nodes_60_seeds_" + xai,
                        summary_path='../results/csvs/summary_driving_quality.csv')"""
    # ----------------------------------- merge dqd csvs -------------------------------------
    """df_dqd = pd.DataFrame()
    for xai in ['FSC', 'SM', 'IG', 'GC']:
        summary_path = '../results/csvs/summary_increment_' + xai + '.csv'
        df_dqd_temp = pd.read_csv(summary_path, index_col=False)
        df_dqd_temp["method combi"] = df_dqd_temp["mutation type"].values + " " + df_dqd_temp["mutation method"].values
        df_dqd_temp["method combi"] = df_dqd_temp["method combi"].apply(lambda x: x if x == "RANDOM random" else x + ' ' + xai)
        df_dqd = pd.concat([df_dqd, df_dqd_temp], ignore_index=True)
        print(len(df_dqd_temp), len(df_dqd))
    df_dqd.to_csv('../results/csvs/summary_increment_all.csv')"""

    # ----------------------------------- dqd plot -------------------------------------
    #dqd_plot_simple(dqd_path='../results/csvs/summary_increment_all.csv')
    # dqd_plot_simple2()

    dqd_plot_simple_horizontal()