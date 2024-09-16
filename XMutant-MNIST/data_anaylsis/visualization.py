import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import csv
import pandas as pd
from os.path import join
import os
import glob

from PIL import Image

REPORT_NAME = 'stats.csv'
class DataPreprocess:
    def __init__(self, dist):
        self.path = dist
        self.mutation_iterations = list()
        self.number_per_iteration = list()
        self.ids = list()
        self.population_size = 0
        self.iteration = 0
        self.misclass_number = 0
        self.read_data()
        self.preprocess()

    def read_data(self):
        csv_path = join(self.path, REPORT_NAME)
        with open(csv_path, mode='r') as report_file:
            csvreader = csv.reader(report_file)

            for rowid, row in enumerate(csvreader):
                if rowid == 1:
                    #print(row)
                    [population_size, iteration, misclass_number, self.mutation_type, _] = row
                    self.population_size = int(population_size)
                    self.iteration = int(iteration)
                    self.misclass_number = int(misclass_number)

                if rowid >= 4:
                    self.ids.append(int(row[0]))
                    if row[3] == "True":
                        self.mutation_iterations.append(int(row[5]))

                    #else:
                    #    self.mutation_iterations.append(int(iteration))

    def preprocess(self):
        for i in range(self.iteration+1):
            number = self.mutation_iterations.count(i)
            if i == 0:
                self.number_per_iteration.append(number)
            else:
                self.number_per_iteration.append(self.number_per_iteration[-1] + number)
        

        #assert (self.number_per_iteration[-1] == self.misclass_number), \
        #    f"{self.number_per_iteration[-1]} != {self.misclass_number}"


class GifMaker:
    def __init__(self, dst):
        """
        dst: ./runs/log11-17_16-32_A_R
        ./runs/log11-17_16-32_A_R/individual_logs/data
        """
        self.dst: str = dst if dst[-1] == '/' else dst + '/'

    #@staticmethod
    def make_gif(self, frame_folder, gif_path):
        #frames = glob.glob(join(frame_folder, '/*.png'))
        frames = [Image.open(image) for image in glob.glob(join(frame_folder, '*.png'))]
        print(f"Number of frames is {len(frames)} in {frame_folder}")
        if len(frames) != 0:
            frame_one = frames[0]
            frame_one.save(gif_path + ".gif", format="GIF", append_images=frames,
                           save_all=True, duration=100, loop=0)

    def make_frames(self, id: int):
        file_name = self.dst + "individual_logs/"
        csv_file = file_name + str(id) + ".csv"
        """files = glob.glob(join(file_name, '*.csv'))
        files = sorted(files)
        try:
            csv_file = files[id]
        except IndexError:
            print("Index out of range of records.")
            return"""

        df = pd.read_csv(csv_file)
        df['mutation_id'] = df['mutation_id'].astype(int)
        df['confidence'] = df['confidence'].astype(float)
        df['mutation_point_x'] = df['mutation_point_x'].astype(float)
        df['mutation_point_y'] = df['mutation_point_y'].astype(float)

        #print(df['mutation_id'][1], df['file'][1], df['confidence'][1], df['mutation_point_x'][1],
        #      df['mutation_point_y'][1])
        #print(df['confidence'].shape[0])

        data = DataPreprocess(self.dst)

        mutation_num = int(data.mutation_iterations[data.ids.index(id)])
        print(mutation_num)
        for num in range(mutation_num):
            self.make_frame(df['mutation_id'][num], df['file'][num].replace("runs", "../../../dataset/flowchart"), df['confidence'].to_numpy(),
                            df['mutation_point_x'][num],
                            df['mutation_point_y'][num])

    def make_frame(self, mutate_id, npy_file, confidences, x, y):
        # TODO if attention base.
        
        with open(npy_file, 'rb') as f:
            image = np.load(f)
            attention_image = np.load(f)

        fig = plt.figure(figsize=(9, 10))
        gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1, 1])
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(image.reshape(28, 28), cmap="gray")
        ax0.set_title("MNIST Mutation", color="red")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.plot(x, y, marker='*', markersize=15, color="y")

        ax1 = fig.add_subplot(gs[0, 1])
        sns.heatmap(attention_image.reshape(28, 28), ax=ax1, square=True, cmap="rocket")
        ax1.set_title("Heat Map", color="blue")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.plot(x, y, marker='*', markersize=15, color="y")

        ax_confidence = fig.add_subplot(gs[1, :])
        generations = range(confidences.shape[0])
        ax_confidence.plot(generations, confidences)
        ax_confidence.plot(generations[mutate_id-1], confidences[mutate_id-1], marker="o", markeredgecolor="blue")
        ax_confidence.set_ylim(-0.7, 1.1)
        ax_confidence.set_xlim(0, 1000)
        #plt.show()
        plt.savefig(npy_file.replace("npy", "png"))
        fig.clf()
        plt.close(fig)

    def read_img(self, file_name):
        # file_name = self.dst + '/individual_logs/data/ID' + str(id) + '_GEN' + str(gen_id) + ".npy"
        try:
            with open(file_name, 'rb') as f:
                img = np.load(f)
                attention_map = np.load(f)
                return img, attention_map
        except FileNotFoundError:
            print(f"oops, there is no file named {file_name}.")


def hist_plot(mutation_iterations, iteration):
    bins = 20
    plt.hist(mutation_iterations, bins=bins)  # , color='skyblue', edgecolor='black')
    # plt.xticks(np.linspace(0, iteration, bins))
    plt.xlabel('Mutation iterations')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mutation iteration')

def line_plot(number_per_iteration, title=""):
    plt.plot(number_per_iteration)
    plt.xlabel('Mutation iterations')
    plt.ylabel('Number of misbehaviours')
    plt.title(title)

def stat_1_config(xai: str, method: str, length=1000):
    end_name = method + xai
    dst = "./runs/" #"/media/xchen/Elements/Xmutant/"

    folders = os.listdir(dst)

    df = pd.DataFrame()


    for i in range(10):

        fs = [f for f in folders if f.endswith(str(i) + end_name)]
        assert len(fs) >= 1, "no such folder end with " +str(i) + end_name
        if len(fs) > 1:
            print("Warning more than one folder")
        dst_digit_i = fs[0]
        print(f"dst_digit_i {dst_digit_i}")
        

        """for j, name in enumerate(namelist):
            data_digit = DataPreprocess(dst + dst_digit_i[j])
            line_plot(data_digit.number_per_iteration, 'Line plot of Mutation iteration of digit '+str(i))
            data_digits.append(data_digit)
        plt.legend([str(i)+"_"+name for name in namelist])"""
        data_digit = DataPreprocess(os.path.join(dst, dst_digit_i))
        
        if len(data_digit.number_per_iteration)<length+1:
            last = data_digit.number_per_iteration[-1]
            print(last)
            array = np.concatenate((data_digit.number_per_iteration, 
                                   np.full(length-len(data_digit.number_per_iteration)+1, last)))
            df[str(i)] = array
        else:
            df[str(i)] = data_digit.number_per_iteration

        #line_plot(data_digit.number_per_iteration, 'Line plot of Mutation iteration of digit '+str(i))


        """if if_save:
            plt.savefig("./result/line_plot_digit" + str(i) + ".png")
            plt.close()
        else:
            plt.show()"""
        
    df['total'] =  df.sum(axis=1)
    df.to_csv(os.path.join("result", "csv_folder", "sum" + end_name + ".csv"))

def main_stat():
    xais = ["_sm", "_GC", "_FSC", "_IG"]
    methods = ["_S_R", "_C_R", "_C_C"]
   
    for xai in xais:
        for method in methods:
            stat_1_config(xai, method)

    stat_1_config("_R", "_R")


def main_plot(if_save=False):
    #col_name = [str(i) for i in range(10)] + ['total']
    #print(col_name)

    name_list = glob.glob(os.path.join("../result", "csv_folder", "cumulative_clear_*.csv"))
    name_list.sort(key=str.lower)

    #change order of r_r
    to_be_moved = [i for i in name_list if "R_R" in i]
    name_list.remove(to_be_moved[0])
    name_list.append(to_be_moved[0])

    print(name_list)

    col_names = [("_").join(name.split("/")[-1][:-4].split("_")[5:]) for name in name_list]

    df = pd.DataFrame(columns=['idx'] + col_names )
    df.idx = np.linspace(1,1000,1000)
    #for col in col_name:
    plt.figure(figsize=(16, 10), dpi=100)
    for col, csv_file in zip(col_names, name_list):
        df_temp = pd.read_csv(csv_file)
        if col[0:3] == "C_C":
            plt.plot(df_temp['idx'], df_temp['pop_cum_num']/(2000 -23), "--")
        elif col[0:3] == "R_R":
            plt.plot(df_temp['idx'], df_temp['pop_cum_num']/(2000 -23), "-.")
        else:
            plt.plot(df_temp['idx'], df_temp['pop_cum_num']/(2000 -23), "-")
        # line_plot(df_temp[col], title='Line plot of Mutation iteration of digit '+col)
        for index, row in df.iterrows():
            #print(row['idx'])

            df_temp_idx = df_temp['idx'][df_temp['idx']<=row['idx']].max()

            df.at[index, col] = df_temp[df_temp['idx'] == df_temp_idx]['pop_cum_num'].values[0]

    #  '../result/csv_folder/cumulative_clear_validity_rate_record_R_R.csv'
    plt.legend(col_names)
    plt.xlabel('Iteration')
    plt.xlabel('Failure Rate')
    plt.title("MNIST Failure Rate over Iteration")
    plt.show()

    if if_save:
        plt.savefig(os.path.join("result", 'figure', "line_plot_digit_" + col + ".png"))
        plt.close()
    else:
        plt.show()

    df.to_csv(os.path.join("result", 'csv_folder', "cumulative_misclassified_all.csv"))
    df.plot(x='idx',)
def integration(name_list, cut_off_id):
    return np.sum(name_list[:, :cut_off_id], axis=1)


def effective_heat_map(if_save = False, length=1000):
    col_name = [str(i) for i in range(10)] + ['total']
    print(col_name)

    name_list = glob.glob(os.path.join("result", "csv_folder", "sum_*.csv"))
    name_list.sort(key=str.lower)
    to_be_moved = [i for i in name_list if "R_R" in i]
    name_list.remove(to_be_moved[0])
    name_list.append(to_be_moved[0])
    print(name_list)

    cut_off_nodes = range(100, 1100, 100)

    data_2d = np.zeros(shape=(len(cut_off_nodes), len(name_list)))

    data_raw = np.zeros(shape=(len(name_list), length+1))

    for i, csv_file in enumerate(name_list):
        df_temp = pd.read_csv(csv_file)
        data_raw[i] = df_temp["total"].to_list()

    for i, cut_off_id in enumerate(cut_off_nodes):
        inte_result = integration(data_raw, cut_off_id)
        data_2d[i] = inte_result/inte_result[-1]
        print(cut_off_id, inte_result/inte_result[-1])

    xtick_labels = [name.split("/")[-1][4:-4] for name in name_list[:-1]]

    plt.figure(figsize=(14, 10), dpi=100)
    s = sns.heatmap(data_2d[:, :-1],
                    xticklabels=xtick_labels,
                    yticklabels=cut_off_nodes,
                    annot=True, fmt='.2f',)
    
    s.set(xlabel="Methods", ylabel="Number of mutations",
          title="Relative Efficiency to the random mutation")

    if if_save:
        plt.savefig(os.path.join("result", 'figure',"Relative Efficiency Map"  + ".png"))
        plt.close()
    else:
        plt.show()


def main_gif(dst):
    gif = GifMaker(dst)

    # produce frames
    path = gif.dst + "individual_logs/"
    print(path)
    files = glob.glob(join(path, '*.csv'))
    files = sorted(files)
    print(files)
    for img_id in range(len(files)):
        gif.make_frames(img_id)

    # produce gif
    # TODO frame order seems incorrect
    # frame_folder = "./runs/log11-20_10-42_A_R/individual_logs/data/ID5/"
    # print(os.listdir(frame_folder))
    # gif_path = "./runs/log11-20_10-42_A_R/individual_logs/data"
    # gif.make_gif(frame_folder, gif_path)

def main_gif_temp(dst):
    gif = GifMaker(dst)

    # produce frames
    path = gif.dst + "individual_logs/"
    print(path)
    for i in range(200):
        gif.make_frames(i)

    # produce gif
    # TODO frame order seems incorrect
    # frame_folder = "./runs/log11-20_10-42_A_R/individual_logs/data/ID5/"
    # print(os.listdir(frame_folder))
    # gif_path = "./runs/log11-20_10-42_A_R/individual_logs/data"
    # gif.make_gif(frame_folder, gif_path)


def validity_plot():
    sm_csv_path = "../result/csv_folder/cumulative_clear_validity_rate_record_C_C_sm.csv"
    gc_csv_path = "../result/csv_folder/cumulative_clear_validity_rate_record_C_C_GC.csv"
    fsc_csv_path = "../result/csv_folder/cumulative_clear_validity_rate_record_C_C_FSC.csv"
    random_csv_path = "../result/csv_folder/cumulative_clear_validity_rate_record_R_R.csv"
    df_xai_sm = pd.read_csv(sm_csv_path)
    df_xai_gc = pd.read_csv(gc_csv_path)
    df_xai_fsc = pd.read_csv(fsc_csv_path)
    df_random = pd.read_csv(random_csv_path)

    plt.figure(figsize=(6,2),dpi=200)
    # plt.subplot(2,1,1)

    plt.plot(df_xai_sm["idx"], df_xai_sm["ID_cum_rate_0.95"])
    plt.plot(df_xai_gc["idx"], df_xai_gc["ID_cum_rate_0.95"])
    plt.plot(df_xai_fsc["idx"], df_xai_fsc["ID_cum_rate_0.95"])
    plt.plot(df_random["idx"], df_random["ID_cum_rate_0.95"], "k--")

    plt.title("cumulative validity rate per iteration by VAE validator")
    plt.ylabel("validity rate")
    plt.ylim([0.68,1.02])
    plt.xlabel("Iteration")
    plt.legend(["SmoothGrad", "Grad-CAM++", "Score-CAM", "Random"], bbox_to_anchor = (1.,0.5), loc='center left')
    """plt.subplot(2,1,2)
    plt.plot(df_xai["idx"], df_xai["ID_rate_0.95"],"*",alpha=0.2)
    plt.plot(df_random["idx"], df_random["ID_rate_0.95"],"o",alpha=0.2)
    plt.title("validity rate per iteration")
    plt.ylabel("validity rate")
    plt.xlabel("Iteration")
    plt.legend(["XAI", "Random"])
    plt.tight_layout()"""
    plt.show()
    #plt.savefig("./result/figure/validity_plot.png")

def two_plot():
    plt.figure(figsize=(6, 4), dpi=200)
    plt.tight_layout()
    plt.subplot(2,1,1)
    sm_csv_path = "../result/csv_folder/cumulative_clear_validity_rate_record_C_C_sm.csv"
    gc_csv_path = "../result/csv_folder/cumulative_clear_validity_rate_record_C_C_GC.csv"
    fsc_csv_path = "../result/csv_folder/cumulative_clear_validity_rate_record_C_C_FSC.csv"
    random_csv_path = "../result/csv_folder/cumulative_clear_validity_rate_record_R_R.csv"
    df_xai_sm = pd.read_csv(sm_csv_path)
    df_xai_gc = pd.read_csv(gc_csv_path)
    df_xai_fsc = pd.read_csv(fsc_csv_path)
    df_random = pd.read_csv(random_csv_path)



    plt.plot(df_xai_sm["idx"], df_xai_sm["ID_cum_rate_0.95"])
    plt.plot(df_xai_gc["idx"], df_xai_gc["ID_cum_rate_0.95"])
    # plt.plot(df_xai_fsc["idx"], df_xai_fsc["ID_cum_rate_0.95"])
    plt.plot(df_random["idx"], df_random["ID_cum_rate_0.95"], "k--")

    plt.title("MNIST: Validity by SelfOracle")
    #plt.ylabel("validity rate")
    plt.ylim([0.68, 1.02])
    #plt.xlabel("Iteration")
    plt.legend(["SmoothGrad \nvalidity rate", "Grad-CAM++\nvalidity rate",
                #"Score-CAM\nvalidity rate",
                "DeepJanus\nvalidity rate"], bbox_to_anchor=(1., 0.5), loc='center left')


    """plt.subplot(2,1,2)
    plt.plot(df_xai["idx"], df_xai["ID_rate_0.95"],"*",alpha=0.2)
    plt.plot(df_random["idx"], df_random["ID_rate_0.95"],"o",alpha=0.2)
    plt.title("validity rate per iteration")
    plt.ylabel("validity rate")
    plt.xlabel("Iteration")
    plt.legend(["XAI", "Random"])
    plt.tight_layout()"""
    # plt.show()
    plt.subplot(2, 1, 2)
    names = [] # Name list of testers

    df_human = pd.read_csv("../result/human assessment/log.csv")
    df_human = df_human.drop(columns=df_human.columns[df_human.columns.str.contains('Unnamed')])
    df_human = df_human[df_human['mutation_number'] != 1]
    df_human['mutation_number'] = df_human['mutation_number'] - 1
    df_xai = df_human[df_human['method'] == "XAI"]
    df_xai['one'] = 1
    df_random = df_human[df_human['method'] == "Random"]
    df_random['one'] = 1

    df_cumulative_xai = pd.DataFrame()
    df_cumulative_xai["idx"] = df_xai.groupby('mutation_number')['method'].sum().to_frame().index
    df_cumulative_xai["pop_num"] = df_xai.groupby('mutation_number')['one'].sum().cumsum().to_list()
    for name in names:
        # df_cumulative_xai[name + "_validity"] = df_xai.groupby('mutation_number')['validity_' + name].sum().cumsum().to_list()
        df_cumulative_xai[name + "_validity_rate"] = df_xai.groupby('mutation_number')[
                                                         'validity_' + name].sum().cumsum().to_list() / \
                                                     df_cumulative_xai["pop_num"]
        # df_cumulative_xai[name + "_preservation"] = df_xai.groupby('mutation_number')['preservation_' + name].sum().cumsum().to_list()
        df_cumulative_xai[name + "_preservation_rate"] = df_xai.groupby('mutation_number')[
                                                             'preservation_' + name].sum().cumsum().to_list() / \
                                                         df_cumulative_xai["pop_num"]

    df_cumulative_xai["mean_validity_rate"] = df_cumulative_xai[
        df_cumulative_xai.columns[df_cumulative_xai.columns.str.contains('validity_rate')]].mean(axis=1)
    df_cumulative_xai["std_validity_rate"] = df_cumulative_xai[
        df_cumulative_xai.columns[df_cumulative_xai.columns.str.contains('validity_rate')]].std(axis=1)
    df_cumulative_xai["mean_preservation_rate"] = df_cumulative_xai[
        df_cumulative_xai.columns[df_cumulative_xai.columns.str.contains('preservation_rate')]].mean(axis=1)
    df_cumulative_xai["std_preservation_rate"] = df_cumulative_xai[
        df_cumulative_xai.columns[df_cumulative_xai.columns.str.contains('preservation_rate')]].std(axis=1)

    print(df_cumulative_xai.to_string())

    df_cumulative_random = pd.DataFrame()
    df_cumulative_random["idx"] = df_random.groupby('mutation_number')['method'].sum().to_frame().index
    df_cumulative_random["pop_num"] = df_random.groupby('mutation_number')['one'].sum().cumsum().to_list()
    for name in names:
        # df_cumulative_random[name + "_validity"] = df_random.groupby('mutation_number')['validity_' + name].sum().cumsum().to_list()
        df_cumulative_random[name + "_validity_rate"] = df_random.groupby('mutation_number')[
                                                            'validity_' + name].sum().cumsum().to_list() / \
                                                        df_cumulative_random["pop_num"]
        # df_cumulative_random[name + "_preservation"] = df_random.groupby('mutation_number')['preservation_' + name].sum().cumsum().to_list()
        df_cumulative_random[name + "_preservation_rate"] = df_random.groupby('mutation_number')[
                                                                'preservation_' + name].sum().cumsum().to_list() / \
                                                            df_cumulative_random["pop_num"]
    df_cumulative_random["mean_validity_rate"] = df_cumulative_random[
        df_cumulative_random.columns[df_cumulative_random.columns.str.contains('validity_rate')]].mean(axis=1)
    df_cumulative_random["std_validity_rate"] = df_cumulative_random[
        df_cumulative_random.columns[df_cumulative_random.columns.str.contains('validity_rate')]].std(axis=1)
    df_cumulative_random["mean_preservation_rate"] = df_cumulative_random[
        df_cumulative_random.columns[df_cumulative_random.columns.str.contains('preservation_rate')]].mean(axis=1)
    df_cumulative_random["std_preservation_rate"] = df_cumulative_random[
        df_cumulative_random.columns[df_cumulative_random.columns.str.contains('preservation_rate')]].std(axis=1)

    print(df_cumulative_random)

    # plt.figure(figsize=(5, 2), dpi=200)
    plt.plot(df_cumulative_xai['idx'].to_list() + [999],
             df_cumulative_xai['mean_validity_rate'].to_list() + [
                 df_cumulative_xai['mean_validity_rate'].to_list()[-1]],
             linestyle="solid", color='#CC4F1B')
    plt.plot(df_cumulative_random['idx'].to_list() + [999],
             df_cumulative_random['mean_validity_rate'].to_list() + [
                 df_cumulative_random['mean_validity_rate'].to_list()[-1]],
             linestyle="solid",
             color='#1B2ACC')

    plt.plot(df_cumulative_xai['idx'].to_list() + [999],
             df_cumulative_xai['mean_preservation_rate'].to_list() + [
                 df_cumulative_xai['mean_preservation_rate'].to_list()[-1]],
             linestyle="dashed", color='#CC4F1B')
    plt.plot(df_cumulative_random['idx'].to_list() + [999],
             df_cumulative_random['mean_preservation_rate'].to_list() + [
                 df_cumulative_random['mean_preservation_rate'].to_list()[-1]],
             linestyle="dashed", color='#1B2ACC')

    """plt.errorbar(df_cumulative_xai['idx'], df_cumulative_xai['mean_validity_rate'],
                 yerr=df_cumulative_xai['std_validity_rate'], linestyle='dotted', label="xai")
    plt.errorbar(df_cumulative_random['idx'], df_cumulative_random['mean_validity_rate'],
                 yerr=df_cumulative_random['std_validity_rate'],  linestyle='dotted', label="random")
    plt.errorbar(df_cumulative_xai['idx'], df_cumulative_xai['mean_preservation_rate'],
                 yerr=df_cumulative_xai['std_preservation_rate'], linestyle='dashed', label="xai")
    plt.errorbar(df_cumulative_random['idx'], df_cumulative_random['mean_preservation_rate'],
                 yerr=df_cumulative_random['std_preservation_rate'], linestyle='dashed', label="random")"""

    plt.legend(["XMutant \nvalidity rate", "DeepJanus \nvalidity rate", "XMutant \npreservation rate",
                "DeepJanus \npreservation rate"], bbox_to_anchor=(1., 0.5),
               # fontsize="12",
               loc='center left')

    x_list = df_cumulative_xai['idx'].to_list() + [999]
    y1_list = (df_cumulative_xai['mean_validity_rate'] - df_cumulative_xai['std_validity_rate']).to_list() + [
        (df_cumulative_xai['mean_validity_rate'] - df_cumulative_xai['std_validity_rate']).to_list()[-1]]
    y2_list = (df_cumulative_xai['mean_validity_rate'] + df_cumulative_xai['std_validity_rate']).to_list() + [
        (df_cumulative_xai['mean_validity_rate'] + df_cumulative_xai['std_validity_rate']).to_list()[-1]]
    plt.fill_between(x_list,
                     y1_list,
                     y2_list,
                     alpha=0.4, edgecolor='#CC4F1B', facecolor='#FF9848', linewidth=0)

    x_list = df_cumulative_random['idx'].to_list() + [999]
    y1_list = (df_cumulative_random['mean_validity_rate'] - df_cumulative_random['std_validity_rate']).to_list() + [
        (df_cumulative_random['mean_validity_rate'] - df_cumulative_random['std_validity_rate']).to_list()[-1]]
    y2_list = (df_cumulative_random['mean_validity_rate'] + df_cumulative_random['std_validity_rate']).to_list() + [
        (df_cumulative_random['mean_validity_rate'] + df_cumulative_random['std_validity_rate']).to_list()[-1]]
    plt.fill_between(x_list,
                     y1_list,
                     y2_list,
                     alpha=0.4, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=0)

    x_list = df_cumulative_xai['idx'].to_list() + [999]
    y1_list = (df_cumulative_xai['mean_preservation_rate'] - df_cumulative_xai['std_preservation_rate']).to_list() + [
        (df_cumulative_xai['mean_preservation_rate'] - df_cumulative_xai['std_preservation_rate']).to_list()[-1]]
    y2_list = (df_cumulative_xai['mean_preservation_rate'] + df_cumulative_xai['std_preservation_rate']).to_list() + [
        (df_cumulative_xai['mean_preservation_rate'] + df_cumulative_xai['std_preservation_rate']).to_list()[-1]]
    plt.fill_between(x_list,
                     y1_list,
                     y2_list,
                     alpha=0.4, edgecolor='#CC4F1B', facecolor='#FF9848', linewidth=0)

    x_list = df_cumulative_random['idx'].to_list() + [999]
    y1_list = (df_cumulative_random['mean_preservation_rate'] - df_cumulative_random[
        'std_preservation_rate']).to_list() + [(df_cumulative_random['mean_preservation_rate'] - df_cumulative_random[
        'std_preservation_rate']).to_list()[-1]]
    y2_list = (df_cumulative_random['mean_preservation_rate'] + df_cumulative_random[
        'std_preservation_rate']).to_list() + [(df_cumulative_random['mean_preservation_rate'] + df_cumulative_random[
        'std_preservation_rate']).to_list()[-1]]
    plt.fill_between(x_list,
                     y1_list,
                     y2_list,
                     alpha=0.4, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=0)

    plt.ylim([0.3, 1.01])
    #plt.xlim([0, 1005])
    plt.title("MNIST: Validity by Human Assessments")
    plt.xlabel('iteration')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(os.getcwd())
    print(os.path.exists("../result/csv_folder/cumulative_clear_validity_rate_record_C_C_sm.csv"))
    # if os.getcwd().split("/")[-1] != "XMutant-MNIST":
    #     print("change directory")
    #     os.chdir("../")


    #main_stat()

    #main_plot(if_save=False)
    
    #effective_heat_map(if_save=True)
    #validity_plot()


    #dst = "../../../dataset/flowchart/log03-14_23-12_5_C_C_sm"
    #main_gif_temp(dst)

    two_plot()
