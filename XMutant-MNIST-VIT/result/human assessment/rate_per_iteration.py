import matplotlib.pyplot as plt
import pandas as pd
import os

def trace_mutation_number():
    print(os.getcwd())

    human_log_file = "log.csv"
    C_C_sm_file = "../digits/record_C_C_sm.csv"
    R_R_file = "../digits/record_R_R.csv"

    df_human = pd.read_csv(human_log_file)
    df_human = df_human.drop(columns=df_human.columns[df_human.columns.str.contains('Unnamed')])
    df_c_c = pd.read_csv(C_C_sm_file)
    df_R_R = pd.read_csv(R_R_file)
    #df_human['short_path'] = None

    for idx, row in df_human.iterrows():
        path = row["path"]
        short_path = '/'.join(path.split('/')[1:])[:-4]
        df_human.at[idx, 'short_path'] = short_path
        cc = df_c_c[df_c_c['image_path'].str.contains(short_path)]
        rr = df_R_R[df_R_R['image_path'].str.contains(short_path)]

        if cc.empty and not rr.empty:
            df_human.at[idx, 'mutation_number'] = rr['mutation_number'].values[0]
        elif not cc.empty and rr.empty:
            df_human.at[idx, 'mutation_number'] = cc['mutation_number'].values[0]
        else:
            print("something wrong")

    print(df_human)
    df_human.to_csv(human_log_file)


def main():
    names = ["AS",
               "Stefano",
               "Thao",
                "Davide",
             "xiang",
             "Shangsu",
             "xin",
             "Linfeng",
             "yuting",
             "He"]

    df_human = pd.read_csv("log.csv")
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
        #df_cumulative_xai[name + "_validity"] = df_xai.groupby('mutation_number')['validity_' + name].sum().cumsum().to_list()
        df_cumulative_xai[name + "_validity_rate"] = df_xai.groupby('mutation_number')['validity_' + name].sum().cumsum().to_list() / df_cumulative_xai["pop_num"]
        #df_cumulative_xai[name + "_preservation"] = df_xai.groupby('mutation_number')['preservation_' + name].sum().cumsum().to_list()
        df_cumulative_xai[name + "_preservation_rate"] = df_xai.groupby('mutation_number')['preservation_' + name].sum().cumsum().to_list()/ df_cumulative_xai["pop_num"]

    df_cumulative_xai["mean_validity_rate"] = df_cumulative_xai[df_cumulative_xai.columns[df_cumulative_xai.columns.str.contains('validity_rate')]].mean(axis=1)
    df_cumulative_xai["std_validity_rate"] = df_cumulative_xai[df_cumulative_xai.columns[df_cumulative_xai.columns.str.contains('validity_rate')]].std(axis=1)
    df_cumulative_xai["mean_preservation_rate"] = df_cumulative_xai[
        df_cumulative_xai.columns[df_cumulative_xai.columns.str.contains('preservation_rate')]].mean(axis=1)
    df_cumulative_xai["std_preservation_rate"] = df_cumulative_xai[
        df_cumulative_xai.columns[df_cumulative_xai.columns.str.contains('preservation_rate')]].std(axis=1)


    print(df_cumulative_xai.to_string())

    df_cumulative_random = pd.DataFrame()
    df_cumulative_random["idx"] = df_random.groupby('mutation_number')['method'].sum().to_frame().index
    df_cumulative_random["pop_num"] = df_random.groupby('mutation_number')['one'].sum().cumsum().to_list()
    for name in names:
        #df_cumulative_random[name + "_validity"] = df_random.groupby('mutation_number')['validity_' + name].sum().cumsum().to_list()
        df_cumulative_random[name + "_validity_rate"] = df_random.groupby('mutation_number')['validity_' + name].sum().cumsum().to_list() /df_cumulative_random["pop_num"]
        #df_cumulative_random[name + "_preservation"] = df_random.groupby('mutation_number')['preservation_' + name].sum().cumsum().to_list()
        df_cumulative_random[name + "_preservation_rate"] = df_random.groupby('mutation_number')['preservation_' + name].sum().cumsum().to_list() / df_cumulative_random["pop_num"]
    df_cumulative_random["mean_validity_rate"] = df_cumulative_random[df_cumulative_random.columns[df_cumulative_random.columns.str.contains('validity_rate')]].mean(axis=1)
    df_cumulative_random["std_validity_rate"] = df_cumulative_random[df_cumulative_random.columns[df_cumulative_random.columns.str.contains('validity_rate')]].std(axis=1)
    df_cumulative_random["mean_preservation_rate"] = df_cumulative_random[
        df_cumulative_random.columns[df_cumulative_random.columns.str.contains('preservation_rate')]].mean(axis=1)
    df_cumulative_random["std_preservation_rate"] = df_cumulative_random[
        df_cumulative_random.columns[df_cumulative_random.columns.str.contains('preservation_rate')]].std(axis=1)

    print(df_cumulative_random)

    plt.figure(figsize=(5, 2), dpi=200)
    plt.plot(df_cumulative_xai['idx'].to_list()+[999],
             df_cumulative_xai['mean_validity_rate'].to_list()+[df_cumulative_xai['mean_validity_rate'].to_list()[-1]],
             linestyle="solid", color='#CC4F1B')
    plt.plot(df_cumulative_random['idx'].to_list()+[999],
             df_cumulative_random['mean_validity_rate'].to_list()+[df_cumulative_random['mean_validity_rate'].to_list()[-1]],
             linestyle="solid",
             color='#1B2ACC')

    plt.plot(df_cumulative_xai['idx'].to_list()+[999],
             df_cumulative_xai['mean_preservation_rate'].to_list()+[df_cumulative_xai['mean_preservation_rate'].to_list()[-1]],
             linestyle="dashed", color='#CC4F1B')
    plt.plot(df_cumulative_random['idx'].to_list()+[999],
             df_cumulative_random['mean_preservation_rate'].to_list()+[df_cumulative_random['mean_preservation_rate'].to_list()[-1]],
             linestyle="dashed", color='#1B2ACC')


    """plt.errorbar(df_cumulative_xai['idx'], df_cumulative_xai['mean_validity_rate'],
                 yerr=df_cumulative_xai['std_validity_rate'], linestyle='dotted', label="xai")
    plt.errorbar(df_cumulative_random['idx'], df_cumulative_random['mean_validity_rate'],
                 yerr=df_cumulative_random['std_validity_rate'],  linestyle='dotted', label="random")
    plt.errorbar(df_cumulative_xai['idx'], df_cumulative_xai['mean_preservation_rate'],
                 yerr=df_cumulative_xai['std_preservation_rate'], linestyle='dashed', label="xai")
    plt.errorbar(df_cumulative_random['idx'], df_cumulative_random['mean_preservation_rate'],
                 yerr=df_cumulative_random['std_preservation_rate'], linestyle='dashed', label="random")"""

    plt.legend(["XMutant \nvalidity rate", "Random \nvalidity rate", "XMutant \npreservation rate", "Random \npreservation rate"], bbox_to_anchor=(1., 0.5),
               fontsize="12",
               loc='center left')

    x_list = df_cumulative_xai['idx'].to_list()+[999]
    y1_list = (df_cumulative_xai['mean_validity_rate']-df_cumulative_xai['std_validity_rate']).to_list() + [(df_cumulative_xai['mean_validity_rate']-df_cumulative_xai['std_validity_rate']).to_list()[-1]]
    y2_list = (df_cumulative_xai['mean_validity_rate']+df_cumulative_xai['std_validity_rate']).to_list() + [(df_cumulative_xai['mean_validity_rate']+df_cumulative_xai['std_validity_rate']).to_list()[-1]]
    plt.fill_between(x_list,
                     y1_list,
                     y2_list,
                     alpha=0.4, edgecolor='#CC4F1B', facecolor='#FF9848', linewidth=0)

    x_list = df_cumulative_random['idx'].to_list()+[999]
    y1_list = (df_cumulative_random['mean_validity_rate']-df_cumulative_random['std_validity_rate']).to_list() + [(df_cumulative_random['mean_validity_rate']-df_cumulative_random['std_validity_rate']).to_list()[-1]]
    y2_list = (df_cumulative_random['mean_validity_rate']+df_cumulative_random['std_validity_rate']).to_list() + [(df_cumulative_random['mean_validity_rate']+df_cumulative_random['std_validity_rate']).to_list()[-1]]
    plt.fill_between(x_list,
                     y1_list,
                     y2_list,
                     alpha=0.4, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=0)

    x_list = df_cumulative_xai['idx'].to_list()+[999]
    y1_list = (df_cumulative_xai['mean_preservation_rate']-df_cumulative_xai['std_preservation_rate']).to_list() + [(df_cumulative_xai['mean_preservation_rate']-df_cumulative_xai['std_preservation_rate']).to_list()[-1]]
    y2_list = (df_cumulative_xai['mean_preservation_rate']+df_cumulative_xai['std_preservation_rate']).to_list() + [(df_cumulative_xai['mean_preservation_rate']+df_cumulative_xai['std_preservation_rate']).to_list()[-1]]
    plt.fill_between(x_list,
                     y1_list,
                     y2_list,
                     alpha=0.4, edgecolor='#CC4F1B', facecolor='#FF9848', linewidth=0)

    x_list = df_cumulative_random['idx'].to_list()+[999]
    y1_list = (df_cumulative_random['mean_preservation_rate']-df_cumulative_random['std_preservation_rate']).to_list() + [(df_cumulative_random['mean_preservation_rate']-df_cumulative_random['std_preservation_rate']).to_list()[-1]]
    y2_list = (df_cumulative_random['mean_preservation_rate']+df_cumulative_random['std_preservation_rate']).to_list() + [(df_cumulative_random['mean_preservation_rate']+df_cumulative_random['std_preservation_rate']).to_list()[-1]]
    plt.fill_between(x_list,
                     y1_list,
                     y2_list,
                     alpha=0.4, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=0)

    plt.ylim([0.3,1.01])
    plt.xlim([0,1005])
    plt.title("Results of human assessments")
    plt.xlabel('iteration')
    plt.show()


if __name__ == "__main__":
    # step 1
    trace_mutation_number()
    # Step 2
    main()
