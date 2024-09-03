import datetime
from os.path import exists, join
from os import makedirs
from config import MUTATION_RECORD, CONTROL_POINT, MUTATION_TYPE #, NUMBER


class Folder:
    def __init__(self, num, xai_method):
        current = datetime.datetime.now()
        file_name = 'log' + current.strftime("%m-%d_%H-%M")
        file_name = file_name + "_" + str(num)

        if CONTROL_POINT == "random":
            file_name = file_name + "_R"
        elif CONTROL_POINT == "square-window":
            file_name = file_name + "_S"
        elif CONTROL_POINT == "clustering":
            file_name = file_name + "_C"

        if MUTATION_TYPE in ["random", "random_cycle"]:
            file_name = file_name + "_R"
        elif MUTATION_TYPE in ["toward_centroid", "backward_centroid", "centroid_based"]:
            file_name = file_name + "_C"

        if xai_method == "SmoothGrad":
            file_name = file_name + "_sm"
        elif xai_method == "GradCAM++":
            file_name = file_name + "_GC"
        elif xai_method == "Faster-ScoreCAM":
            file_name = file_name + "_FSC"
        elif xai_method == "IntegratedGradients":
            file_name = file_name + "_IG"    

        #run_id = str(Timer.start.strftime('%s'))
        self.DST = "runs/" + file_name
        if not exists(self.DST):
            makedirs(self.DST)
        if MUTATION_RECORD:
            self.mutation_logs_folder = self.DST + "/individual_logs"
            if not exists(self.mutation_logs_folder):
                makedirs(self.mutation_logs_folder)
                self.mutation_logdata_folder = self.mutation_logs_folder+"/data"
                if not exists(self.mutation_logdata_folder):
                    makedirs(self.mutation_logdata_folder)
        self.DST_ARC = join(self.DST, "archive")
        self.DST_IND = join(self.DST, "inds")
