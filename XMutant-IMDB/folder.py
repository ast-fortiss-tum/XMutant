import datetime
from os.path import exists, join
from os import makedirs
from config import MUTATION_RECORD, XAI_METHOD #, NUMBER


class Folder:
    def __init__(self, xai_method = XAI_METHOD):
        current = datetime.datetime.now()
        file_name = 'log' + current.strftime("%m-%d_%H-%M")
        file_name = file_name + "_" + xai_method

        #run_id = str(Timer.start.strftime('%s'))
        self.DST = "runs/" + file_name
        if not exists(self.DST):
            makedirs(self.DST)
        if MUTATION_RECORD:
            self.archive_folder = self.DST + "/archive"
            if not exists(self.archive_folder):
                makedirs(self.archive_folder)
            self.individual_folder = self.DST + "/individual_logs"
            if not exists(self.individual_folder):
                makedirs(self.individual_folder)

                # self.mutation_logdata_folder = self.mutation_logs_folder +"/data"
                # if not exists(self.mutation_logdata_folder):
                #     makedirs(self.mutation_logdata_folder)
        # self.DST_ARC = join(self.DST, "archive")
        # self.DST_IND = join(self.DST, "inds")
