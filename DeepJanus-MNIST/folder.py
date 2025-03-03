from timer import Timer
from os.path import exists, join
from os import makedirs
from config import EXPLABEL, NGEN,DATALOADER, XMUTANT_CONFIG, ARCHIVE_THRESHOLD, DATASET_NAME
from datetime import datetime

class Folder:
    run_id = Timer.start.strftime('%m-%d_%H-%M')#str(Timer.start.strftime('%s'))
    if DATALOADER == 'xmutant':
        DST = f"runs/{DATASET_NAME}_{run_id}_label_{EXPLABEL}_gen_{NGEN}_AT_{ARCHIVE_THRESHOLD}_{XMUTANT_CONFIG['xai']}"
    else:
        DST = "runs/run_" + run_id
    if not exists(DST):
        makedirs(DST)
    DST_ARC = join(DST, "archive")
    DST_IND = join(DST, "inds")
