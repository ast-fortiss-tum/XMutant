import os
import subprocess

labels = list(range(10))

archive_thresholds = [0.25, 1, 4]

xmutant_configs = [{"xai": 'none', "selection": "random", "direction": "random"},
                   {"xai": "SmoothGrad", "selection": "clustering", "direction": "random_cycle"},
                   {"xai": "SmoothGrad", "selection": "clustering", "direction": "toward_centroid"}
                    ]


for conf in xmutant_configs:
    for archive_thres in archive_thresholds:
        for label in range(10):
            os.environ['EXPECT_LABEL'] = str(label)
            os.environ['AR_THRES'] = str(archive_thres)
            print(conf)
            os.environ['XAI'] = conf['xai']
            os.environ['SELECTION'] = conf['selection']
            os.environ['DIRECTION'] = conf['direction']
            print(f"===========================Running config {conf} with label {label} archive_threshold {archive_thres} ============================")
            subprocess.run(['python', 'main.py'])