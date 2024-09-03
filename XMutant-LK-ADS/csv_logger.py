import csv
import os


def episode_logger(filepath: str, log_info: dict):
    if not filepath.endswith(".csv"):
        filepath += ".csv"

    # write CSV to file
    if not os.path.exists(filepath):
        # create csv file header
        with open(filepath, 'w', encoding='UTF8') as f:
            writer = csv.writer(f,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            # write the header
            writer.writerow(list(log_info.keys()))

    with open(filepath, 'a', encoding='UTF8') as f:
        writer = csv.writer(f,
                            delimiter=',',
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL,
                            lineterminator='\n')
        writer.writerow(list(log_info.values()))

