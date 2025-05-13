import os
from datasets import load_dataset

def custom_load_dataset(dataset_path, streaming=True):
    def detect_format(path):
        for ext in ['.parquet', '.csv', '.json', '.txt']:
            if os.path.exists(path + ext):
                return ext[1:], path + ext
        return None, None

    train_format, train_file = detect_format(os.path.join(dataset_path, "train"))
    val_format, val_file = detect_format(os.path.join(dataset_path, "val"))

    if not train_file:
        raise ValueError("Training file not found.")

    if val_file and train_format != val_format:
        raise ValueError("Train and validation files must have the same format")

    data_files = {"train": train_file}
    if val_file:
        data_files["val"] = val_file

    dataset = load_dataset(train_format, data_files=data_files, streaming=streaming)
    return dataset

