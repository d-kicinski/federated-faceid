import argparse
import glob
import os
import time

import pandas as pd
from tqdm import tqdm


def generate_csv_file(dataroot, csv_name="vggface2.csv"):
    """Generates a csv file containing the image paths of the VGGFace2 dataset for use in triplet selection in
    triplet loss training.

    Args:
        dataroot (str): absolute path to the training dataset.
        csv_name (str): name of the resulting csv file.
    """
    print("\nLoading image paths ...")
    files = glob.glob(dataroot + "/*/*")
    start_time = time.time()
    list_rows = []

    print("Number of files: {}".format(len(files)))
    print("\nGenerating csv file ...")

    progress_bar = enumerate(tqdm(files))

    for file_index, file in progress_bar:
        face_id = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))

        # Better alternative than dataframe.append()
        row = {'id': face_id, 'name': face_label}
        list_rows.append(row)

    df = pd.DataFrame(list_rows)
    df = df.sort_values(by=['name', 'id']).reset_index(drop=True)

    # Encode names as categorical classes
    df['class'] = pd.factorize(df['name'])[0]
    df.to_csv(path_or_buf=csv_name, index=False)

    elapsed_time = time.time() - start_time
    print("\nDone! Elapsed time: {:.2f} minutes.".format(elapsed_time / 60))


def parse_args():
    parser = argparse.ArgumentParser(description="Generating csv file for triplet loss!")
    parser.add_argument('--dataroot', '-d', type=str, required=True,
                        help="(REQUIRED) Absolute path to the dataset folder to generate a csv "
                             "file containing the paths of the images for triplet loss. ")

    parser.add_argument('--csv_name', type=str,
                        help="Required name of the csv file to be generated. (default: "
                             "'vggface2.csv') ")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    generate_csv_file(dataroot=args.dataroot, csv_name=args.csv_name)


if __name__ == '__main__':
    main()
