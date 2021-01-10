import os, csv, argparse
import pandas as pd

def process_file(file):
    df = pd.read_csv(file, names=["image", "label", "probability", "x_max", "x_min", "y_max", "y_min"])
    print("total number of row: %s" %len(df))
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", "-dir", help="directory of your detected files")
    parser.add_argument("--outputfile", "-outfile", help="name of the output file")

    # read arguments from the command line
    args = parser.parse_args()
    # default values
    outfile='salicon_detected_objects.csv'
    dir_path="data/salicon/detected"

    if args.outputfile:
        outfile = args.outputfile
    if args.dir:
        dir_path = args.dir
    
    # read csv files into dataframs and concat them
    dfs = [ process_file(os.path.join(root, file)) for root, dirs, files in os.walk(dir_path) for file in files]
    df_new = pd.concat(dfs, ignore_index=True).groupby(['image'])

    total_rows = []

    for group_name, df_group in df_new:
        new_row = [[row.values[1], row.values[2], row.values[3], row.values[4], row.values[5], row.values[6]] for _, row in df_group.iterrows()]
        total_rows.append([group_name, new_row])

    print("total number of row: %s" %len(total_rows))
    csv_file = os.path.join(dir_path, outfile)

    with open(csv_file, mode='w', newline='', encoding='utf-8') as afile: 
        file_writer = csv.writer(afile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerows(total_rows)