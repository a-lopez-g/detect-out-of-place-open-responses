import argparse
import pandas as pd
import numpy as np
import os
clear = lambda: os.system('clear')

test_texts = ["foo", "bar", "fum"]
label_map = {
    "1": "open", 
    "2": "no open", 
    "3": "ignore"
}

def main(test_mode=False):
    df = pd.read_csv("label.csv")
    try:
        print("Loading checkpoint...")
        checkpoint_df = pd.read_csv(f"checkpoint_labelled_asr.csv")
        labelled_indexes = checkpoint_df["index"].to_list()
        df = df[~df.index.isin(labelled_indexes)]
        print(f"Checkpoint loaded: {len(labelled_indexes)} labeled iterations found.")
    except FileNotFoundError:
        print("Checkpoint not found")
        checkpoint_df = pd.DataFrame()

    labelled_list = []
    counters = np.zeros((4,))
    #last_labelled_index = max(labelled_indexes)
    last_labelled_index = -1
    for i in range(len(df)):
        text = df.iloc[i, 0]
        
        data_dict = {}
        data_dict["index"] = last_labelled_index + i
        data_dict["asr"] = text
        
        print(f"\nITERATION {i} of {len(df)}:\n{text}\n")
        print(f"\nLabel counts:\n  - Open: {counters[0]}\n  - No open: {counters[1]}\n  - Ignore: {counters[2]}")
        while True:
            try:
                usr_input = input("\nEnter label for asr (1: open, 2: no open, 3: ignore):\n")
                data_dict["label"] = label_map[usr_input]
                counters[int(usr_input) - 1] += 1
                break
            except KeyError:
                print("\nWRONG INPUT!! must be one of the following: 1: open, 2: no open, 3: ignore. Try again")
        labelled_list.append(data_dict)
        clear()
        # Save checkpoint every 5 iterations
        if (i+1)%5 == 0:
            print("\nSAVING CHECKPOINT")
            checkpoint_df = pd.concat([checkpoint_df, pd.DataFrame(labelled_list)])
            checkpoint_df.to_csv(f"checkpoint_labelled_asr.csv")
            labelled_list = []
    df_labelled = pd.concat([checkpoint_df, pd.DataFrame(labelled_list)])
    if not test_mode:
        df_labelled.to_csv("dataset_out_of_place_open_responses.csv")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="Run in test mode.", action="store_true")
    args = parser.parse_args()
    main(args.test)