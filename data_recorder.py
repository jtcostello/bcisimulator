import pickle
import datetime
import os
import pandas as pd


class DataRecorder:
    """Records movement data and saves to file"""
    def __init__(self):
        self.data = []

    def reset(self):
        self.data = []

    def record(self, timestep, trial_number, current_position, target_position, online, decodername=None):
        entry = {
            "timestep": timestep,
            "trial_number": trial_number,
            "current_position": current_position,
            "target_position": target_position,
            "online": online,
            "decodername": decodername,
        }
        self.data.append(entry)

    def save_to_file(self):
        if len(self.data) < 1:
            print("no data - failed to save")
            return

        # convert to dataframe
        df = pd.DataFrame(self.data)

        # save to the data folder
        datestr = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"dataset_{datestr}.pkl"
        fpath = os.path.join("data", "movedata", fname)
        with open(fpath, "wb") as f:
            pickle.dump(df, f)
        print(f"Saved data to {fpath}")

        self.reset()


