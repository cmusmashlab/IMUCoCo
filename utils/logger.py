import json
import os
from turtle import pd
import datetime


class ExpLogger:
    def __init__(self, exp_type, exp_name):
        self.exp_name = exp_name
        self.log_message_file = "./exp_out/" + exp_type + '/' + exp_name + "/messages.txt"
        self.log_structural_file = "./exp_out/" + exp_type + '/' + exp_name + "/exp_data.csv"
        self.log_meta_file = "./exp_out/" + exp_type + '/' + exp_name + "/meta_data.json"

        if not os.path.exists(os.path.dirname(self.log_message_file)):
            os.makedirs(os.path.dirname(self.log_message_file))
        if not os.path.exists(os.path.dirname(self.log_structural_file)):
            os.makedirs(os.path.dirname(self.log_structural_file))
        self.structural_msgs = []
        self.meta_msgs = {}

    def log_msg(self, msg, verbose=True):
        # prepend message with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{timestamp}: {msg}"
        with open(self.log_message_file, 'a') as f:
            if verbose:
                print(msg)
            f.write(msg + '\n')

    def log_structural_msg(self, msg, verbose=True):
        # add timestamp to message
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg['timestamp'] = timestamp
        self.structural_msgs.append(msg)
        # log dictionary as pandas dataframe
        if verbose:
            print(msg)
        df = pd.DataFrame(self.structural_msgs)
        df.to_csv(self.log_structural_file, index=False)

    def log_meta_msg(self, msg, verbose=True):
        # add timestamp to message
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg['timestamp'] = timestamp
        self.meta_msgs.update(msg)
        # log dictionary as pandas dataframe
        if verbose:
            print(msg)
        # log as json into meta
        with open(self.log_meta_file, 'w') as f:
            f.write(json.dumps(self.meta_msgs, indent=4))



