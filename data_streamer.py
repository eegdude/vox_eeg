from ast import Pass
import mne
# import mne_realtime
import pathlib
from tqdm import tqdm
from pythonosc import udp_client
import argparse
import time
import random

class conf:
    scaler = 1e6
    ip = "127.0.0.1"
    port = 8090
    osc_address = "/filter"

def check_data_loaded(func):
    def wrapper(*args):
        if not args[0].raw:
            print("data not loaded!")
            raise StopIteration
        else:
            return func(*args)
    return wrapper

class EEG:
    def __init__(self):
        self.raw = None
    
    @check_data_loaded
    def __getitem__(self, sample:int=None):
        return self.raw._data[:, sample]
    
    @check_data_loaded
    def __len__(self):
        return self.raw._data.shape[1]

    @check_data_loaded
    def __iter__(self):
        self.sample_n = 0
        self.size = self.raw._data.shape[1]
        return self
    
    # @check_iteration
    def __next__(self):
        if self.sample_n < self.size:
            sample = self.raw._data[:,self.sample_n]
            self.sample_n += 1
            return sample
        else:
            raise StopIteration
    
    def read_eeg_file(self, filename:pathlib.Path):
        self.read_xdf_file(filename=filename)
        self.raw._data *= conf.scaler # fix for different units
    
    def read_xdf_file(self, filename:pathlib.Path=None):
        print(filename.suffix)
        if filename.suffix == '.bdf':
            self.raw = mne.io.read_raw_bdf(filename, preload=True)
        elif filename.suffix == '.edf':
            self.raw = mne.io.read_raw_edf(filename, preload=True)

    def get_eeg_from_stream(self):
        pass

    def generate_eeg(self, fs, frequency_repsonce, ):
        yield

class Streamer():
    def __init__(self) -> None:
        pass
    
    def create_osc_stream(self, ip:str="127.0.0.1", port:int=8090) ->None:
        self.client = udp_client.SimpleUDPClient(ip, port)

    def stream_to_osc(self, data):
        for n_, sample in enumerate(tqdm(data)):
            self.client.send_message(conf.osc_address, sample)

class Processor():
    def __init__(self) -> None:
        pass

    def transform_eeg_to_osc_features(self):
        pass

if __name__ == "__main__":
    eeg = EEG()
    eeg.read_eeg_file(pathlib.Path(r"C:\Data\kenul\raw\sdrnk\Эксперимент\WG_14_male_WG_15_male_17-04-2020_12-14.bdf"))

    streamer = Streamer()
    streamer.create_osc_stream(ip=conf.ip, port=conf.port)
    streamer.stream_to_osc(eeg)