from ast import Pass
from importlib.metadata import entry_points
import mne
import mne_realtime
import pathlib
from tqdm import tqdm
from pythonosc import udp_client
from matplotlib import pyplot as plt
import numpy as np 
import threading
import copy
import time
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from scipy import signal

mne.set_log_level(verbose="ERROR")

class conf:
    scaler = 1e6
    ip = "127.0.0.1"
    port = 8090
    osc_address = "/filter"
    lsl_host = "127.0.0.1"
    pull_n_samples = 251
    spectral_analyis_window = 1000
    n_minutes_buffer = 1

def check_data_loaded(func):
    # raise something besides StopIteration IF LSL STREAM STOPPED
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
        self.data_buffer = np.empty(0)
        self.nsamp = 0
    
    def read_eeg_file(self, filename:pathlib.Path):
        self.raw = self.read_xdf_file(filename=filename)
        self.raw._data *= conf.scaler # fix for different units
    
    def read_xdf_file(self, filename:pathlib.Path=None):
        print(filename.suffix)
        if filename.suffix == '.bdf':
            return mne.io.read_raw_bdf(filename, preload=True)
        elif filename.suffix == '.edf':
            return mne.io.read_raw_edf(filename, preload=True)

    def connect_to_stream(self):
        if 1: # fake
            self.stream = mne_realtime.MockLSLStream(raw=self.raw, host=conf.lsl_host, ch_type="eeg")
            self.stream.start()
        else:
            raise NotImplementedError
    
    def create_client(self):
        self.client = mne_realtime.LSLClient(host=conf.lsl_host, buffer_size = 100)
        self.client.start()

    def get_eeg_data(self):
        
        epoch = self.client.get_data_as_epoch(n_samples=conf.pull_n_samples) # get info
        self.info = epoch.info
        self.create_ring_buffer()

        for a in self.client.iter_raw_buffers(): #get raw data
            if a.size:
                self.add_to_buffer(a)

    def create_ring_buffer(self):
        self.ring_buffer = np.zeros((self.info['nchan'], int(self.info['sfreq']*60*conf.n_minutes_buffer)))*np.nan
        self.nsamp = 0
        print(f"created ring buffer {self.ring_buffer.shape}")
    
    def add_to_buffer(self, data):
        self.nsamp += data.shape[-1]
        if self.nsamp > self.ring_buffer.shape[-1]:
            transfer_data_to_new_buffer = copy.deepcopy(self.ring_buffer[:,-1*conf.spectral_analyis_window*2:])
            transfer_data_to_new_buffer = transfer_data_to_new_buffer[:,~np.isnan(transfer_data_to_new_buffer[0,:])]

            self.ring_buffer*=np.nan
            self.nsamp = 0
            self.add_to_buffer(transfer_data_to_new_buffer)
            print('updated buffer')
        else:
            self.ring_buffer[:,self.nsamp - data.shape[1]:self.nsamp] = data

class OSCStreamer():
    def __init__(self) -> None:
        pass
    
    def create_osc_stream(self, ip:str="127.0.0.1", port:int=8090) ->None:
        self.client = udp_client.SimpleUDPClient(ip, port)

    def stream_to_osc(self, data):
        for n_, sample in enumerate(tqdm(data)):
            self.client.send_message(conf.osc_address, sample)

class Processor(): #  transformer mixin?
    def __init__(self, eeg:EEG) -> None:
        self.eeg = eeg
        self.nsamp = None
    
    def realtime_filter(self):
        pass
    
    def get_window_eeg(self):
        print ('Processing thread started')
        while not hasattr(self.eeg, 'info'):
            pass
        flt = signal.butter(4, [1, 30], btype = 'bandpass',fs = self.eeg.info['sfreq'])
        while 1:
            time.sleep(0.1)
            if self.nsamp != self.eeg.nsamp:
                self.nsamp = self.eeg.nsamp
                start_window = self.nsamp - conf.spectral_analyis_window

                if start_window > 0:
                    window = self.eeg.ring_buffer[:, start_window:self.nsamp]
                    self.window = signal.filtfilt(*flt, window)
                    self.realtime_welch(self.window)
                    # print(self.eeg.ring_buffer.shape, start_window, self.nsamp)

    def realtime_welch(self, window):
        self.spectrum = signal.welch(self.window[0,:], fs=self.eeg.info['sfreq'], window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')

    def transform_eeg_to_osc_features(self):
        pass

    def fit(self):
        pass
    
    def transform(self):
        pass

class Plotter_pyqtgraph():
    def __init__(self):
        self.win = pg.GraphicsWindow(title="Signal") # creates a window
        p = self.win.addPlot(title="Realtime plot")
        p2 = self.win.addPlot(title="Spectum plot")  # creates empty space for the plot in the window
          # creates empty space for the plot in the window
        self.curve = p.plot()                        # create an empty "plot" (a curve to plot)
        self.spectrum = p2.plot()                        # create an empty "plot" (a curve to plot)
        self.ptr = -conf.spectral_analyis_window                      # set first x position

    # Realtime data plot. Each time this function is called, the data display is updated
    def update(self, data, data2):
        # print(data.shape, data2.shape)
        self.ptr += 1                              # update x position for displaying the curve
        self.curve.setData(data)                     # set the curve with this data
        self.curve.setPos(self.ptr,0)                   # set x position in the graph to 0
        self.spectrum.setData(data2[1]) 
        time.sleep(0.1)
    
    def run(self, processor):
        print ('Plotter thread started')

        while not hasattr(processor, 'window') or not hasattr(processor, 'spectrum'):
            pass
        while 1:
            self.update(processor.window[0,:], processor.spectrum)
            

def do_eeg(eeg:EEG):
    print ('EEG thread started')
    eeg.read_eeg_file(pathlib.Path(r"C:\Data\kenul\raw\sdrnk\Эксперимент\WG_14_male_WG_15_male_17-04-2020_12-14.bdf"))
    eeg.connect_to_stream()
    eeg.create_client()
    eeg.get_eeg_data()

if __name__ == "__main__":
    app = QtGui.QApplication([])

    eeg = EEG()
    th = threading.Thread(target=do_eeg, args = (eeg,))
    th.start()
    
    processor = Processor(eeg)
    th2 = threading.Thread(target=processor.get_window_eeg)
    th2.start()
    
    plotter = Plotter_pyqtgraph()
    th3 = threading.Thread(target=plotter.run, args = (processor,))
    th3.start()

    pg.QtGui.QApplication.exec_() # you MUST put this at the end

    # plotter = Plotter(processor)
    # plotter.start()
    # th3 = threading.Thread(target=plotter.start)
    # th3.start()
    # streamer = OSCStreamer()
    # streamer.create_osc_stream(ip=conf.ip, port=conf.port)
    # streamer.stream_to_osc(eeg)


    