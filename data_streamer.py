from doctest import testfile
import mne
import mne_realtime
import pathlib
from pythonosc import udp_client
import numpy as np 
import threading
import multiprocessing
import copy
import time
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from scipy import signal
import copy

mne.set_log_level(verbose="ERROR")

class conf:
    scaler = 1e6
    ip = "127.0.0.1"
    port = 9090
    osc_address = "/"
    lsl_host = "127.0.0.1"
    pull_n_samples = 251
    spectral_analyis_window = 1000
    n_minutes_buffer = 0.5

    frequency_bands = {'alpha':[8,12],
                         'beta':[14,20],
                        'gamma':[21,33]}

    test_file_name = 'test_file.txt'

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

    def connect_to_stream(self, stream_type='fake', raw_start_time=0):
        if stream_type == 'fake': # fake
            self.stream = mne_realtime.MockLSLStream(raw=self.raw.crop(raw_start_time), host=conf.lsl_host, ch_type="eeg")
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
        self.global_nsamp = 0

        print(f"created ring buffer {self.ring_buffer.shape}")
    
    def add_to_buffer(self, data):
        self.nsamp += data.shape[-1]
        self.global_nsamp += data.shape[-1]
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
        print(f'connecting to OSC at {ip}:{port}')
        self.client = udp_client.SimpleUDPClient(ip, port)
        print(f'connected to OSC at {ip}:{port}')


    def stream_to_osc(self, processor):
        while not processor.payload_ready:
            pass
        while 1:
            # if 
            if processor.payload_ready:
                pl = copy.deepcopy(processor.payload)
                for key, value in pl.items():
                    self.client.send_message(conf.osc_address + key, value)
                processor.payload_ready=False
                time.sleep(0.5)

class Processor(): #  transformer mixin?
    def __init__(self, eeg:EEG) -> None:
        self.eeg = eeg
        self.nsamp = None
        self.payload_ready = False
    
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
                    self.realtime_welch(channel=0)
                    self.transform_eeg_to_osc_features()
                    # print(self.eeg.ring_buffer.shape, start_window, self.nsamp)

    def realtime_welch(self, channel=0):
        # self.spectrum = signal.welch(self.window[channel,:], fs=self.eeg.info['sfreq'], window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
        # print(self.spectrum)
        # print(self.window.shape)
        self.spectrum = mne.time_frequency.psd_array_welch(self.window, sfreq = self.eeg.info['sfreq'], fmin=1, fmax=35)


    def transform_eeg_to_osc_features(self):
        self.payload = {}

        for band in conf.frequency_bands:
            bandstart = conf.frequency_bands[band][0]
            bandend = conf.frequency_bands[band][1]

            msk = np.ma.masked_where(np.logical_and(self.spectrum[1]>=bandstart, self.spectrum[1]<bandend), self.spectrum[1])
            feature_value = np.average(self.spectrum[0][:,msk.mask], axis=1)
            for n_, channel in enumerate(feature_value):
                self.payload[f"{band}_{n_+1}"] = feature_value[n_]
        self.payload_ready = True
    def fit(self):
        pass
    
    def transform(self):
        pass

class Plotter_pyqtgraph():
    def __init__(self, eeg):
        while not hasattr(eeg, 'ring_buffer'):
            pass
        
        self.win = pg.GraphicsWindow(title="Signal") # creates a window
        self.eeg_plots = {}
        self.spectrum_plots = {}
        for channel in range(eeg.ring_buffer.shape[0]):

            p = self.win.addPlot(row=channel, col=0, title="Realtime plot")
            p2 = self.win.addPlot(row=channel, col=1, title="Spectum plot")  # creates empty space for the plot in the window
            self.eeg_plots[channel] = p.plot()                        # create an empty "plot" (a curve to plot)
            self.spectrum_plots[channel] = p2.plot()                        # create an empty "plot" (a curve to plot)

        QtGui.QApplication.processEvents()
        
    def update(self, data, data2, x):
        # print(data.shape, data2.shape)
        for channel in range(eeg.ring_buffer.shape[0]):
            # print(channel)
            self.eeg_plots[channel].setData(x, data[channel,:])                     # set the curve with this data
            # self.curve.setPos(self.ptr,0)                   # set x position in the graph to 0
            self.spectrum_plots[channel].setData(data2[1], data2[0][channel]) 
        time.sleep(0.1)
    
    def run(self, processor):
        print ('Plotter thread started')

        while not hasattr(processor, 'window') or not hasattr(processor, 'spectrum'):
            pass
        while 1:
            x = [(processor.eeg.global_nsamp+n)/processor.eeg.info['sfreq'] for n in range(len(processor.window[0,:]))]
            self.update(processor.window, processor.spectrum, x)
            

def do_eeg(eeg:EEG):
    print ('EEG thread started')
    with open(conf.test_file_name, 'r') as f:
        test_file_name = f.read()
    eeg.read_eeg_file(pathlib.Path(test_file_name))
    eeg.connect_to_stream(raw_start_time=60*5.5)
    eeg.create_client()
    eeg.get_eeg_data()

if __name__ == "__main__":
    app = QtGui.QApplication([])
    
    mgr = multiprocessing.Manager()
    namespace = mgr.Namespace()
    
    eeg = EEG()
    th = threading.Thread(target=do_eeg, args = (eeg,))
    th.start()
    
    processor = Processor(eeg)
    th2 = threading.Thread(target=processor.get_window_eeg)
    th2.start()
    
    plotter = Plotter_pyqtgraph(eeg)
    th3 = threading.Thread(target=plotter.run, args = (processor,))
    th3.start()


    # streamer = OSCStreamer()
    # streamer.create_osc_stream(ip=conf.ip, port=conf.port)
    # streamer.stream_to_osc(processor)


    pg.QtGui.QApplication.exec_() # you MUST put this at the end
    