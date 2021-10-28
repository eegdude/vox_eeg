'''
IMPORTANT
patch lsl_client.py ._connect()
    ids.append(stream_info.type())

to make it work with NeoRec
'''
import asyncio
from email import message
import mne
import mne_realtime
import pathlib
import numpy as np 
import json
import pickle
import multiprocessing
import copy
import time
from scipy import signal
import copy

mne.set_log_level(verbose="ERROR")

class conf:
    scaler = 1e6
    osc_ip_max = "127.0.0.1"
    osc_ip_td = "127.0.0.1"
    host_integrator = "2.0.0.151"
    
    lsl_host = "NE-ENOBIO20 (00:07:80:64:EB:62)" # not host
    lsl_port = 16575
    lsl_fakehost = "EEG"

    port_max = 9090
    port_td = 9094
    port_integrator = 8084


    osc_address = "/"
    pull_n_samples = 251
    spectral_analyis_window = 1000
    downsample_div = 10
    n_minutes_buffer = 0.5
    
    info = mne.create_info(['Cz', 'C3', 'C4', 'P3', 'Pz', 'P4', 'PO3'], 1000, ch_types='eeg')

    frequency_bands = {'alpha':[8,12],
                         'beta':[14,20],
                        'gamma':[21,33]}

    test_file_name = 'test_file.json'

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
    def __init__(self, namespace, fake):
        self.ns = namespace
        self.raw = None
        self.fake = fake
        self.data_buffer = np.empty(0)
        self.ns.eeg_nsamp = 0
    
    def read_eeg_file(self, filename:pathlib.Path):
        print(filename, filename.suffix)
        if filename.suffix in ['.edf', '.bdf']:
            self.raw = self.read_xdf_file(filename=filename)
        elif filename.suffix == '.vhdr':
            self.raw = mne.io.read_raw_brainvision(vhdr_fname=filename)
        else:
            raise NotImplementedError

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
            pass
    
    def create_client(self):
        if self.fake:
            self.client = mne_realtime.LSLClient(host=conf.lsl_host, buffer_size = 100)
        else:
            self.client = mne_realtime.LSLClient(conf.info, host=conf.lsl_host, buffer_size = 100)
        self.client.start()

    def get_eeg_data(self):
        
        epoch = self.client.get_data_as_epoch(n_samples=conf.pull_n_samples) # get info # fix for different units

        self.ns.info = epoch.info
        self.create_ring_buffer()

        for a in self.client.iter_raw_buffers(): #get raw data
            # self.ns.t = a[1]
            # a = a[0]

            if a.size:
                self.add_to_buffer(a)

    def create_ring_buffer(self):
        self.ns.ring_buffer = np.zeros((self.ns.info['nchan'], int(self.ns.info['sfreq']*60*conf.n_minutes_buffer)))*np.nan
        self.ring_buffer = np.zeros((self.ns.info['nchan'], int(self.ns.info['sfreq']*60*conf.n_minutes_buffer)))*np.nan

        self.ns.eeg_nsamp = 0
        self.ns.global_nsamp = 0

        print(f"created ring buffer {self.ring_buffer.shape}")
    
    def add_to_buffer(self, data):
        data *= conf.scaler
        nsamp = self.ns.eeg_nsamp + data.shape[-1]
        if nsamp> self.ring_buffer.shape[-1]:
            transfer_data_to_new_buffer = copy.deepcopy(self.ring_buffer[:,-1*conf.spectral_analyis_window*2:])
            transfer_data_to_new_buffer = transfer_data_to_new_buffer[:,~np.isnan(transfer_data_to_new_buffer[0,:])]

            self.ring_buffer = self.ring_buffer*np.nan
            nsamp = 0
            self.add_to_buffer(transfer_data_to_new_buffer)
            print('updated buffer')
        else:
            self.ring_buffer[:,nsamp - data.shape[1]:nsamp] = data
            self.ns.ring_buffer = self.ring_buffer

        self.ns.eeg_nsamp = nsamp
        self.ns.global_nsamp += data.shape[-1]


class Processor(): #  transformer mixin?
    def __init__(self, namespace) -> None:
        self.ns = namespace
        self.nsamp = None
        self.ns.payloads = []
    
    def realtime_filter(self):
        pass
    
    def get_window_eeg(self):
        print ('Processing thread started')
        while not hasattr(self.ns, 'info'):
            pass
        flt = signal.butter(4, [1, 30], btype = 'bandpass',fs = self.ns.info['sfreq'])
        while 1:
            time.sleep(0.1)

            if self.nsamp != self.ns.eeg_nsamp: # new sample arrives
                self.nsamp = self.ns.eeg_nsamp
                start_window = self.nsamp - conf.spectral_analyis_window
                if start_window > 0: # if there is at least conf.spectral_analyis_window samples
                    window = self.ns.ring_buffer[:, start_window:self.nsamp]
                    self.ns.window = signal.filtfilt(*flt, window)
                    self.realtime_welch(channel=0)
                    self.transform_eeg_to_osc_features()
                    # print(self.eeg.ring_buffer.shape, start_window, self.nsamp)

    def realtime_welch(self, channel=0):
        # self.spectrum = signal.welch(self.window[channel,:], fs=self.eeg.info['sfreq'], window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
        # print(self.spectrum)
        # print(self.window.shape)
        # print(self.ns.window)
        self.ns.spectrum = mne.time_frequency.psd_array_welch(self.ns.window, sfreq = self.ns.info['sfreq'], fmin=1, fmax=35)

    def transform_eeg_to_osc_features(self):
        #unfuck the cycle
        payload = {}

        for band in conf.frequency_bands:
            bandstart = conf.frequency_bands[band][0]
            bandend = conf.frequency_bands[band][1]

            msk = np.ma.masked_where(np.logical_and(self.ns.spectrum[1]>=bandstart, self.ns.spectrum[1]<bandend), self.ns.spectrum[1])
            feature_value = np.average(self.ns.spectrum[0][:,msk.mask], axis=1)
            for n_, value in enumerate(feature_value):
                payload[f"{band}_{n_+1}"] = feature_value[n_]
        
                payload[f"eeg_{n_}"] = signal.decimate(self.ns.window[n_,:], conf.downsample_div).tolist()

        for n_, value in enumerate(feature_value):
        
            payload[f"eeg_{n_}"] = signal.decimate(self.ns.window[n_,:], conf.downsample_div).tolist()

        for band in conf.frequency_bands:
            payload[f"average_{band}"] = np.average([payload[a] for a in payload.keys() if band in a])
        self.ns.payloads = self.ns.payloads + [(payload)]
        # self.ns.payload_ready = {0:True, 1:True}



class AsyncClient():

    def __init__(self, namespace, host, port):
        self.host= host
        self.port = port
        self.ns = namespace
        
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(self.create())
        asyncio.run(self.run())

    async def create(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)


    def writable(self):
        if hasattr(self.ns, 'payloads'):
            if len(self.ns.payloads) > 0:
                return True
        return False
    
    async def run(self):
        while 1:
            if self.writable():
                print ("writeable")
                message = self.ns.payloads[0]
                self.ns.payloads = self.ns.payloads[1:]

                self.writer.write(pickle.dumps(message) + b"\n"*20)
                await self.writer.drain()

def do_eeg(namespace, fake = False):
    eeg = EEG(namespace, fake=fake)

    print ('EEG thread started')

    if fake:
        with open(conf.test_file_name, 'r') as f:
            test_file = json.loads(f.read())['test_file']
        eeg.read_eeg_file(pathlib.Path(test_file))
        eeg.connect_to_stream(raw_start_time=60*5.5)
    eeg.create_client()
    eeg.get_eeg_data()


def do_processing(namespace):
    processor = Processor(namespace)
    processor.get_window_eeg()

def push_data(namespace, host, port):
    pusher = AsyncClient(namespace, host, port)
    asyncio.run(pusher.run())
    
if __name__ == "__main__":

    
    mgr = multiprocessing.Manager()
    namespace = mgr.Namespace()

    th1 = multiprocessing.Process(target=do_eeg, args = (namespace, True))
    th1.start()
    
    th2 = multiprocessing.Process(target=do_processing, args = (namespace,))
    th2.start()
    
    th3 = multiprocessing.Process(target=push_data, args = (namespace, conf.host_integrator, conf.port_integrator))
    th3.start()
    
    th1.join()
    th2.join()
    th3.join()
