import asyncio
import pickle
import threading
from pythonosc import udp_client
import socket
import time
import threading


class conf:
    osc_ip_max = "127.0.0.1"
    osc_ip_td = "127.0.0.1"
    # host_integrator = "2.0.0.151"
    host_integrator = "2.0.0.151"

    
    port_max = 9090
    port_td = 9094
    port_integrator = 8084

    osc_address = "/"



class OSCServer():
    def __init__(self, ip_osc, port_osc) -> None:
        self.payload_buffer = []
        self.create_osc_stream(ip_osc, port_osc)
        self.sent_cc = 0

    async def handle_client(self, reader, writer):
        request = None
        while request != 'quit':
            request = (await reader.readuntil(separator=b'\n'*20))
            try:
                self.payload_buffer.append(pickle.loads(request))
                self.stream_to_osc()
            except Exception as e:
                print (f"\n\n\nException in OSC server: {e}\n\n\n")
            await writer.drain()
        writer.close()

    async def run_server(self):
        server = await asyncio.start_server(self.handle_client, conf.host_integrator, conf.port_integrator)
        async with server:
            await server.serve_forever()
    
    def create_osc_stream(self, ip:str="127.0.0.1", port:int=8090) ->None:
        print(f'connecting to OSC at {ip}:{port}')
        self.client = udp_client.SimpleUDPClient(ip, port)
        print(f'connected to OSC at {ip}:{port}')

    def stream_to_osc(self):
        for i in self.payload_buffer:
            pl = self.payload_buffer.pop()
            for key, value in pl.items():
                self.client.send_message(conf.osc_address + key, value)
            self.sent_cc += 1
            if self.sent_cc%10 ==0:
                print(f"{time.time()} sent {self.sent_cc*10} data packets, {len(self.payload_buffer)}")
            time.sleep(0.1)
    

srv = OSCServer(ip_osc=conf.osc_ip_max, port_osc=conf.port_td)

th1 = threading.Thread(target=asyncio.run(srv.run_server()))
th1.start()


