"""Expose the real PicoRV32 over the existing PC demo protocol through XSDB.

The relay detects the loaded firmware: ABI v2 (window model) or v3 (streaming
model); board_server.handle() serves whichever clients match (protocol.py).
"""
import argparse
from pathlib import Path
import socket
import subprocess
from board_server import handle

ROOT=Path(__file__).resolve().parent


class JtagBoard:
    def __init__(self,xsdb):
        self.proc=subprocess.Popen([str(xsdb),str(ROOT/'jtag_server.tcl')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        while True:
            line=self.proc.stdout.readline()
            if not line: raise RuntimeError('XSDB exited before initializing')
            if line.strip()=='READY':break
            print(line.rstrip(),flush=True)
        self.sock=socket.create_connection(('127.0.0.1',5557),timeout=5)
        self.stream=self.sock.makefile('rwb',buffering=0)

    def _ask(self,line,prefix,n):
        self.stream.write((line+'\n').encode())
        while True:
            reply=self.stream.readline().decode()
            if not reply or reply.startswith('ERROR'):
                raise RuntimeError('JTAG transport failed: '+reply.strip())
            if reply.startswith(prefix+' '):
                values=tuple(map(int,reply.split()[1:]))
                if len(values)!=n:raise RuntimeError('Malformed XSDB response')
                return values

    def info(self):
        if not hasattr(self,'_info'):self._info=self._ask('I','INFO',4)
        return self._info

    def infer(self,payload,opcode=1):
        return self._ask(f'{opcode} {payload.hex()}','RESULT',6)

    def frames(self,payload):
        return self._ask(f'4 {payload.hex()}','RESULT',8)

    def reset(self):
        return self._ask('5','RESULT',8)

    def close(self):
        self.stream.close();self.sock.close()
        # xsdb.bat spawns xsdb.exe; terminate() alone orphans it and leaves port 5557 held.
        subprocess.run(['taskkill','/F','/T','/PID',str(self.proc.pid)],capture_output=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    # Newest installed Vivado (2025.2 and 2026.1 both work).
    installed=sorted(Path('C:/AMDDesignTools').glob('*/Vivado/bin/xsdb.bat'))
    p.add_argument('--xsdb',type=Path,default=installed[-1] if installed else Path('C:/AMDDesignTools/2025.2/Vivado/bin/xsdb.bat'))
    p.add_argument('--port',type=int,default=5556)
    a=p.parse_args()
    backend=JtagBoard(a.xsdb)
    try:
        with socket.create_server(('127.0.0.1',a.port)) as server:
            print(f'JTAG board server ready on 127.0.0.1:{a.port}',flush=True)
            while True:
                client,_=server.accept()
                with client:
                    try:handle(client,backend)
                    except (ConnectionError,EOFError,TimeoutError):pass
    finally:backend.close()


if __name__=='__main__':main()
