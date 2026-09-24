"""Expose the real PicoRV32 over the existing PC demo protocol through XSDB."""
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

    def infer(self,payload,opcode=1):
        self.stream.write(f'{opcode} {payload.hex()}\n'.encode())
        while True:
            line=self.stream.readline().decode()
            if not line or line.startswith('ERROR'):
                raise RuntimeError('JTAG transport failed: '+line.strip())
            if line.startswith('RESULT '):
                values=tuple(map(int,line.split()[1:]))
                if len(values)!=6:raise RuntimeError('Malformed XSDB response')
                return values

    def close(self):
        self.stream.close();self.sock.close()
        # xsdb.bat spawns xsdb.exe; terminate() alone orphans it and leaves port 5557 held.
        subprocess.run(['taskkill','/F','/T','/PID',str(self.proc.pid)],capture_output=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xsdb',type=Path,default=Path('C:/AMDDesignTools/2025.2/Vivado/bin/xsdb.bat'))
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
