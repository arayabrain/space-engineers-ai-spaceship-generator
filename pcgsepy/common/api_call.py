from enum import Enum
import json
import random
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

from .vecs import Vec
from ..config import HOST, PORT


class GameMode(Enum):
    PLACING = False
    EVALUATING = True


def generate_json(method: str,
                  params: Optional[List[Any]] = None) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params if params else [],
        "id": random.getrandbits(16)
    }

def compactify_jsons(jsons: List[Dict[str, Any]]) -> List[str]:
    """ JSONs should be single line & compact formatted before calling the API. (@Karel Hovorka)"""
    compacted_jsons = ''
    for j in jsons:
        compacted_jsons += json.dumps(obj=j,
                               separators=(',', ':'),
                               indent=None)
        compacted_jsons += '\r\n'

    return compacted_jsons

# Modified from https://www.binarytides.com/receive-full-data-with-the-recv-socket-function-in-python/
def recv_with_timeout(s: socket.socket,
                      timeout: int = 2):
    #make socket non blocking
    s.setblocking(0)
    #total data partwise in an array
    total_data = [];
    #beginning time
    begin = time.time()
    while True:
        # timeout termination
        if total_data and time.time() - begin > timeout:
            break
        # wait to get data
        elif time.time() - begin > timeout * 2:
            break
        # recv
        try:
            data = s.recv(8192).decode("utf-8")
            if data:
                # early break if socket returns the same data
                if len(total_data) > 1 and data == total_data[-1]:
                    break
                # append new data
                total_data.append(data)
                # change the beginning time for measurement
                begin = time.time()
            else:
                #sleep to indicate a gap
                time.sleep(0.1)
        except:
            # note: Exceptions are thrown due to socket not reading anything before and after data is passed
            pass
    # join all parts to make final string
    return ''.join(total_data)

def call_api(host: str = HOST,
             port: int = PORT,
             jsons: List[Dict[str, Any]] = {}) -> List[Dict[str, Any]]:
    # open socket for communication
    s = socket.socket(family=socket.AF_INET,
                      type=socket.SOCK_STREAM)
    s.connect((host, port))
    # send methods as compacted JSON bytearray
    s.sendall(compactify_jsons(jsons=jsons).encode('utf-8'))
    # get response
    res = recv_with_timeout(s)
    # close socket
    s.close()
    # due to TCP streming packets, it's possible some JSON-RPC responses are the same;
    # workaround: identify unique JSON-RPC responses by unique id
    return [json.loads(x) for x in list(set(res.strip().split('\r\n')))]

def get_base_values() -> Tuple[Vec, Vec, Vec]:
    obs = call_api(jsons=[generate_json(method="Observer.Observe")])[0]
    base_position = Vec.from_json(obs['result']['Position'])
    orientation_forward = Vec.from_json(obs['result']['OrientationForward'])
    orientation_up = Vec.from_json(obs['result']['Camera']['OrientationUp'])
    return base_position, orientation_forward, orientation_up

def toggle_gamemode(mode: GameMode) -> None:
    # "we plan to change the API and rename this function in near future" (@Karel Hovorka)
    # Switch between `GameMode.PLACING` (fast) and `GameMode.EVALUATING` (slow).
    call_api(jsons=[generate_json(method="Admin.SetFrameLimitEnabled",
                                  params=[mode.value])])
