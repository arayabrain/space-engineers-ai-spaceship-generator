import json
import random
import socket
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from zmq import Socket
from .jsonrpc import TransportTcpIp

from ..config import HOST, PORT
from .vecs import Vec


class GameMode(Enum):
    """
    Enum for the game mode.
    """
    PLACING = False
    EVALUATING = True


def generate_json(method: str,
                  params: Optional[List[Any]] = None,
                  request_id: int = random.getrandbits(32)) -> Dict[str, Any]:
    """
    Create the JSONRPC-compatible JSON data.

    Parameters
    ----------
    method : str
        The Space Engineer's API method name.
    params : Optional[List[Any]]
        Additional method's parameters (default: None).

    Returns
    -------
    Dict[str, Any]
        The JSONRPC-compatible JSON data.
    """
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params if params else [],
        "id": request_id
    }


def compactify_jsons(jsons: List[Dict[str, Any]]) -> List[str]:
    """
    Compactify JSON data.
    Removes any space in the JSOn and uses no indentation.
    *"JSONs should be single line & compact formatted before calling the API." (@Karel Hovorka)*

    Parameters
    ----------
    jsons : List[Dict[str, Any]]
        The JSON data to compactify.

    Returns
    -------
    List[str]
        The compacted JSON data.
    """
    """ """
    compacted_jsons = ''
    for j in jsons:
        compacted_jsons += json.dumps(obj=j,
                                      separators=(',', ':'),
                                      indent=None)
        compacted_jsons += '\r\n'

    return compacted_jsons


def recv_with_timeout(s: socket.socket,
                      timeout: int = 2) -> str:
    """
    Receive data from a socket using a timeout.

    Parameters
    ----------
    s : socket.socket
        The socket to read data from.
    timeout : int
        The timeout in seconds (default: 2).

    Returns
    -------
    str
        The data received.
    """
    # make socket non blocking
    s.setblocking(0)
    # total data partwise in an array
    total_data = []
    # beginning time
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
                # sleep to indicate a gap
                time.sleep(0.1)
        except Exception:
            # note: Exceptions are thrown due to socket not reading anything
            # before and after data is passed
            pass
    # join all parts to make final string
    return ''.join(total_data)


def call_api(host: str = HOST,
             port: int = PORT,
             jsons: List[Dict[str, Any]] = {}) -> List[Dict[str, Any]]:
    """
    Call the Space Engineer's API.

    Parameters
    ----------
    host : str
        The host address (default: config-defined HOST).
    port : int
        The port number (default: config-defined PORT).
    jsons : List[Dict[str, Any]]
        The list of methods and parameters as a list of JSONs.

    Returns
    -------
    List[Dict[str, Any]]
        The data returned from the Space Engineer's API.
    """
    # # open socket for communication
    # s = socket.socket(family=socket.AF_INET,
    #                   type=socket.SOCK_STREAM)
    # s.connect((host, port))
    # # send methods as compacted JSON bytearray
    # s.sendall(compactify_jsons(jsons=jsons).encode('utf-8'))
    # # get response
    # res = recv_with_timeout(s)
    # # close socket
    # s.close()
    # # due to TCP streming packets, it's possible some JSON-RPC responses are
    # # the same; workaround: identify unique JSON-RPC responses by unique id
    # print(res)
    # return [json.loads(x) for x in list(set(res.strip().split('\r\n')))]
    s = TransportTcpIp(addr=(host, port), timeout=2)
    res = s.sendrecv(string=compactify_jsons(jsons=jsons).encode('utf-8'))
    res = res.decode(encoding='utf-8').strip().split('\r\n')
    valid = []
    for r in res:
        r = validate_json_str(s=r)
        if r != '':
            valid.append(r)
    valid = list(set(valid))
    return [json.loads(x) for x in valid if x.startswith('{"jsonrpc"')]


def validate_json_str(s: str) -> str:
    n = 0
    for i, c in enumerate(s):
        if c == '{':
            n += 1
        elif c == '}':
            n -= 1
        if n == 0:
            return s[:i+1]
    return ''


def get_base_values() -> Tuple[Vec, Vec, Vec]:
    """
    Get the position, orientation forward and orientation up of the player.

    Returns
    -------
    Tuple[Vec, Vec, Vec]
        The data as tuple of `Vec`s.
    """
    obs = call_api(jsons=[generate_json(method="Observer.Observe")])[0]
    base_position = Vec.from_json(obs['result']['Position'])
    orientation_forward = Vec.from_json(obs['result']['OrientationForward'])
    orientation_up = Vec.from_json(obs['result']['Camera']['OrientationUp'])
    return base_position, orientation_forward, orientation_up


def toggle_gamemode(mode: GameMode) -> None:
    """
    Switch between `GameMode.PLACING` (fast) and `GameMode.EVALUATING` (slow).
    *"we plan to change the API and rename this function in near future" (@Karel Hovorka)*

    Parameters
    ----------
    mode : GameMode
        The game mode to toggle.
    """
    call_api(jsons=[generate_json(method="Admin.SetFrameLimitEnabled",
                                  params=[mode.value])])


def get_batch_ranges(batch_size: int,
                     length: int,
                     drop_last: bool = False) -> List[Tuple[int, int]]:
    """Get the index ranges for batch iterations of an iterable object.

    Args:
        batch_size (int): The size of the batch.
        length (int): The length of the iterable.
        drop_last (bool, optional): Whether to drop the last batch if `< batch_size`. Defaults to False.

    Returns:
        List[Tuple[int, int]]: The index ranges for batch iterations.
    """
    ranges = []
    for batch_n in range(0, (length // batch_size) + (0 if (length % batch_size == 0 or drop_last) else 1)):
        offset = batch_size if (length > (batch_n + 1) * batch_size) else batch_size - ((batch_n + 1) * batch_size - length)
        ranges.append((batch_n * batch_size, batch_n * batch_size + offset))
    return ranges