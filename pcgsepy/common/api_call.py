import json
import os
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pcgsepy.common.jsonrpc import TransportTcpIp
from pcgsepy.common.vecs import Vec
from pcgsepy.config import HOST, PORT


class GameMode(Enum):
    """Enum for the game mode."""
    PLACING = False
    EVALUATING = True


def generate_json(method: str,
                  params: Optional[List[Any]] = [],
                  request_id: int = random.getrandbits(32)) -> Dict[str, Any]:
    """Create the JSONRPC-compatible JSON data

    Args:
        method (str): The Space Engineer's API method name.
        params (Optional[List[Any]], optional): Additional method's parameters. Defaults to `[]`.
        request_id (int, optional): Unique ID of the request. Defaults to `random.getrandbits(32)`.

    Returns:
        Dict[str, Any]: The JSONRPC-compatible JSON data
    """
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": request_id
    }


def compactify_jsons(jsons: List[Dict[str, Any]]) -> str:
    """Compactify JSON data. Removes any space in the JSOn and uses no indentation.

    Args:
        jsons (List[Dict[str, Any]]): The JSON data to compactify.

    Returns:
        List[str]: The compacted JSON data.
    """
    return '\r\n'.join([f"{json.dumps(obj=j, separators=(',', ':'), indent=None)}" for j in jsons]) + '\r\n'


def call_api(host: str = HOST,
             port: int = PORT,
             jsons: List[Dict[str, Any]] = {}) -> List[Dict[str, Any]]:
    """Call the Space Engineer's API.

    Args:
        host (str, optional): The host address . Defaults to `HOST`.
        port (int, optional): The port number. Defaults to `PORT`.
        jsons (List[Dict[str, Any]], optional): The list of methods and parameters as a list of JSONs. Defaults to `{}`.

    Returns:
        List[Dict[str, Any]]: The data returned from the Space Engineer's API.
    """
    s = TransportTcpIp(addr=(host, port),
                       timeout=2,
                       limit=4096)
    return json.loads(s.sendrecv(string=compactify_jsons(jsons=jsons).encode('utf-8')).decode(encoding='utf-8'))


# block_definitions as a module-level variable
if not os.path.exists('./block_definitions.json'):
    # poll API for block definition ids
    jsons = [generate_json(method="Definitions.BlockDefinitions")]
    res = call_api(jsons=jsons)
    block_definitions = {}
    for v in res['result']:
        block_definitions['_'.join([v['DefinitionId']['Id'],
                                    v['DefinitionId']['Type']])] = {
            'cube_size': v['CubeSize'],
            'size': v['Size'],
            'mass': v['Mass'],
            'definition_id': {'Id': v['DefinitionId']['Id'],
                              'Type': v['DefinitionId']['Type']},
            'mountpoints': v['MountPoints']
        }
    with open('./block_definitions.json', 'w') as f:
        json.dump(block_definitions, f)
else:
    with open('./block_definitions.json', 'r') as f:
        block_definitions = json.load(f)


def get_base_values() -> Tuple[Vec, Vec, Vec]:
    """Get the position, orientation forward and orientation up of the player.

    Returns:
        Tuple[Vec, Vec, Vec]: The data as tuple of `Vec`s
    """
    obs = call_api(jsons=[generate_json(method="Observer.Observe")])
    return Vec.from_json(obs['result']['Position']), Vec.from_json(obs['result']['OrientationForward']), Vec.from_json(obs['result']['Camera']['OrientationUp'])


def toggle_gamemode(mode: GameMode) -> None:
    """Switch between `GameMode.PLACING` (fast) and `GameMode.EVALUATING` (slow). This could become deprecated in the future.

    Args:
        mode (GameMode): The game mode to toggle.
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
    def __offset(batch_size: int,
                 length: int,
                 batch_n: int) -> int:
        """Compute the batch offset.

        Args:
            batch_size (int): The size of the batch.
            length (int): The length of the iterable.
            batch_n (int): The current batch number.

        Returns:
            int: The offset.
        """
        return batch_size if (length > (batch_n + 1) * batch_size) else batch_size - ((batch_n + 1) * batch_size - length)
    return [(batch_n * batch_size, batch_n * batch_size + __offset(batch_size=batch_size,
                                                                   length=length,
                                                                   batch_n=batch_n)) for batch_n in range(0, (length // batch_size) + (0 if (length % batch_size == 0 or drop_last) else 1))]


def place_blocks(blocks: List[Any],
                 sequential: False) -> None:
    """Place the blocks in-game.

    Args:
        blocks (List[Block]): The list of blocks.
        sequential (bool): Flag to either make the `Admin.Blocks.PlaceAt` call for each block or for the entire list.
    """
    # prepare jsons
    jsons = [generate_json(
        method='Admin.Blocks.PlaceAt',
        params={
            "blockDefinitionId": block.definition_id,
            "position": block.position.as_dict(),
            "orientationForward": block.orientation_forward.as_dict(),
            "orientationUp": block.orientation_up.as_dict()
            }) for block in blocks]
    # place blocks
    if not sequential:
        call_api(jsons=jsons)
    else:
        for j in jsons:
            call_api(jsons=j)

# TODO: This code is currently unused

# def rotate_and_normalize_block(rotation_matrix: np.ndarray,
#                                normalizing_block: Block,
#                                block: Block,
#                                to_int: bool = True) -> Tuple[Vec, Vec, Vec]:
#     of = rotate(rotation_matrix=rotation_matrix,
#                 vector=block.orientation_forward)
#     ou = rotate(rotation_matrix=rotation_matrix,
#                 vector=block.orientation_up)

#     pos = block.position.sum(normalizing_block.position.scale(-1))
#     pos = rotate(rotation_matrix=rotation_matrix,
#                 vector=pos)

#     return (of.to_veci(), ou.to_veci(), pos.to_veci()) if to_int else (of, ou, pos)


# def try_place_block(block: Block,
#                     rotation_matrix: npt.NDArray,
#                     normalizing_block: Block,
#                     grid_id: str) -> bool:
#     of, ou, pos = rotate_and_normalize_block(rotation_matrix=rotation_matrix,
#                                             normalizing_block=normalizing_block,
#                                             block=block,
#                                             to_int=True)
#     res = call_api(jsons=[
#         generate_json(method='Admin.Blocks.PlaceInGrid',
#                     params={
#                         "blockDefinitionId": block_definitions[block.block_type]['definition_id'],
#                         "gridId": grid_id,
#                         "minPosition": pos.as_dict(),
#                         "orientationForward": of.as_dict(),
#                         "orientationUp": ou.as_dict()
#                     })
#     ])
#     return res[0].get('error', None) is None


# class PlacementException(Exception):
#     pass


# def place_structure(structure: Structure,
#                     position: Vec,
#                     orientation_forward: Vec = Orientation.FORWARD.value,
#                     orientation_up: Vec = Orientation.UP.value,
#                     batchify: bool = True) -> None:
#     """
#     Place the structure in-game.

#     Parameters
#     ----------
#     structure : Structure
#         The structure to place.
#     position : Vec
#         The minimum position of the structure to place at.
#     orientation_forward : Vec
#         The Forward orientation, as vector.
#     orientation_up : Vec
#         The Up orientation, as vector.
#     """
#     # ensure structure position and orientation
#     structure.update(
#         origin=Vec.v3f(0., 0., 0.),
#         orientation_forward=orientation_forward,
#         orientation_up=orientation_up,
#     )
#     structure.sanify()
#     to_place = structure.get_all_blocks(to_place=False)
#     # toggle gamemode to place faster
#     toggle_gamemode(GameMode.PLACING)
#     # get lowest-index block in structure
#     first_block = None
#     for block in to_place:
#         if first_block:
#             if block.position.x <= first_block.position.x and block.position.y <= first_block.position.y and block.position.z <= first_block.position.z:
#                 first_block = block
#         else:
#             first_block = block
#     # remove first block from list
#     to_place.remove(first_block)
#     # place first block
#     call_api(jsons=[
#         generate_json(
#             method='Admin.Blocks.PlaceAt',
#             params={
#                 "blockDefinitionId": block_definitions[first_block.block_type]['definition_id'],
#                 "position": position.as_dict(),
#                 "orientationForward": first_block.orientation_forward.as_dict(),
#                 "orientationUp": first_block.orientation_up.as_dict()
#             })
#     ])
#     # get placed block's grid
#     observation = call_api(jsons=[generate_json(method='Observer.ObserveBlocks', params={})])
#     grid_id = observation[0]["result"]["Grids"][0]["Id"]
#     grid_orientation_forward = Vec.from_json(observation[0]["result"]["Grids"][0]["OrientationForward"])
#     grid_orientation_up = Vec.from_json(observation[0]["result"]["Grids"][0]["OrientationUp"])
#     rotation_matrix = get_rotation_matrix(forward=grid_orientation_forward,
#                                         up=grid_orientation_up)

#     # TODO: Move character away so that the spaceship can be built entirely

#     # reorder blocks
#     occupied_space = [first_block.position]
#     ordered_blocks = []
#     while to_place:
#         to_rem = []
#         for block in to_place:
#             if (block.position.sum(Vec.v3i(-1, 0, 0)) in occupied_space or
#                 block.position.sum(Vec.v3i(0, -1, 0)) in occupied_space or
#                 block.position.sum(Vec.v3i(0, 0, -1)) in occupied_space or
#                 block.position.sum(Vec.v3i(1, 0, 0)) in occupied_space or
#                 block.position.sum(Vec.v3i(0, 1, 0)) in occupied_space or
#                 block.position.sum(Vec.v3i(0, 0, 1)) in occupied_space):
#                 to_rem.append(block)
#                 occupied_space.append(block.position)
#                 ordered_blocks.append(block)
#         for r in to_rem:
#             to_place.remove(r)
#     if batchify:
#         # attempt placement in batches
#         batch_size = 64
#         jsons = []
#         for n, block in enumerate(ordered_blocks):
#             of, ou, pos = rotate_and_normalize_block(rotation_matrix=rotation_matrix,
#                                                     normalizing_block=first_block,
#                                                     block=block,
#                                                     to_int=True)
#             jsons.append(generate_json(method='Admin.Blocks.PlaceInGrid',
#                                        params={
#                                            "blockDefinitionId": block_definitions[block.block_type]['definition_id'],
#                                            "gridId": grid_id,
#                                            "minPosition": pos.as_dict(),
#                                            "orientationForward": of.as_dict(),
#                                            "orientationUp": ou.as_dict()},
#                                        request_id=n))
#         n_requests = 0
#         while jsons:
#             to_rem = []
#             for (idx_from, idx_to) in get_batch_ranges(batch_size=batch_size,
#                                                        length=len(jsons)):
#                 res_list = call_api(jsons=jsons[idx_from:idx_to])
#                 for res in res_list:
#                     if res.get('error', None) is None:
#                         to_rem.append(res)
#             for res in to_rem:
#                 for i, req in enumerate(jsons):
#                     if req['id'] == res['id']:
#                         jsons.pop(i)
#                         break
#             if len(jsons) == n_requests:
#                 raise PlacementException(f'Error during spaceship placement: missing {len(jsons)} blocks to place.')
#             else:
#                 n_requests = len(jsons)
#     else:
#         # attempt placement sequentially
#         errored_out = []
#         for block in ordered_blocks:
#             # always try to place blocks that we failed to place previously
#             to_rem = []
#             for b in errored_out:
#                 res = try_place_block(block=b,
#                                     rotation_matrix=rotation_matrix,
#                                     normalizing_block=first_block,
#                                     grid_id=grid_id)
#                 if res:
#                     to_rem.append(b)
#             for b in to_rem:
#                 errored_out.remove(b)
#             # try and place current block
#             res = try_place_block(block=block,
#                                 rotation_matrix=rotation_matrix,
#                                 normalizing_block=first_block,
#                                 grid_id=grid_id)
#             if not res:
#                 errored_out.append(block)
#         if len(errored_out) != 0:
#             raise PlacementException(f'Error during spaceship placement: missing {len(errored_out)} blocks to place.')
#     # toggle back gamemode
#     toggle_gamemode(GameMode.EVALUATING)
