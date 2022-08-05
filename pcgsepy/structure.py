import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from mpl_toolkits import mplot3d

from .common.api_call import (GameMode, call_api, generate_json,
                              get_base_values, get_batch_ranges, toggle_gamemode)
from .common.vecs import Orientation, Vec, get_rotation_matrix, rotate

# block_definitions as a module-level variable
if not os.path.exists('./block_definitions.json'):
    # poll API for block definition ids
    jsons = [generate_json(method="Definitions.BlockDefinitions")]
    res = call_api(jsons=jsons)[0]
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

# Sizes of blocks in grid spaces
_blocks_sizes = {'Small': 1, 'Normal': 2, 'Large': 5}

grid_to_coords = 0.5


class MountPoint:
    def __init__(self,
                 face: Dict[str, int],
                 start: Dict[str, float],
                 end: Dict[str, float],
                 exclusion_mask: int,
                 properties_mask: int,
                 block_size: Vec) -> None:
        # https://forum.keenswh.com/threads/mount-points-help.7320047/
        # https://steamcommunity.com/sharedfiles/filedetails/?id=581270158
        self.face = Vec.from_json(face)
        self.start = Vec.from_json(start).dot(block_size)#.to_veci()
        self.start = Vec(np.rint(self.start.x), np.rint(self.start.y), np.rint(self.start.z))
        self.end = Vec.from_json(end).dot(block_size)#.to_veci()
        self.end = Vec(np.rint(self.end.x), np.rint(self.end.y), np.rint(self.end.z))
        self.exclusion_mask = exclusion_mask
        self.properties_mask = properties_mask
        
        # check implementation from https://github.com/KeenSoftwareHouse/SpaceEngineers/blob/master/Sources/Sandbox.Game/Definitions/MyCubeBlockDefinition.cs#L236
        # self.computed_mps: Dict[str, List[Vec]] = {}
        
    # def _compute_valid_mountpoints(self) -> None:
    #     # mountpoints define areas of valid placement, so we can precompute them all for the block
    #     # when checking, remember to pick the correct face (orientation)!
    #     for mp in self.mountpoints:
    #         valid_mps = []
    #         for x in range(mp.start.x, mp.end.x + 1, 1 if mp.end.x >= mp.start.x else -1):
    #             for y in range(mp.start.y, mp.end.y + 1, 1 if mp.end.y >= mp.start.y else -1):
    #                 for z in range(mp.start.z, mp.end.z + 1, 1 if mp.end.z >= mp.start.z else -1):
    #                     valid_mps.append(mp.start.sum(Vec(x, y, z)))
    #         self.computed_mps[mp.face.as_tuple()] = valid_mps
    
    def __repr__(self) -> str:
        return f'Normal: {self.face}\tStart: {self.start}\tEnd: {self.end}'


class Block:
    """
    Space Engineer's Block class.
    """

    def __init__(self,
                 block_type: str,
                 orientation_forward: Orientation = Orientation.FORWARD,
                 orientation_up: Orientation = Orientation.UP):
        """
        Create a Block object.

        Parameters
        ----------
        block_type : str
            The type of the block (unique type from Space Engineers API).
        orientation_forward : Orientation
            The Forward orientation of the block.
        orientation_up : Orientation
            The Up orientation of the block.

        Returns
        -------
        Block
            The finalized block.
        """
        self.block_type = block_type
        self.orientation_forward = orientation_forward.value
        self.orientation_up = orientation_up.value
        self.position = Vec.v3f(0., 0., 0.)
        self.cube_size = block_definitions[self.block_type]['cube_size']
        self.size = Vec.from_json(block_definitions[self.block_type]['size'])
        self.scaled_size = self.size.scale(_blocks_sizes[self.cube_size])
        self.mass = float(block_definitions[self.block_type]['mass'])
        self.mountpoints = [MountPoint(face=v['Normal'],
                                       start=v['Start'],
                                       end=v['End'],
                                       exclusion_mask=v['ExclusionMask'],
                                       properties_mask=v['PropertiesMask'],
                                       block_size=self.size.scale(_blocks_sizes[self.cube_size])) for v in block_definitions[self.block_type]['mountpoints']]

    def __str__(self) -> str:
        return f'{self.block_type} at {self.position}; OF {self.orientation_forward}; OU {self.orientation_up}'

    def __repr__(self) -> str:
        return f'{self.block_type} at {self.position}; OF {self.orientation_forward}; OU {self.orientation_up}'

    @property
    def volume(self) -> float:
        s = self.size
        m = _blocks_sizes[self.cube_size]
        return s.x * s.y * s.z * m * m * m


class IntersectionException(Exception):
    """
    Exception to throw when an intersection occurs.
    """
    pass


class Structure:
    """
    Custom Block structure.
    Similar to `GridBlocks` in Space Engineers' API.
    """

    def __init__(self, origin: Vec, orientation_forward: Vec,
                 orientation_up: Vec) -> None:
        """
        Create a Structure object.

        Parameters
        ----------
        origin : Vec
            The XYZ origin coordinates of the Structure.
        orientation_forward : Vec
            The Forward orientation of the block as a `Vec` object.
        orientation_up : Vec
            The Up orientation of the block as a `Vec` object.

        Returns
        -------
        Structure
            The finalized structure.
        """
        # self._VALUE = 0.5
        self.origin_coords = origin
        self.orientation_forward = orientation_forward
        self.orientation_up = orientation_up
        # keeps track of blocks info
        self._blocks: Dict[Tuple(int, int, int), Block] = {}
        self.ks = list(block_definitions.keys())
        self.grid_size = 5

    def add_block(self,
                  block: Block,
                  grid_position: Tuple[int, int, int],
                  exit_on_duplicates: bool = False) -> None:
        """
        Add a block to the structure.

        Parameters
        ----------
        block : Block
            The block to add.
        grid_position : Tuple[int, int, int]
            The position in the grid at which the block is placed.
            Obtained from `Vec.to_tuple()`.
        exit_on_duplicates : bool
            Flag to check for existing blocks when adding new ones.
            May raise an `IntersectionException`.
        """
        i, j, k = grid_position
        block.position = Vec.v3i(i, j, k)
        if exit_on_duplicates and (i, j, k) in self._blocks.keys():
            raise IntersectionException
        self._blocks[(i, j, k)] = block

    @property
    def _min_dims(self) -> Tuple[int, int, int]:
        """
        Compute the minimum dimension of the Structure.

        Returns
        -------
        Tuple[int, int, int]
            The XYZ minimum dimensions
        """
        min_x, min_y, min_z = self._max_dims
        for x, y, z in self._blocks.keys():
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if z < min_z:
                min_z = z
        return min_x, min_y, min_z

    @property
    def _max_dims(self) -> Tuple[int, int, int]:
        """
        Compute the maximum dimension of the Structure.

        Returns
        -------
        Tuple[int, int, int]
            The XYZ maximum dimensions
        """
        max_x, max_y, max_z = 0, 0, 0
        for x, y, z in self._blocks.keys():
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
            if z > max_z:
                max_z = z
        return max_x, max_y, max_z

    def sanify(self) -> None:
        """
        Correct the structure's blocks to be >0 on every axis.
        """
        min_x, min_y, min_z = self._min_dims
        updated_blocks = {}
        for x, y, z in self._blocks.keys():
            block = self._blocks[(x, y, z)]
            block.position = self.origin_coords.sum(
                Vec.v3i(x=x - min_x, y=y - min_y, z=z - min_z))
            updated_blocks[(x - min_x, y - min_y, z - min_z)] = block
        self._blocks = updated_blocks

    def update(self, origin: Vec, orientation_forward: Vec,
               orientation_up: Vec) -> None:
        """
        Update the structure position and orientations.

        Parameters
        ----------
        origin : Vec
            The new XYZ origin coordinates of the Structure.
        orientation_forward : Vec
            The new Forward orientation of the block as a `Vec` object.
        orientation_up : Vec
            The new Up orientation of the block as a `Vec` object.
        """
        # update structure position and orientation
        self.origin_coords = origin
        self.orientation_forward = orientation_forward
        self.orientation_up = orientation_up
        # update all blocks accordingly
        self.sanify()

    def get_all_blocks(self,
                       to_place: bool = True) -> List[Block]:
        """
        Get all the blocks in the structure.

        Returns
        -------
        List[Block]
            The list of all blocks.
        """
        all_blocks = list(self._blocks.values())
        if to_place:
            for block in all_blocks:
                block.position = Vec.v3f(x=grid_to_coords * block.position.x,
                                         y=grid_to_coords * block.position.y,
                                         z=grid_to_coords * block.position.z)
        else:
            for block in all_blocks:
                block.position = Vec.v3i(x=int(block.position.x / self.grid_size),
                                         y=int(block.position.y / self.grid_size),
                                         z=int(block.position.z / self.grid_size))
        return all_blocks

    def as_array(self) -> np.ndarray:
        """
        Conver the structure to its equivalent NumPy array.
        Each point in the XYZ matrix represents the block type.

        Returns
        -------
        np.ndarray
            The 3D NumPy array.
        """
        max_x, max_y, max_z = self._max_dims
        structure = np.zeros(shape=(max_x + self.grid_size, max_y + self.grid_size, max_z + self.grid_size),
                             dtype=np.uint32)
        for i, j, k in self._blocks.keys():
            block = self._blocks[(i, j, k)]
            r = block.size
            r = Vec.v3f(x=r.x * _blocks_sizes[block.cube_size],
                        y=r.y * _blocks_sizes[block.cube_size],
                        z=r.z * _blocks_sizes[block.cube_size])
            v = self.ks.index(block.block_type)
            structure[i:i + r.x, j:j + r.y, k:k + r.z] = v + 1
        return structure

    def as_grid_array(self) -> np.ndarray:
        """Convert the structure to the grid-sized array.

        Returns:
            np.ndarray: The grid-sized array.
        """
        dims = self._max_dims
        dims = (int(dims[0] / self.grid_size) + 1,
                int(dims[1] / self.grid_size) + 1,
                int(dims[2] / self.grid_size) + 1)
        arr = np.zeros(shape=dims,
                       dtype=np.uint32)
        for coords, block in self._blocks.items():
            idx = (int(coords[0] / self.grid_size),
                   int(coords[1] / self.grid_size),
                   int(coords[2] / self.grid_size))
            arr[idx] = self.ks.index(block.block_type) + 1
        return arr

    def _clean_label(self,
                     a: str) -> str:
        """Remove prefix block type from label.

        Args:
            a (str): The label.

        Returns:
            str: The label without prefix.
        """
        for d in [
            'MyObjectBuilder_CubeBlock_',
            'MyObjectBuilder_Gyro_',
            'MyObjectBuilder_Reactor_',
            'MyObjectBuilder_CargoContainer_',
            'MyObjectBuilder_Cockpit_',
            'MyObjectBuilder_Thrust_',
            'MyObjectBuilder_InteriorLight_',
            'MyObjectBuilder_CubeBlock_',
        ]:
            a = a.replace(d, '')
        return a

    def show(self, title: str, title_len: int = 90, save: bool = False) -> None:
        """
        Plot the structure.

        Parameters
        ----------
        title : str
            Title of the plot.
        title_len : int
            Maximum length of the title (default: 90).
        save : bool
            Flag to salve the plot as picture (default: False).
        """
        # structure = self.as_array()
        structure = self.as_grid_array()
        ax = plt.axes(projection='3d')
        arr = np.nonzero(structure)
        x, y, z = arr
        cs = [structure[i, j, k] for i, j, k in zip(x, y, z)]
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        scatter = ax.scatter(x, y, z, c=cs, cmap='jet', linewidth=0.1)
        legend = scatter.legend_elements(
            num=len(np.unique(structure[arr])) - 1)
        for i, v in zip(range(len(legend[1])),
                        np.unique(structure[arr])):
            legend[1][i] = self._clean_label(self.ks[v - 1])
        ax.legend(*legend,
                  bbox_to_anchor=(1.2, 1),
                  loc="upper left",
                  title="Block types")
        axis_limit = max(structure.shape)
        ax.set_xlim3d(0, axis_limit)
        ax.set_ylim3d(0, axis_limit)
        ax.set_zlim3d(0, axis_limit)
        ax.set_xlabel("$\\vec{x}$")
        ax.set_ylabel("$\\vec{y}$")
        ax.set_zlabel("$\\vec{z}$")
        plot_title = title if len(title) <= title_len else title[:title_len - 3] + '...'
        plt.title(plot_title)
        plt.autoscale(enable=True,
                      axis='x',
                      tight=True)
        if save:
            plt.savefig('content_plot.png',
                        transparent=True,
                        bbox_inches='tight')
        plt.show()


def place_blocks(blocks: List[Block],
                 sequential: False) -> None:
    """
    Place the blocks in-game.

    Parameters
    ----------
    blocks : List[Block]
        The list of blocks.
    sequential : bool
        Flag to either make the `Admin.Blocks.PlaceAt` call for each block or
        for the entire list.
    """
    # prepare jsons
    jsons = [
        generate_json(
            method='Admin.Blocks.PlaceAt',
            params={
                "blockDefinitionId": block_definitions[block.block_type]['definition_id'],
                "position": block.position.as_dict(),
                "orientationForward": block.orientation_forward.as_dict(),
                "orientationUp": block.orientation_up.as_dict()
            }) for block in blocks
    ]
    # place blocks
    if not sequential:
        call_api(jsons=jsons)
    else:
        for j in jsons:
            call_api(jsons=j)


def rotate_and_normalize_block(rotation_matrix: np.ndarray,
                               normalizing_block: Block,
                               block: Block,
                               to_int: bool = True) -> Tuple[Vec, Vec, Vec]:
    of = rotate(rotation_matrix=rotation_matrix,
                vector=block.orientation_forward)
    ou = rotate(rotation_matrix=rotation_matrix,
                vector=block.orientation_up)
    
    pos = block.position.sum(normalizing_block.position.scale(-1))
    pos = rotate(rotation_matrix=rotation_matrix,
                vector=pos)
    
    return (of.to_veci(), ou.to_veci(), pos.to_veci()) if to_int else (of, ou, pos)


def try_place_block(block: Block,
                    rotation_matrix: npt.NDArray,
                    normalizing_block: Block,
                    grid_id: str) -> bool:
    of, ou, pos = rotate_and_normalize_block(rotation_matrix=rotation_matrix,
                                            normalizing_block=normalizing_block,
                                            block=block,
                                            to_int=True)    
    res = call_api(jsons=[
        generate_json(method='Admin.Blocks.PlaceInGrid',
                    params={
                        "blockDefinitionId": block_definitions[block.block_type]['definition_id'],
                        "gridId": grid_id,
                        "minPosition": pos.as_dict(),
                        "orientationForward": of.as_dict(),
                        "orientationUp": ou.as_dict()
                    })
    ])
    return res[0].get('error', None) is None


class PlacementException(Exception):
    pass


def place_structure(structure: Structure,
                    position: Vec,
                    orientation_forward: Vec = Orientation.FORWARD.value,
                    orientation_up: Vec = Orientation.UP.value,
                    batchify: bool = True) -> None:
    """
    Place the structure in-game.

    Parameters
    ----------
    structure : Structure
        The structure to place.
    position : Vec
        The minimum position of the structure to place at.
    orientation_forward : Vec
        The Forward orientation, as vector.
    orientation_up : Vec
        The Up orientation, as vector.
    """
    # ensure structure position and orientation
    structure.update(
        origin=Vec.v3f(0., 0., 0.),
        orientation_forward=orientation_forward,
        orientation_up=orientation_up,
    )
    structure.sanify()
    to_place = structure.get_all_blocks(to_place=False)
    # toggle gamemode to place faster
    toggle_gamemode(GameMode.PLACING)
    # get lowest-index block in structure
    first_block = None
    for block in to_place:
        if first_block:
            if block.position.x <= first_block.position.x and block.position.y <= first_block.position.y and block.position.z <= first_block.position.z:
                first_block = block
        else:
            first_block = block
    # remove first block from list
    to_place.remove(first_block)
    # place first block
    call_api(jsons=[
        generate_json(
            method='Admin.Blocks.PlaceAt',
            params={
                "blockDefinitionId": block_definitions[first_block.block_type]['definition_id'],
                "position": position.as_dict(),
                "orientationForward": first_block.orientation_forward.as_dict(),
                "orientationUp": first_block.orientation_up.as_dict()
            })
    ])
    # get placed block's grid
    observation = call_api(jsons=[generate_json(method='Observer.ObserveBlocks', params={})])
    grid_id = observation[0]["result"]["Grids"][0]["Id"]
    grid_orientation_forward = Vec.from_json(observation[0]["result"]["Grids"][0]["OrientationForward"])
    grid_orientation_up = Vec.from_json(observation[0]["result"]["Grids"][0]["OrientationUp"])
    rotation_matrix = get_rotation_matrix(forward=grid_orientation_forward,
                                        up=grid_orientation_up)
    
    # TODO: Move character away so that the spaceship can be built entirely
    
    # reorder blocks
    occupied_space = [first_block.position]
    ordered_blocks = []
    while to_place:
        to_rem = []
        for block in to_place:
            if (block.position.sum(Vec.v3i(-1, 0, 0)) in occupied_space or
                block.position.sum(Vec.v3i(0, -1, 0)) in occupied_space or
                block.position.sum(Vec.v3i(0, 0, -1)) in occupied_space or
                block.position.sum(Vec.v3i(1, 0, 0)) in occupied_space or
                block.position.sum(Vec.v3i(0, 1, 0)) in occupied_space or
                block.position.sum(Vec.v3i(0, 0, 1)) in occupied_space):
                to_rem.append(block)
                occupied_space.append(block.position)
                ordered_blocks.append(block)
        for r in to_rem:
            to_place.remove(r)
    if batchify:
        # attempt placement in batches
        batch_size = 64
        jsons = []
        for n, block in enumerate(ordered_blocks):
            of, ou, pos = rotate_and_normalize_block(rotation_matrix=rotation_matrix,
                                                    normalizing_block=first_block,
                                                    block=block,
                                                    to_int=True)
            jsons.append(generate_json(method='Admin.Blocks.PlaceInGrid',
                                       params={
                                           "blockDefinitionId": block_definitions[block.block_type]['definition_id'],
                                           "gridId": grid_id,
                                           "minPosition": pos.as_dict(),
                                           "orientationForward": of.as_dict(),
                                           "orientationUp": ou.as_dict()},
                                       request_id=n))
        n_requests = 0
        while jsons:
            to_rem = []
            for (idx_from, idx_to) in get_batch_ranges(batch_size=batch_size,
                                                       length=len(jsons)):
                res_list = call_api(jsons=jsons[idx_from:idx_to])
                for res in res_list:
                    if res.get('error', None) is None:
                        to_rem.append(res)
            for res in to_rem:
                for i, req in enumerate(jsons):
                    if req['id'] == res['id']:
                        jsons.pop(i)
                        break
            if len(jsons) == n_requests:
                raise PlacementException(f'Error during spaceship placement: missing {len(jsons)} blocks to place.')
            else:
                n_requests = len(jsons)
    else:
        # attempt placement sequentially
        errored_out = []
        for block in ordered_blocks:
            # always try to place blocks that we failed to place previously
            to_rem = []
            for b in errored_out:
                res = try_place_block(block=b,
                                    rotation_matrix=rotation_matrix,
                                    normalizing_block=first_block,
                                    grid_id=grid_id)
                if res:
                    to_rem.append(b)
            for b in to_rem:
                errored_out.remove(b)
            # try and place current block
            res = try_place_block(block=block,
                                rotation_matrix=rotation_matrix,
                                normalizing_block=first_block,
                                grid_id=grid_id)
            if not res:
                errored_out.append(block)
        if len(errored_out) != 0:
            raise PlacementException(f'Error during spaceship placement: missing {len(errored_out)} blocks to place.')
    # toggle back gamemode
    toggle_gamemode(GameMode.EVALUATING)
