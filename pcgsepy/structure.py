import json
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import os
from typing import Dict, List, Optional, Tuple

from .common.vecs import Orientation, Vec
from .common.api_call import call_api, generate_json

# block_definitions as a module-level variable
if not os.path.exists('./block_definitions.json'):
    # poll API for block definition ids
    jsons = [generate_json(method="Definitions.BlockDefinitions")]
    res = call_api(jsons=jsons)[0]
    # transform to map of type:id
    block_definitions = {
        v['Type']: v
        for v in [entry['DefinitionId'] for entry in res['result']]
    }
    with open('./block_definitions.json', 'w') as f:
        json.dump(block_definitions, f)
else:
    with open('./block_definitions.json', 'r') as f:
        block_definitions = json.load(f)

# Sizes of blocks in grid spaces
_blocks_sizes = {'s': 1, 'n': 2, 'l': 5}


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
        self._type = 's' if self.block_type.startswith('Small') else (
            'l' if self.block_type.startswith('Large') else 'n')
        self.size = _blocks_sizes[self._type]

    def __str__(self) -> str:
        return f'{self.block_type} at {self.position}; OF {self.orientation_forward}; OU {self.orientation_up}'

    def __repr__(self) -> str:
        return f'{self.block_type} at {self.position}; OF {self.orientation_forward}; OU {self.orientation_up}'


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
        self._VALUE = 0.5
        self.origin_coords = origin
        self.orientation_forward = orientation_forward
        self.orientation_up = orientation_up
        # keeps track of blocks info
        self._blocks: Dict[Tuple(int, int, int), Block] = {}
        self.ks = list(block_definitions.keys())

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
        # block.position = Vec.v3f(i * self._VALUE, j * self._VALUE,
        #                          k * self._VALUE)
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
        min_x, min_y, min_z = 0, 0, 0
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
            # block.position = self.origin_coords.sum(
            #     Vec.v3i(x=(x - min_x) * self._VALUE,
            #             y=(y - min_y) * self._VALUE,
            #             z=(z - min_z) * self._VALUE))
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

    def get_all_blocks(self) -> List[Block]:
        """
        Get all the blocks in the structure.

        Returns
        -------
        List[Block]
            The list of all blocks.
        """
        return list(self._blocks.values())

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
        structure = np.zeros(shape=(max_x + 5, max_y + 5, max_z + 5),
                             dtype=np.uint32)
        for i, j, k in self._blocks.keys():
            block = self._blocks[(i, j, k)]
            r = block.size
            v = self.ks.index(block.block_type)
            structure[i:i + r, j:j + r, k:k + r] = v + 1
        return structure

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
        structure = self.as_array()
        ax = plt.axes(projection='3d')
        arr = np.nonzero(structure)
        x, y, z = arr
        cs = [structure[i, j, k] for i, j, k in zip(x, y, z)]
        scatter = ax.scatter(x, y, z, c=cs, cmap='jet', linewidth=0.1)
        legend = scatter.legend_elements()
        for i, v in zip(range(len(legend[1])),
                        np.unique(structure[np.nonzero(structure)])):
            legend[1][i] = self.ks[v - 1]
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
        plot_title = title if len(title) <= title_len else title[:title_len -
                                                                 3] + '...'
        plt.title(plot_title)
        if save:
            plt.savefig(f'{title}.png', transparent=True)
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
                "blockDefinitionId": block_definitions[block.block_type],
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


def place_structure(structure: Structure,
                    position: Vec,
                    orientation_forward: Vec = Orientation.FORWARD.value,
                    orientation_up: Vec = Orientation.UP.value) -> None:
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
    # ensure grid positionment
    structure.update(
        origin=Vec.v3f(0., 0., 0.),
        orientation_forward=orientation_forward,
        orientation_up=orientation_up,
    )
    structure.sanify()
    all_blocks = structure.get_all_blocks()
    # get lowest-index block in structure
    first_block = None
    for block in all_blocks:
        if first_block:
            if block.position.x <= first_block.position.x and block.position.y <= first_block.position.y and block.position.z <= first_block.position.z:
                first_block = block
        else:
            first_block = block
    # remove first block from list
    all_blocks.remove(first_block)
    # update first block's position
    first_block.position = position
    # place first block
    call_api(jsons=[
        generate_json(
            method='Admin.Blocks.PlaceAt',
            params={
                "blockDefinitionId": block_definitions[first_block.block_type],
                "position": first_block.position.as_dict(),
                "orientationForward": first_block.orientation_forward.as_dict(),
                "orientationUp": first_block.orientation_up.as_dict()
            })
    ])
    # get placed block's grid
    observation = call_api(
        jsons=[generate_json(method='Observer.ObserveBlocks', params={})])
    grid_id = observation[0]["result"]["Grids"][0]["Id"]
    # place all other blocks in the grid
    call_api(jsons=[
        generate_json(
            method='Admins.Blocks.PlaceInGrid',
            params={
                "blockDefinitionId": block_definitions[block.block_type],
                "gridID": grid_id,
                "minPosition": block.position.as_dict(),
                "orientationForward": block.orientation_forward.as_dict(),
                "orientationUp": block.orientation_up.as_dict()
            }) for block in all_blocks
    ])
