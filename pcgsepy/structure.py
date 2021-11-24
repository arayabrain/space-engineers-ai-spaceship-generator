import json
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from .common.vecs import Orientation, Vec, rotate, get_rotation_matrix
from .common.api_call import call_api, generate_json
from .config import FUSE_OVERLAPS
from typing import Any, Dict, List, Tuple


# block_definitions as a module-level variable
if not os.path.exists('./block_definitions.json'):
    # poll API for block definition ids
    jsons = [ 
        generate_json(method="Definitions.BlockDefinitions")
        ]
    res = call_api(jsons=jsons)[0]
    # transform to map of type:id
    block_definitions = {v['Type']:v for v in [entry['DefinitionId'] for entry in res['result']]}
    with open('./block_definitions.json', 'w') as f:
        json.dump(block_definitions, f)
else:
    with open('./block_definitions.json', 'r') as f:
        block_definitions = json.load(f)

_blocks_dims = {
    's': 0.5,  # small
    'n': 1.,  # normal
    'l': 2.5  # large
}

_blocks_sizes = {
    's': 1,
    'n': 2,
    'l': 5
}


class Block:
    def __init__(self,
                 block_type: str,
                 orientation_forward: Orientation = Orientation.FORWARD,
                 orientation_up: Orientation = Orientation.UP):
        self.block_type = block_type
        self.orientation_forward = orientation_forward.value
        self.orientation_up = orientation_up.value
        self.position = Vec.v3f(0., 0., 0.)
        self._type = 's' if self.block_type.startswith('Small') else ('l'if self.block_type.startswith('Large') else 'n')
        self.dim = _blocks_dims[self._type]
        self.size = _blocks_sizes[self._type]
    
    def __str__(self) -> str:
        return f'{self.block_type} at {self.position}; OF {self.orientation_forward}; OU {self.orientation_up}'
    
    def __repr__(self) -> str:
        return f'{self.block_type} at {self.position}; OF {self.orientation_forward}; OU {self.orientation_up}'


class Structure:
    def __init__(self,
                 origin: Vec,
                 orientation_forward: Vec,
                 orientation_up: Vec,
                 dimensions: Tuple[int, int, int] = (10, 10, 10)) -> None:
        self._VALUE = 0.5
        self.origin_coords = origin
        self.orientation_forward = orientation_forward
        self.orientation_up = orientation_up
        self.rotation_matrix = get_rotation_matrix(forward=self.orientation_forward,
                                                   up=self.orientation_up)
        self.dimensions = dimensions
        self._structure: np.ndarray = 0.5 * np.ones(dimensions)  # keeps track of occupied spaces
        self._blocks: Dict[float, Block] = {}  # keeps track of blocks info
    
    @property
    def starting_coords(self) -> Tuple[int, int, int]:
        x, y, z = self.dimensions
        return Vec.v3i(x // 2, y // 2, z // 2)

    def add_block(self,
                  block: Block,
                  grid_position: Tuple[int, int, int]) -> None:
        i, j, k = grid_position
        if FUSE_OVERLAPS and self._structure[i][j][k] != 0.5: return
        assert self._structure[i][j][k] == 0.5, f'Error when attempting to place block {block.block_type}: space already occupied.'
        r = block.size
        if r != 1:
            # update neighbouring cells
            n, target = np.sum(self._structure[i:i+r, j:j+r, k:k+r]) - self._VALUE, self._VALUE * ((r ** 3) - 1)
            if FUSE_OVERLAPS and n != target: return
            assert n == target, f'Error when placing block {block.block_type}: block would intersect already existing block(s).'
        block_id = float(self._VALUE + len(self._blocks.keys()) + 1)  # unique sequential block id as key in dict
        self._structure[i:i+r, j:j+r, k:k+r] = block_id  # volume of block id
        self._blocks[block_id] = {'block': block,
                                  'grid_idxs': (i, j, k)}
        self._update_block_pos_rot(block, (i, j, k))

    def _update_block_pos_rot(self,
                             block: Block,
                             grid_coords: Tuple[int, int, int]) -> None:
        i, j, k = grid_coords
        # update block position
        dx = self._compute_offset('x', i, j, k)
        dy = self._compute_offset('y', i, j, k)
        dz = self._compute_offset('z', i, j, k)
        block.position = self.origin_coords.sum(Vec.v3f(dx, dy, dz))
        # update block orientation
        # block.orientation_forward = rotate(self.rotation_matrix, block.orientation_forward)
        # block.orientation_up = rotate(self.rotation_matrix, block.orientation_up)
    
    def update(self,
               origin: Vec,
               orientation_forward: Vec,
               orientation_up: Vec) -> None:
        # update structure position and orientation
        self.origin_coords = origin
        self.orientation_forward = orientation_forward
        self.orientation_up = orientation_up
        # update all blocks accordingly
        blocks = self.get_all_blocks()
        grid_coords = [x['grid_idxs'] for x in self._blocks.values()]
        for block, coords in zip(blocks, grid_coords):
            self._update_block_pos_rot(block=block,
                                       grid_coords=coords)

    def _compute_offset(self,
                        dimension: str,
                        i: int = 0,
                        j: int = 0,
                        k: int = 0) -> float:
        if dimension == 'x' and i > 0:
            vs = self._structure[:i, j, k]
        elif dimension == 'y' and j > 0:
            vs = self._structure[i, :j, k]
        elif dimension == 'z' and k > 0:
            vs = self._structure[i, j, :k]
        else:
            return 0.
        n = 0
        offset = 0.
        while n < len(vs):
            v = vs[n]
            if v > self._VALUE:
                offset +=  self._blocks[v]['block'].dim
                n += self._blocks[v]['block'].size
            else:
                offset += v
                n += 1
        return offset
    
    def get_all_blocks(self) -> List[Block]:
        return [b['block'] for b in self._blocks.values()]


def place_blocks(blocks: List[Block],
                 sequential: False) -> None:
    # prepare jsons
    jsons = [generate_json(method="Admin.Blocks.PlaceAt",
                           params={
                               "blockDefinitionId": block_definitions[block.block_type],
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

def plot_structure(structure: Structure,
                   title: str,
                   axis_limits: Tuple[int, int, int],
                   title_len: int = 30,
                   save: bool = False) -> None:
        ax = plt.axes(projection='3d')
        arr = structure._structure != structure._VALUE
        x, y, z = np.nonzero(arr)
        ax.scatter(x, y, z, c=np.ones(len(x)), cmap='jet', linewidth=0.1)
        ax.set_xlim3d(0, axis_limits[0])
        ax.set_ylim3d(0, axis_limits[1])
        ax.set_zlim3d(0, axis_limits[2])
        ax.set_xlabel("$\\vec{x}$")
        ax.set_ylabel("$\\vec{y}$")
        ax.set_zlabel("$\\vec{z}$")
        title = title if len(title) <= title_len else title[:title_len - 3] + '...'
        plt.title(title)
        if save:
            plt.savefig(f'{title}.png', transparent=True)
        plt.show()
