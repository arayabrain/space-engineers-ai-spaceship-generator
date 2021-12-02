import json
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from .common.vecs import Orientation, Vec, rotate
from .common.api_call import call_api, generate_json
from .config import FUSE_OVERLAPS
from .lsystem.constraints import HLStructure, build_polyhedron
from typing import Any, Dict, List, Tuple


# block_definitions as a module-level variable
if not os.path.exists('./block_definitions.json'):
    # poll API for block definition ids
    jsons = [
        generate_json(method="Definitions.BlockDefinitions")
    ]
    res = call_api(jsons=jsons)[0]
    # transform to map of type:id
    block_definitions = {v['Type']: v for v in [
        entry['DefinitionId'] for entry in res['result']]}
    with open('./block_definitions.json', 'w') as f:
        json.dump(block_definitions, f)
else:
    with open('./block_definitions.json', 'r') as f:
        block_definitions = json.load(f)

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
        self._type = 's' if self.block_type.startswith('Small') else (
            'l'if self.block_type.startswith('Large') else 'n')
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
        # self._structure = HLStructure()  # keeps track of occupied spaces
        self._VALUE = 0.5
        self.origin_coords = origin
        self.orientation_forward = orientation_forward
        self.orientation_up = orientation_up
        # keeps track of blocks info
        self._blocks: Dict[Tuple(int, int, int), Block] = {}

    def add_block(self,
                  block: Block,
                  grid_position: Tuple[int, int, int]) -> None:
        i, j, k = grid_position
        # cube = build_polyhedron(position=Vec.v3i(i, j, k),
        #                         dims=Vec.v3i(block.size * self._VALUE,
        #                                      block.size * self._VALUE,
        #                                      block.size * self._VALUE))
        # self._structure.add_hl_poly(p=cube)
        # if self._structure.test_intersections():
        #     raise Exception(f'Error when placing block {block.block_type}: block intersect with existing block(s))')
        self._blocks[(i, j, k)] = block

    @property
    def _min_dims(self) -> Tuple[int, int, int]:
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
        min_x, min_y, min_z = self._min_dims
        updated_blocks = {}
        for x, y, z in self._blocks.keys():
            block = self._blocks[(x, y, z)]
            self.origin_coords.sum(
                Vec.v3i(x=x - min_x, y=y - min_y, z=z - min_z))
            updated_blocks[(x - min_x, y - min_y, z - min_z)] = block
        self._blocks = updated_blocks

    def update(self,
               origin: Vec,
               orientation_forward: Vec,
               orientation_up: Vec) -> None:
        # update structure position and orientation
        self.origin_coords = origin
        self.orientation_forward = orientation_forward
        self.orientation_up = orientation_up
        # update all blocks accordingly
        self.sanify()

    def get_all_blocks(self) -> List[Block]:
        return list(self._blocks.values())

    def show(self,
             title: str,
             title_len: int = 90,
             save: bool = False) -> None:
        max_x, max_y, max_z = self._max_dims
        structure = np.zeros(shape=(max_x + 5, max_y + 5, max_z + 5),
                             dtype=np.uint32)
        ks = list(block_definitions.keys())
        for i, j, k in self._blocks.keys():
            block = self._blocks[(i, j, k)]
            r = block.size
            v = ks.index(block.block_type)
            structure[i:i+r, j:j+r, k:k+r] = v + 1

        ax = plt.axes(projection='3d')
        arr = np.nonzero(structure)
        x, y, z = arr
        cs = [structure[i, j, k] for i, j, k in zip(x, y, z)]
        scatter = ax.scatter(x, y, z, c=cs, cmap='jet', linewidth=0.1)
        legend = scatter.legend_elements()
        for i, v in zip(range(len(legend[1])), np.unique(structure[np.nonzero(structure)])):
            legend[1][i] = ks[v - 1]
        ax.legend(*legend, bbox_to_anchor=(1.2, 1),
                  loc="upper left", title="Block types")
        ax.set_xlim3d(0, max_x + 5)
        ax.set_ylim3d(0, max_y + 5)
        ax.set_zlim3d(0, max_z + 5)
        ax.set_xlabel("$\\vec{x}$")
        ax.set_ylabel("$\\vec{y}$")
        ax.set_zlabel("$\\vec{z}$")
        plot_title = title if len(
            title) <= title_len else title[:title_len - 3] + '...'
        plt.title(plot_title)
        if save:
            plt.savefig(f'{title}.png', transparent=True)
        plt.show()


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
