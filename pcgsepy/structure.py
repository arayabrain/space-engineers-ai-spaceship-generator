import json
import numpy as np
import os
from .common.vecs import Orientation, Vec, rotate, get_rotation_matrix
from .common.api_call import call_api, generate_json
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


class Block:
    def __init__(self,
                 block_type: str,
                 orientation_forward: Orientation = Orientation.FORWARD,
                 orientation_up: Orientation = Orientation.UP):
        self.block_type = block_type
        self.orientation_forward = orientation_forward.value
        self.orientation_up = orientation_up.value
        self.position = Vec.v3f(0., 0., 0.)
    
    def __str__(self) -> str:
        return f'{self.block_type} at {self.position}; OF {self.orientation_forward}; OU {self.orientation_up}'
    
    def __repr__(self) -> str:
        return f'{self.block_type} at {self.position}; OF {self.orientation_forward}; OU {self.orientation_up}'

blocks_dims = {
    's': 0.5,  # small
    'n': 1.,  # normal
    'l': 2.5  # large
}

def get_block_size(block: Block) -> str:
    return 's' if block.block_type.startswith('Small') else ('l'if block.block_type.startswith('Large') else 'n')

def get_block_dim(block: Block) -> float:
    return blocks_dims[get_block_size(block)]


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
            
    def add_block(self,
                  block: Block,
                  grid_position: Tuple[int, int, int]) -> None:
        i, j, k = grid_position
        assert self._structure[i][j][k] == 0.5, f'Error when attempting to place block {block.block_type}: space already occupied.'
        block_size = get_block_size(block)
        if block_size != 's':
            # update neighbouring cells
            r = 5 if block_size == 'l' else 2
            n, target = np.sum(self._structure[i:i+r, j:j+r, k:k+r]) - self._VALUE, self._VALUE * ((r ** 3) - 1)
            assert n == target, f'Error when placing block {block.block_type}: block would intersect already existing block(s).'
            self._structure[i:i+r, j:j+r, k:k+r] = np.zeros((r,r,r))
        block_id = float(self._VALUE + len(self._blocks.keys()) + 1)  # unique sequential block id as key in dict
        self._structure[i, j, k] = block_id
        self._blocks[block_id] = block
        # update block position
        dx, dy, dz = 0., 0., 0.
        if i > 0:
            for e in self._structure[0:i, j, k]:
                dx += get_block_dim(self._blocks[e]) if e > self._VALUE else e
        if j > 0:
            for e in self._structure[i, 0:j, k]:
                dy += get_block_dim(self._blocks[e]) if e > self._VALUE else e
        if k > 0:
            for e in self._structure[i, j, 0:k]:
                dz += get_block_dim(self._blocks[e]) if e > self._VALUE else e        
        block.position = self.origin_coords.sum(Vec.v3f(dx, dy, dz))
        # update block orientation
#         block.orientation_forward = rotate(self.rotation_matrix, block.orientation_forward)
#         block.orientation_up = rotate(self.rotation_matrix, block.orientation_up)
    
    def get_all_blocks(self) -> List[Block]:
        return list(self._blocks.values())

def place_blocks(blocks: List[Block]) -> None:
    # prepare jsons
    jsons = [generate_json(method="Admin.Blocks.PlaceAt",
                           params={
                               "blockDefinitionId": block_definitions[block.block_type],
                               "position": block.position.as_dict(),
                               "orientationForward": block.orientation_forward.as_dict(),
                               "orientationUp": block.orientation_up.as_dict()
                           }) for block in blocks]
    # place blocks
    call_api(jsons=jsons)