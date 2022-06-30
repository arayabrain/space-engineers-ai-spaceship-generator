from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import grey_erosion, binary_erosion
import numpy as np
from pcgsepy.common.vecs import Orientation, Vec, orientation_from_vec
from pcgsepy.structure import Block, Structure
from typing import Optional, Tuple

from typing import Tuple
from pcgsepy.structure import Block

def vec_to_idx(v: Vec) -> Tuple[int, int, int]:
    return (v.x, v.y, v.z)

def idx_to_vec(idx: Tuple[int, int, int]) -> Vec:
    return Vec(x=idx[0], y=idx[1], z=idx[2])

def is_slope_block(block: Block) -> bool:
    return "Slope" in block.block_type or "Corner" in block.block_type

class HullBuilder:
    def __init__(self,
                 erosion_type: str,
                 apply_erosion: bool,
                 apply_smoothing: bool):
        self.AIR_BLOCK_VALUE = 0
        self.BASE_BLOCK_VALUE = 1
        self.SLOPE_BLOCK_VALUE = 2
        self.CORNER_BLOCK_VALUE = 3
        self.CORNERINV_BLOCK_VALUE = 4
        self.available_erosion_types = ['grey', 'bin']
        self.erosion_type = erosion_type
        assert self.erosion_type in self.available_erosion_types, f'Unrecognized erosion type {self.erosion_type}; available are {self.available_erosion_types}.'
        if self.erosion_type == 'grey':
            self.erosion = grey_erosion
            self.footprint=[
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, False]
                ],
                [
                    [False, True, False],
                    [True, True, True],
                    [False, True, False]
                ],
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, False]
                ]
            ]
        elif self.erosion_type == 'bin':
            self.erosion = binary_erosion
            self.iterations = 2
        self.apply_erosion = apply_erosion
        self.apply_smoothing = apply_smoothing
        
        self.base_block = 'MyObjectBuilder_CubeBlock_LargeBlockArmorBlock'
        self._blocks_set = {}
    
    def _get_convex_hull(self,
                         arr: np.ndarray) -> np.ndarray:
        """Compute the convex hull of the given array.

        Args:
            arr (np.ndarray): The Structure's array.

        Returns:
            np.ndarray: The convex hull.
        """
        points = np.transpose(np.where(arr))
        hull = ConvexHull(points)
        deln = Delaunay(points[hull.vertices])
        idx = np.stack(np.indices(arr.shape), axis=-1)
        out_idx = np.nonzero(deln.find_simplex(idx) + 1)
        out_arr = np.zeros(arr.shape)
        out_arr[out_idx] = self.BASE_BLOCK_VALUE
        # out_arr[np.nonzero(arr)] = arr[np.nonzero(arr)]
        return out_arr
        
    def _adj_to_spaceship(self,
                          i: int,
                          j: int,
                          k: int,
                          spaceship: np.ndarray) -> bool:
        """Check coordinates adjacency to original spaceship hull.

        Args:
            i (int): The i coordinate
            j (int): The j coordiante
            k (int): The k coordinate
            spaceship (np.ndarray): The original spaceship hull

        Returns:
            bool: Whether the coordinate is adjacent to the original spaceship
        """
        adj = False
        for di, dj, dk in zip([+1, 0, 0, 0, 0, -1], [0, +1, 0, 0, -1, 0], [0, 0, +1, -1, 0, 0]):
            if 0 < i + di < spaceship.shape[0] and 0 < j + dj < spaceship.shape[1] and 0 < k + dk < spaceship.shape[2]:
                adj |= spaceship[i + di, j + dj, k + dk] != 0
        return adj

    def _add_block(self,
                   block_type: str,
                   structure: Structure,
                   pos: Tuple[int, int, int],
                   orientation_forward: Orientation = Orientation.FORWARD,
                   orientation_up: Orientation = Orientation.UP) -> None:
        """Add the block to the structure.

        Args:
            block_type (str): The block type.
            structure (Structure): The structure.
            pos (Tuple[int, int, int]): The grid coordinates (non-grid-size specific)
            orientation_forward (Orientation, optional): The forward orientation of the block. Defaults to Orientation.FORWARD.
            orientation_up (Orientation, optional): The up orientation of the block. Defaults to Orientation.UP.
        """
        i, j, k = pos
        block = Block(block_type=block_type,
                      orientation_forward=orientation_forward,
                      orientation_up=orientation_up)
        block.position = Vec.v3i(x=int(i * structure.grid_size),
                                 y=int(j * structure.grid_size),
                                 z=int(k * structure.grid_size))
        self._blocks_set[(i, j, k)] = block

    def _tag_internal_air_blocks(self,
                                 arr: np.ndarray):
        air_blocks = np.zeros(shape=arr.shape)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    if sum(arr[0:i, j, k]) != 0 and \
                        sum(arr[i:arr.shape[0], j, k]) != 0 and \
                        sum(arr[i, 0:j, k]) != 0 and \
                        sum(arr[i, j:arr.shape[1], k]) != 0 and \
                        sum(arr[i, j, 0:k]) != 0 and \
                        sum(arr[i, j, k:arr.shape[2]]) != 0:
                            air_blocks[i, j, k] = self.BASE_BLOCK_VALUE
        return air_blocks
        
    def _exists_block(self,
                      idx: Tuple[int, int, int],
                      structure: Structure) -> bool:
        return structure._blocks.get(idx, None) is not None
    
    def _is_valid_block(self,
                        loc: Vec,
                        structure: Structure,
                        hull: np.typing.NDArray) -> bool:
        return (self._exists_block(idx=vec_to_idx(v=loc), structure=structure) and not is_slope_block(structure._blocks.get(vec_to_idx(v=loc), None))) or hull[vec_to_idx(v=loc.scale(1 / structure.grid_size).to_veci())] == self.BASE_BLOCK_VALUE
    
    def _is_air_block(self,
                      loc: Vec,
                      structure: Structure,
                      hull: np.typing.NDArray) -> bool:
        return not self._exists_block(idx=vec_to_idx(v=loc), structure=structure) and hull[vec_to_idx(v=loc.scale(1 / structure.grid_size).to_veci())] == self.AIR_BLOCK_VALUE
    
    def _pointing_against(self,
                          loc: Vec,
                          structure: Structure,
                          direction: Vec) -> bool:
        direction = direction.scale(v=1 / structure.grid_size)
        print(f'Observing block at {loc} via {direction}:')
        if self._exists_block(vec_to_idx(v=loc), structure=structure):
            obs_block = structure._blocks.get(vec_to_idx(v=loc))
        else:
            obs_block = self._blocks_set.get(vec_to_idx(v=loc.scale(1 / structure.grid_size).to_veci()))
            if obs_block is None:
                return False
        # return obs_block.orientation_forward == direction.opposite() or obs_block.orientation_up == direction.opposite()            
        return obs_block.orientation_up == direction.opposite()            
    
    def simple_conversion(self,
                          idx: Tuple[int, int, int],
                          hull: np.typing.NDArray,
                          structure: Structure) -> Optional[Block]:
        i, j, k = idx
        scale = structure.grid_size
        loc = idx_to_vec(idx=(scale * i, scale * j, scale * k))
        
        # for slope, this is a test
        dd, du = Orientation.DOWN.value.scale(v=scale), Orientation.UP.value.scale(v=scale)
        dr, dl = Orientation.RIGHT.value.scale(v=scale), Orientation.LEFT.value.scale(v=scale)
        df, db = Orientation.FORWARD.value.scale(v=scale), Orientation.BACKWARD.value.scale(v=scale)
        
        # debugging
        print(f'\n\nBlock at {i} {j}, {k}:')
        
        # print(f' {hull[i,j+1,k+1]} \n{hull[i-1,j,k+1]} {hull[i+1,j,k+1]}\n {hull[i,j-1,k+1]}')
        # print(f' {hull[i,j+1,k]} \n{hull[i-1,j,k]} {hull[i+1,j,k]}\n {hull[i,j-1,k]}')
        # print(f' {hull[i,j+1,k-1]} \n{hull[i-1,j,k-1]} {hull[i+1,j,k-1]}\n {hull[i,j-1,k-1]}')
        
        # removal check
        # slopes checks
        for direction in [dd, du, dr, dl, df, db]:
            if not self._is_valid_block(loc=loc.sum(direction), structure=structure, hull=hull) and \
                self._pointing_against(loc=loc.sum(direction), structure=structure, direction=direction):
                    return None, self.AIR_BLOCK_VALUE
        
        # case-based slope assignment
        if hull[i, j, k] == self.BASE_BLOCK_VALUE:
            # slope connecting DOWN-LEFT requires air UP-RIGHT
            if self._is_valid_block(loc=loc.sum(dd), structure=structure, hull=hull) and \
                self._is_valid_block(loc=loc.sum(dl), structure=structure, hull=hull) and \
                    self._is_air_block(loc=loc.sum(du), structure=structure, hull=hull) and \
                    self._is_air_block(loc=loc.sum(dr), structure=structure, hull=hull):
                return Block(block_type='MyObjectBuilder_CubeBlock_LargeBlockArmorSlope',
                                orientation_forward=Orientation.DOWN,
                                orientation_up=Orientation.RIGHT), self.SLOPE_BLOCK_VALUE
            # slope connecting DOWN-RIGHT requires air UP-LEFT
            elif self._is_valid_block(loc=loc.sum(dd), structure=structure, hull=hull) and \
                self._is_valid_block(loc=loc.sum(dr), structure=structure, hull=hull) and \
                    self._is_air_block(loc=loc.sum(du), structure=structure, hull=hull) and \
                    self._is_air_block(loc=loc.sum(dl), structure=structure, hull=hull):
                return Block(block_type='MyObjectBuilder_CubeBlock_LargeBlockArmorSlope',
                                orientation_forward=Orientation.DOWN,
                                orientation_up=Orientation.LEFT), self.SLOPE_BLOCK_VALUE
            # slope connecting UP-LEFT requires air DOWN-RIGHT
            elif self._is_valid_block(loc=loc.sum(du), structure=structure, hull=hull) and \
                self._is_valid_block(loc=loc.sum(dl), structure=structure, hull=hull) and \
                    self._is_air_block(loc=loc.sum(dd), structure=structure, hull=hull) and \
                    self._is_air_block(loc=loc.sum(dr), structure=structure, hull=hull):
                return Block(block_type='MyObjectBuilder_CubeBlock_LargeBlockArmorSlope',
                                orientation_forward=Orientation.UP,
                                orientation_up=Orientation.RIGHT), self.SLOPE_BLOCK_VALUE
            # slope connecting UP-RIGHT requires air DOWN-LEFT
            elif self._is_valid_block(loc=loc.sum(du), structure=structure, hull=hull) and \
                self._is_valid_block(loc=loc.sum(dr), structure=structure, hull=hull) and \
                    self._is_air_block(loc=loc.sum(dd), structure=structure, hull=hull) and \
                    self._is_air_block(loc=loc.sum(dl), structure=structure, hull=hull):
                return Block(block_type='MyObjectBuilder_CubeBlock_LargeBlockArmorSlope',
                                orientation_forward=Orientation.UP,
                                orientation_up=Orientation.LEFT), self.SLOPE_BLOCK_VALUE

        return None, self.BASE_BLOCK_VALUE
        
    def add_external_hull(self,
                          structure: Structure) -> None:
        """Add an external hull to the given Structure.
        This process adds the hull blocks directly into the Structure, so it can be used only once per spaceship.

        Args:
            structure (Structure): The spaceship.
        """
        arr = structure.as_grid_array()
        air = self._tag_internal_air_blocks(arr=arr)
        hull = self._get_convex_hull(arr=arr)
        hull[np.nonzero(air)] = self.AIR_BLOCK_VALUE
        hull[np.nonzero(arr)] = self.AIR_BLOCK_VALUE
        
        
        if self.apply_erosion:
            if self.erosion_type == 'grey':
                hull = grey_erosion(input=hull,
                                    footprint=self.footprint,
                                    mode='constant',
                                    cval=1)
                hull = hull.astype(int)
                hull *= self.BASE_BLOCK_VALUE                
            elif self.erosion_type == 'bin':
                mask = np.zeros(arr.shape)
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        for k in range(mask.shape[2]):
                            mask[i, j, k] = self.AIR_BLOCK_VALUE if self._adj_to_spaceship(i=i, j=j, k=k, spaceship=arr) else self.BASE_BLOCK_VALUE
                hull = binary_erosion(input=hull,
                                      mask=mask,
                                      iterations=self.iterations)
                hull = hull.astype(int)
                hull *= self.BASE_BLOCK_VALUE
        
        # add blocks to self._blocks_set
        for i in range(hull.shape[0]):
                for j in range(hull.shape[1]):
                    for k in range(hull.shape[2]):
                        if hull[i, j, k] != self.AIR_BLOCK_VALUE:
                            self._add_block(block_type=self.base_block,
                                            structure=structure,
                                            pos=(i, j, k),
                                            orientation_forward=Orientation.FORWARD,
                                            orientation_up=Orientation.UP)
        
        if self.apply_smoothing:
            modified = 1
            while modified != 0:
                modified = 0
                to_rem = []      
                for (i, j, k), block in self._blocks_set.items():
                    substitute_block, val = self.simple_conversion(idx=(i, j, k),
                                                                hull=hull,
                                                                structure=structure)
                    if substitute_block is not None:
                        substitute_block.position = block.position
                        self._blocks_set[(i, j, k)] = substitute_block
                        modified += 1
                    elif substitute_block is None and val == self.AIR_BLOCK_VALUE:
                        to_rem.append((i, j, k))
                        modified += 1
                    hull[i, j, k] = val
                for r in to_rem:
                    self._blocks_set.pop(r)
        
        # add blocks to structure
        for k, block in self._blocks_set.items():
            structure.add_block(block=block,
                                grid_position=block.position.as_tuple())