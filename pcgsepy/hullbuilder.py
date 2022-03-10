from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import grey_erosion, binary_erosion
import numpy as np
from pcgsepy.common.vecs import Orientation, Vec
from pcgsepy.structure import Block, Structure
from typing import Tuple

class HullBuilder:
    def __init__(self,
                 erosion_type: str,
                 apply_erosion: bool,
                 apply_smoothing: bool):
        self.BASE_BLOCK_VALUE = 0.1
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
        # TODO: Test and expand this map
        self.smoothing_blocks = {
            #fu
            4259840: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fd
            4195328: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fr
            4198400: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fl
            4210688: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #ul
            81920: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #ur
            69632: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #dl
            17408: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #dr
            5120: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #bu
            65552: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #bd
            1040: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #br
            4112: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #bl
            16400: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #ful
            4276224: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fur
            4263936: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fdl
            4211712: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fdr
            4199424: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #bul
            81936: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #bur
            69648: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #bdl
            17424: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #bdr
            5136: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fbrd
            4199440: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerInv', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fblu
            4276240: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerInv', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fbru
            4263952: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerInv', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fbld
            4211728: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerInv', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fblrd
            4215824: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #fblru
            4280336: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #dlrb
            21520: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #dlrf
            4215808: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #ulrb
            86032: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #ulrf
            4280320: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #ludb
            82960: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #ludf
            4277248: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #rudb
            70672: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP},
            #rudf
            4264960: {'block': 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope', 'of': Orientation.FORWARD, 'ou': Orientation.UP}
        }
    
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
        position = Vec.v3i(x=int(i * structure.grid_size),
                           y=int(j * structure.grid_size),
                           z=int(k * structure.grid_size))
        block.position = position
        structure.add_block(block=block,
                            grid_position=Vec.as_tuple(position))    

    def _get_neighbourhood(self,
                           idx: Tuple[int, int, int],
                           arr: np.ndarray) -> np.ndarray:
        """Get the 3x3x3 neighbourhood around idx in the array.

        Args:
            idx (Tuple[int, int, int]): The index (center of neighbourhood).
            arr (np.ndarray): The array.

        Returns:
            np.ndarray: The neighbourhood around idx.
        """
        neighbourhood = np.zeros(shape=(3, 3, 3), dtype=int)
        i, j, k = idx
        for di, dj, dk in zip([+1, 0, 0, 0, 0, -1], [0, +1, 0, 0, -1, 0], [0, 0, +1, -1, 0, 0]):
            ni, nj, nk = 1 + di, 1 + dj, 1 + dk
            if 0 < i + di < arr.shape[0] and 0 < j + dj < arr.shape[1] and 0 < k + dk < arr.shape[2]:
                v = 1 if arr[i, j, k] == self.BASE_BLOCK_VALUE else 0 
            else:
                v = 0
            neighbourhood[ni, nj, nk] = v
        return neighbourhood
    
    def _get_bit_neighbourhood(self,
                               neighbourhood: np.ndarray) -> int:
        """Get the bit (numerical) representation of the 3x3x3 neighbourhood.

        Args:
            neighbourhood (np.ndarray): The neighbourhood.

        Returns:
            int: The numerical representation.
        """
        v = 0
        for i in range(neighbourhood.shape[0]):
            for j in range(neighbourhood.shape[1]):
                for k in range(neighbourhood.shape[2]):
                    v = v << 1
                    v |= neighbourhood[i, j, k]
        return v

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
        hull[np.nonzero(air)] = 0
        hull[np.nonzero(arr)] = 0
        
        
        if self.apply_erosion:
            if self.erosion_type == 'grey':
                hull = grey_erosion(input=hull,
                                    footprint=self.footprint,
                                    mode='constant',
                                    cval=1)
                hull = hull.astype(float)
                hull *= self.BASE_BLOCK_VALUE                
            elif self.erosion_type == 'bin':
                mask = np.zeros(arr.shape)
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        for k in range(mask.shape[2]):
                            mask[i, j, k] = 0 if self._adj_to_spaceship(i=i,
                                                                        j=j,
                                                                        k=k,
                                                                        spaceship=arr) else 1
                hull = binary_erosion(input=hull,
                                      mask=mask,
                                      iterations=self.iterations)
                hull = hull.astype(float)
                hull *= self.BASE_BLOCK_VALUE
        
        if self.apply_smoothing:
            smoothed_idxs = []
            # For each block in the hull
            for i in range(hull.shape[0]):
                for j in range(hull.shape[1]):
                    for k in range(hull.shape[2]):
                        # if it has neighbouring block in direction
                        if hull[i, j, k] == self.BASE_BLOCK_VALUE:
                            neighbourhood = self._get_neighbourhood(idx=(i, j, k),
                                                                    arr=hull)
                            bit_n = self._get_bit_neighbourhood(neighbourhood=neighbourhood)
                            # add the correct block to the structure
                            if bit_n in self.smoothing_blocks.keys():
                                smoothed_idxs.append((i, j, k))
                                self._add_block(block_type=self.smoothing_blocks[bit_n]['block'],
                                                structure=structure,
                                                pos=(i, j, k),
                                                orientation_forward=self.smoothing_blocks[bit_n]['of'],
                                                orientation_up=self.smoothing_blocks[bit_n]['ou'])
            # and set to 0 the value in the hull
            for (i, j, k) in smoothed_idxs:
                hull[i, j, k] = 0
            
        
        for i in range(hull.shape[0]):
            for j in range(hull.shape[1]):
                for k in range(hull.shape[2]):
                    if hull[i, j, k] != 0:
                        self._add_block(block_type=self.base_block,
                                        structure=structure,
                                        pos=(i, j, k))