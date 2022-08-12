from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import grey_erosion, binary_erosion, binary_dilation
import numpy as np
import numpy.typing as npt
from pcgsepy.common.vecs import Orientation, Vec, orientation_from_vec
from pcgsepy.structure import Block, Structure, MountPoint
from typing import List, Optional, Tuple
from itertools import product
from pcgsepy.common.vecs import rotate, get_rotation_matrix
from scipy.spatial.transform import Rotation

# NOTE: This class currently cannot smooth the hull as intended due to probably incorrect calculations of mountpoints.
# Will be updated--and improved--once that's fixed.

class HullBuilder:
    def __init__(self,
                 erosion_type: str,
                 apply_erosion: bool,
                 apply_smoothing: bool):
        # TODO: change this to Enum(int)s
        self.AIR_BLOCK_VALUE = 0
        self.BASE_BLOCK_VALUE = 1
        self.SLOPE_BLOCK_VALUE = 2
        self.CORNER_BLOCK_VALUE = 3
        self.CORNERINV_BLOCK_VALUE = 4
        self.CORNERSQUARE_BLOCK_VALUE = 5
        self.CORNERSQUAREINV_BLOCK_VALUE = 6
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
        self.obstruction_targets = ['window']
        self._blocks_set = {}
        
        self._orientations = [Orientation.FORWARD, Orientation.BACKWARD, Orientation.UP, Orientation.DOWN, Orientation.LEFT, Orientation.RIGHT]
        self._valid_orientations = [(of, ou) for (of, ou) in list(product(self._orientations, self._orientations)) if of != ou and of != orientation_from_vec(ou.value.opposite())]
        self._smoothing_order = {
            self.BASE_BLOCK_VALUE: [self.SLOPE_BLOCK_VALUE, self.CORNERSQUARE_BLOCK_VALUE, self.CORNER_BLOCK_VALUE],
            self.CORNERSQUAREINV_BLOCK_VALUE: [],
            self.CORNERINV_BLOCK_VALUE: [],
            self.SLOPE_BLOCK_VALUE: [self.CORNERSQUARE_BLOCK_VALUE, self.CORNER_BLOCK_VALUE]
            }
        self.block_value_types = {
            self.BASE_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorBlock',
            self.SLOPE_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope',
            self.CORNER_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner',
            self.CORNERINV_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerInv',
            self.CORNERSQUARE_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerSquare',
            self.CORNERSQUAREINV_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerSquareInverted',
        }
    
    def get_structure_iterators(self,
                                structure: Structure) -> Tuple[Vec, List[Vec]]:
        scale = structure.grid_size
        return scale, self._orientations
    
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
        return out_arr
        
    def _add_block(self,
                   block_type: str,
                   idx: Tuple[int, int, int],
                   pos: Vec,
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
        block = Block(block_type=block_type,
                      orientation_forward=orientation_forward,
                      orientation_up=orientation_up)
        block.position = pos
        self._blocks_set[idx] = block

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
    
    def _within_hull(self,
                     loc: Vec,
                     hull: npt.NDArray[np.float32]) -> bool:
        i, j, k = loc.as_tuple()
        return 0 <= i < hull.shape[0] and 0 <= j < hull.shape[1] and 0 <= k < hull.shape[2]
    
    def _is_air_block(self,
                      loc: Vec,
                      structure: Structure,
                      hull: np.typing.NDArray) -> bool:
        return not self._exists_block(idx=loc.as_tuple(), structure=structure) and\
            (self._within_hull(loc=loc.scale(1 / structure.grid_size).to_veci(), hull=hull) and hull[loc.scale(1 / structure.grid_size).to_veci().as_tuple()] == self.AIR_BLOCK_VALUE)
    
    def _next_to_target(self,
                        loc: Vec,
                        structure: Structure,
                        direction: Vec) -> bool:
        dloc = loc.sum(direction)
        if self._exists_block(idx=dloc.as_tuple(), structure=structure):
            obs_block = structure._blocks.get(dloc.as_tuple())
            return any([target.lower() in obs_block.block_type.lower() for target in self.obstruction_targets])
    
    def _remove_in_direction(self,
                             loc: Vec,
                             hull: npt.NDArray[np.float32],
                             direction: Vec) -> npt.NDArray[np.float32]:
        i, j, k = loc.as_tuple()
        di, dj, dk = direction.as_tuple()
        while 0 < i < hull.shape[0] and 0 < j < hull.shape[1] and 0 < k < hull.shape[2]:
            hull[i, j, k] = self.AIR_BLOCK_VALUE
            i += di
            j += dj
            k += dk
        return hull
    
    def _remove_obstructing_blocks(self,
                                   hull: npt.NDArray[np.float32],
                                   structure: Structure) -> npt.NDArray[np.float32]:
        ii, jj, kk = hull.shape
        for i in range(ii):
            for j in range(jj):
                for k in range(kk):
                    if hull[i, j, k] != self.AIR_BLOCK_VALUE:
                        scale = structure.grid_size
                        loc = Vec.from_tuple((scale * i, scale * j, scale * k))
                        for direction in self._orientations:
                            if self._next_to_target(loc=loc,
                                                    structure=structure,
                                                    direction=direction.value.scale(scale)):
                                hull = self._remove_in_direction(loc=loc.scale(v=1 / structure.grid_size).to_veci(),
                                                                 hull=hull,
                                                                 direction=direction.value.opposite())
        return hull
    
    def _get_outer_indices(self,
                           arr: npt.NDArray[np.float32],
                           edges_only: bool = False,
                           corners_only: bool = False) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        # TODO: find a way to make this work without the moving slice hack
        # i.e.: get the below commented code to do the same thing
        # __threshold = 1
        # mag = generic_gradient_magnitude(arr, sobel, mode='nearest', cval=self.AIR_BLOCK_VALUE)
        # mag[arr == self.AIR_BLOCK_VALUE] = self.AIR_BLOCK_VALUE
        # return np.where(mag >= (np.max(mag) * __threshold))
        idxs_i, idxs_j, idxs_k = [], [], []
        for direction in self._orientations:
            di, dj, dk = direction.value.x, direction.value.y, direction.value.z
            
            if di != 0:
                face = np.ones(shape=(arr.shape[1], arr.shape[2])) * self.AIR_BLOCK_VALUE
                i = 0 if di >= 0 else -1
                while -arr.shape[0] < i < arr.shape[0]:
                    for j in range(face.shape[0]):
                        for k in range(face.shape[1]):
                            if face[j, k] == self.AIR_BLOCK_VALUE and arr[i, j, k] != self.AIR_BLOCK_VALUE:
                                face[j, k] = 1 if self.AIR_BLOCK_VALUE != 1 else -self.AIR_BLOCK_VALUE
                                idxs_i.append((i, j, k) if i >= 0 else (arr.shape[0] + i, j, k))
                    i += di
                
            elif dj != 0:
                face = np.ones(shape=(arr.shape[0], arr.shape[2])) * self.AIR_BLOCK_VALUE
                
                j = 0 if dj >= 0 else -1
                while -arr.shape[1] < j < arr.shape[1]:
                    for i in range(face.shape[0]):
                        for k in range(face.shape[1]):
                            if face[i, k] == self.AIR_BLOCK_VALUE and arr[i, j, k] != self.AIR_BLOCK_VALUE:
                                face[i, k] = 1 if self.AIR_BLOCK_VALUE != 1 else -self.AIR_BLOCK_VALUE
                                idxs_j.append((i, j, k) if j >= 0 else (i, arr.shape[1] + j, k))
                    j += dj
                                
            else:
                face = np.ones(shape=(arr.shape[0], arr.shape[1])) * self.AIR_BLOCK_VALUE
                
                k = 0 if dk >= 0 else -1
                while -arr.shape[2] < k < arr.shape[2]:
                    for i in range(face.shape[0]):
                        for j in range(face.shape[1]):
                            if face[i, j] == self.AIR_BLOCK_VALUE and arr[i, j, k] != self.AIR_BLOCK_VALUE:
                                face[i, j] = 1 if self.AIR_BLOCK_VALUE != 1 else -self.AIR_BLOCK_VALUE
                                idxs_k.append((i, j, k) if k >= 0 else (i, j, arr.shape[2] + k))
                    k += dk
        if corners_only:
            indices = list(set(idxs_i).intersection(set(idxs_j)).intersection(set(idxs_k)))
        elif edges_only:
            indices =  list(set(idxs_i).intersection(set(idxs_j))) + list(set(idxs_j).intersection(set(idxs_k))) + list(set(idxs_j).intersection(set(idxs_k)))
        else:
            indices =  list(set(idxs_i + idxs_j + idxs_k))
        indices = list(set(indices))
        return np.asarray([idx[0] for idx in indices]), np.asarray([idx[1] for idx in indices]), np.asarray([idx[2] for idx in indices])
    
    def try_and_get_block(self,
                          idx: Tuple[int, int, int],
                          offset: Vec,
                          structure: Structure) -> Optional[Block]:
        """Try and get a block from either the hull or the structure.

        Args:
            idx (Tuple[int, int, int]): The index of the block (hull-scaled).
            offset (Vec): The offset vector (structure-scaled).
            structure (Structure): The structure.

        Returns:
            Optional[Block]: The block at `idx + offset`, if it exists.
        """
        dloc = Vec.from_tuple(idx).scale(structure.grid_size).sum(offset)
        if self._exists_block(idx=dloc.as_tuple(),
                              structure=structure):
            return structure._blocks[dloc.as_tuple()]
        else:
            return self._blocks_set.get(Vec.from_tuple(idx).sum(offset.scale(1 / structure.grid_size)).to_veci().as_tuple(), None)
    
    def _get_mountpoint_limits(self,
                               mountpoints: List[MountPoint],
                               block: Block,
                               rot_direction: Vec,
                               realign: bool = False) -> Tuple[List[Vec], List[Vec], List[int]]:
        starts, ends = [], []
        for mp in mountpoints:
            realigned_start = Vec(x=mp.start.x if rot_direction.x == -1 or rot_direction.x == 0 else mp.start.x - block.scaled_size.x,
                                  y=mp.start.y if rot_direction.y == -1 or rot_direction.y == 0 else mp.start.y - block.scaled_size.y,
                                  z=mp.start.z if rot_direction.z == -1 or rot_direction.z == 0 else mp.start.z - block.scaled_size.z)
            realigned_end = Vec(x=mp.end.x if rot_direction.x == -1 or rot_direction.x == 0 else mp.end.x - block.scaled_size.x,
                                y=mp.end.y if rot_direction.y == -1 or rot_direction.y == 0 else mp.end.y - block.scaled_size.y,
                                z=mp.end.z if rot_direction.z == -1 or rot_direction.z == 0 else mp.end.z - block.scaled_size.z)

            if realign:
                inv_rot_matrix = Rotation.from_matrix(get_rotation_matrix(forward=block.orientation_forward,
                                                                        up=block.orientation_up)).inv().as_matrix()
                
                realigned_start = rotate(rotation_matrix=inv_rot_matrix,
                                        vector=realigned_start)
                realigned_end = rotate(rotation_matrix=inv_rot_matrix,
                                    vector=realigned_end)
                
            starts.append(realigned_start)
            ends.append(realigned_end)
            
        ordered_ends = [Vec(x=max(start.x, end.x), y=max(start.y, end.y), z=max(start.z, end.z)) for (start, end) in zip(starts, ends)]
        ordered_starts = [Vec(x=min(start.x, end.x), y=min(start.y, end.y), z=min(start.z, end.z)) for (start, end) in zip(starts, ends)]
        
        planes = [end.diff(start).vol(ignore_zero=False) for start, end in zip(ordered_starts, ordered_ends)]
        
        return ordered_starts, ordered_ends, planes
    
    def _check_valid_placement(self,
                               idx: Tuple[int, int, int],
                               block: Block,
                               direction: Orientation,
                               hull: np.typing.NDArray,
                               structure: Structure) -> Tuple[bool, int]:
        """Check if the block could be placed with the given orientation when checking in the specified direction.

        Args:
            idx (Tuple[int, int, int]): The index of the block (hull-scaled).
            block (Block): The block to be checked.
            direction (Orientation): The direction to check placement for.
            hull (np.typing.NDArray): The hull.
            structure (Structure): The structure.

        Returns:
            bool: Whether the block could be placed with the given orientation when checking in the specified direction.
        """
        rot_direction = rotate(get_rotation_matrix(forward=block.orientation_forward,
                                                   up=block.orientation_up),
                               vector=direction.value).to_veci()
        mp1 = [mp for mp in block.mountpoints if mp.face == rot_direction]
        starts1, ends1, planes1 = self._get_mountpoint_limits(mountpoints=mp1,
                                                              block=block,
                                                              rot_direction=rot_direction,
                                                              realign=True)
        other_block = self.try_and_get_block(idx=idx,
                                             offset=direction.value.scale(structure.grid_size).to_veci(),
                                             structure=structure)
        if other_block is None:
            # facing air block, can always be placed
            if mp1 != []:
                return True, sum(planes1)
            else:
                face = Vec(x=block.scaled_size.x if rot_direction.x == 0 else 0,
                           y=block.scaled_size.y if rot_direction.y == 0 else 0,
                           z=block.scaled_size.z if rot_direction.z == 0 else 0)
                return True, face.vol(ignore_zero=False)
        else:
            if mp1 == []:
                return False, 0
        
        opposite_direction = orientation_from_vec(direction.value.opposite())
        rot_other = rotate(get_rotation_matrix(forward=other_block.orientation_forward,
                                               up=other_block.orientation_up),
                           vector=opposite_direction.value).to_veci()
                
        mp2 = [mp for mp in other_block.mountpoints if mp.face == rot_other]
        if mp2 == []:
            return False, 0
        starts2, ends2, planes2 = self._get_mountpoint_limits(mountpoints=mp2,
                                                              block=other_block,
                                                              rot_direction=rot_other,
                                                              realign=True)
        
        # assume block will always be "smaller" than other_block
        # then all mp1 should be contained in some mp2
        all_valid = []
        coverage_err = 0
        for eo1, so1, p1 in zip(ends1, starts1, planes1):
            assert p1 != 0, f'Mountpoint with empty surface: {mp1} has {so1}-{eo1} (from block {block})'
            mp_valid = True
            for eo2, so2, p2 in zip(ends2, starts2, planes2):
                assert p2 != 0, f'Mountpoint with empty surface: {mp2} has {so2}-{eo2} (from block {other_block})'
                # NOTE: This check does not take into account exclusions and properties masks (yet)
                # Sources\VRage.Math\BoundingBoxI.cs#328
                # (double)this.Max.X >= (double)box.Min.X && (double)this.Min.X <= (double)box.Max.X && ((double)this.Max.Y >= (double)box.Min.Y && (double)this.Min.Y <= (double)box.Max.Y) && ((double)this.Max.Z >= (double)box.Min.Z && (double)this.Min.Z <= (double)box.Max.Z);
                mp_valid &= so1.x >= so2.x and eo1.x <= eo2.x and so1.y >= so2.y and eo1.y <= eo2.y and so1.z >= so2.z and eo1.z <= eo2.z
                # compute coverage error for the mountpoint
                coverage_err += abs(p2 - p1)
            all_valid.append(mp_valid)
        return all(all_valid), coverage_err
        
    def _check_valid_position(self,
                              idx: Tuple[int, int, int],
                              block: Block,
                              hull: np.typing.NDArray,
                              structure: Structure) -> Tuple[bool, int]:
        valid = True
        area_err = 0
        for direction in self._orientations:
            res, delta_area = self._check_valid_placement(idx=idx,
                                                          block=block,
                                                          direction=direction,
                                                          hull=hull,
                                                          structure=structure)
            valid &= res
            area_err += delta_area
            if not valid:
                break
            
        return valid, delta_area
    
    def try_smoothing(self,
                      idx: Tuple[int, int, int],
                      hull: np.typing.NDArray,
                      structure: Structure) -> Optional[Block]:
        i, j, k = idx
        block_type = hull[i, j, k]
        block = self._blocks_set[idx]
        
        # removal check
        valid, _ = self._check_valid_position(idx=idx,
                                              block=block,
                                              hull=hull,
                                              structure=structure)
        if not valid:
            return None, self.AIR_BLOCK_VALUE
        # replacement check   
        elif block_type in self._smoothing_order.keys():
            for possible_type in self._smoothing_order[block_type]:
                orientation_scores, valids = np.zeros(shape=len(self._valid_orientations), dtype=np.int64), np.zeros(shape=len(self._valid_orientations), dtype=np.bool8)
                for i, (of, ou) in enumerate(self._valid_orientations):
                    possible_block = Block(block_type=self.block_value_types[possible_type],
                                           orientation_forward=of,
                                           orientation_up=ou)
                    valid, err = self._check_valid_position(idx=idx,
                                                  block=possible_block,
                                                  hull=hull,
                                                  structure=structure)
                    orientation_scores[i] = err if valid else 9999
                    valids[i] = valid
                
                if idx == (1, 18, 3):
                    print(orientation_scores, valids)
                
                if any(valids):
                    of, ou = self._valid_orientations[np.argmin(orientation_scores)]
                    return Block(block_type=self.block_value_types[possible_type],
                                 orientation_forward=of,
                                 orientation_up=ou), possible_type
            return None, block_type
        # skip
        else:
            return None, block_type
        
    def _get_ordered_idxs(self,
                          arr: npt.NDArray[np.float32]) -> List[Tuple[int, int, int]]:
        xxs, yys, zzs = [], [], []
        edges = self._get_outer_indices(arr=arr,
                                        edges_only=True)
        arr_copy = np.zeros_like(arr)
        arr_copy[edges] = 1
        while np.sum(arr_copy.flatten()) != 0:
            idxs = self._get_outer_indices(arr=arr_copy,
                                        corners_only=True)
            xxs.extend(idxs[0].tolist())
            yys.extend(idxs[1].tolist())
            zzs.extend(idxs[2].tolist())
            arr_copy[idxs] = 0
            
        return [(x, y, z) for x, y, z in zip(xxs, yys, zzs)]
           
    def add_external_hull(self,
                          structure: Structure) -> None:
        """Add an external hull to the given Structure.
        This process adds the hull blocks directly into the Structure, so it can be used only once per spaceship.

        Args:
            structure (Structure): The spaceship.
        """
        self._blocks_set = {}
        
        arr = structure.as_grid_array
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
                mask = np.ones(arr.shape, dtype=np.uint8) * self.BASE_BLOCK_VALUE
                adj_to_spaceship = np.zeros(arr.shape, dtype=np.uint8)
                adj_to_spaceship[np.nonzero(arr)] = 1
                adj_to_spaceship = binary_dilation(adj_to_spaceship)
                mask[np.nonzero(adj_to_spaceship)] = self.AIR_BLOCK_VALUE
                hull = binary_erosion(input=hull,
                                      mask=mask,
                                      iterations=self.iterations)
                hull = hull.astype(int)
                hull *= self.BASE_BLOCK_VALUE
        
        # remove all blocks that obstruct target block type
        hull = self._remove_obstructing_blocks(hull=hull,
                                               structure=structure)
        
        # add blocks to self._blocks_set
        for i in range(hull.shape[0]):
                for j in range(hull.shape[1]):
                    for k in range(hull.shape[2]):
                        if hull[i, j, k] != self.AIR_BLOCK_VALUE:
                            self._add_block(block_type=self.base_block,
                                            idx=(i, j, k),
                                            pos=Vec.v3i(i, j, k).scale(v=structure.grid_size),
                                            orientation_forward=Orientation.FORWARD,
                                            orientation_up=Orientation.UP)
        
        # initial blocks removal check
        # for i in range(hull.shape[0]):
        #         for j in range(hull.shape[1]):
        #             for k in range(hull.shape[2]):
        #                 if hull[i, j, k] != self.AIR_BLOCK_VALUE:
        #                     block = self._blocks_set[(i, j, k)]
        #                     valid, _ = self._check_valid_position(idx=(i, j, k),
        #                                                             block=block,
        #                                                             hull=hull,
        #                                                             structure=structure)
        #                     # TODO: Some blocks are not removed even though their symmetrical counterpart are.
        #                     # It seems to be a problem with the orientations of the blocks in the spaceship.
        #                     # Example: (9, 8, 1) and (9, 8, 7), both have a LargeBlockArmorCornerInv on the RIGHT
        #                     # but only one of them is removed correctly (due to mountpoints).
        #                     if not valid:
        #                         hull[i, j, k] = self.AIR_BLOCK_VALUE
        #                         self._blocks_set.pop((i, j, k))
        
        if self.apply_smoothing:
            modified = 1
            while modified != 0:
                modified = 0
                to_rem = []
                for (i, j, k) in self._get_ordered_idxs(arr=hull):
                    block = self._blocks_set[(i, j, k)]
                    substitute_block, val = self.try_smoothing(idx=(i, j, k),
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
                break
        
        # add blocks to structure
        for block in self._blocks_set.values():
            structure.add_block(block=block,
                                grid_position=block.position.as_tuple())
        
        structure.sanify()