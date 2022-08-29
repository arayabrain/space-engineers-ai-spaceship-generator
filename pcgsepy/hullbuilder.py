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
from enum import IntEnum


class Block(IntEnum):
    AIR_BLOCK_VALUE: 0
    BASE_BLOCK_VALUE: 1
    SLOPE_BLOCK_VALUE: 2
    CORNER_BLOCK_VALUE: 3
    CORNERINV_BLOCK_VALUE: 4
    CORNERSQUARE_BLOCK_VALUE: 5
    CORNERSQUAREINV_BLOCK_VALUE: 6


block_value_types = {
    Block.BASE_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorBlock',
    Block.SLOPE_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope',
    Block.CORNER_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner',
    Block.CORNERINV_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerInv',
    Block.CORNERSQUARE_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerSquare',
    Block.CORNERSQUAREINV_BLOCK_VALUE: 'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerSquareInverted',
}


class HullBuilder:
    def __init__(self,
                 erosion_type: str,
                 apply_erosion: bool,
                 apply_smoothing: bool):
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
        # self._smoothing_order = {
        #     Block.BASE_BLOCK_VALUE: [Block.SLOPE_BLOCK_VALUE, Block.CORNERSQUARE_BLOCK_VALUE, Block.CORNER_BLOCK_VALUE],
        #     Block.CORNERSQUAREINV_BLOCK_VALUE: [],
        #     Block.CORNERINV_BLOCK_VALUE: [],
        #     Block.SLOPE_BLOCK_VALUE: [Block.CORNERSQUARE_BLOCK_VALUE, Block.CORNER_BLOCK_VALUE]
        #     }
        self._smoothing_order = {
            Block.BASE_BLOCK_VALUE: [Block.SLOPE_BLOCK_VALUE],
            Block.SLOPE_BLOCK_VALUE: [Block.CORNER_BLOCK_VALUE]
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
        out_arr[out_idx] = block.BASE_BLOCK_VALUE
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
                                 arr: np.ndarray) -> npt.NDArray[np.uint8]:
        """Create a mask for air blocks within the structure.

        Args:
            arr (np.ndarray): The array representation of the structure.

        Returns:
            npt.NDArray[np.uint8]: The mask array.
        """
        air_blocks = np.zeros(shape=arr.shape, dtype=np.uint8)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    if sum(arr[0:i, j, k]) != 0 and \
                        sum(arr[i:arr.shape[0], j, k]) != 0 and \
                        sum(arr[i, 0:j, k]) != 0 and \
                        sum(arr[i, j:arr.shape[1], k]) != 0 and \
                        sum(arr[i, j, 0:k]) != 0 and \
                        sum(arr[i, j, k:arr.shape[2]]) != 0:
                            air_blocks[i, j, k] = block.BASE_BLOCK_VALUE
        return air_blocks
        
    def _exists_block(self,
                      idx: Tuple[int, int, int],
                      structure: Structure) -> bool:
        """Check if a block exists within the structure.

        Args:
            idx (Tuple[int, int, int]): The index to check at.
            structure (Structure): The structure.

        Returns:
            bool: Whether the block exists.
        """
        return structure._blocks.get(idx, None) is not None
    
    def _within_hull(self,
                     loc: Vec,
                     hull: npt.NDArray[np.float32]) -> bool:
        """Check if a block exists within the hull.

        Args:
            loc (Vec): The index to check at.
            hull (npt.NDArray[np.float32]): The hull array.

        Returns:
            bool: Whether the block is within the hull.
        """
        i, j, k = loc.as_tuple()
        return 0 <= i < hull.shape[0] and 0 <= j < hull.shape[1] and 0 <= k < hull.shape[2]
    
    def _is_air_block(self,
                      loc: Vec,
                      structure: Structure,
                      hull: np.typing.NDArray[np.float32]) -> bool:
        """Check if the block is an air block.

        Args:
            loc (Vec): The index to check at.
            structure (Structure): The structure.
            hull (np.typing.NDArray[np.float32]): The hull array.

        Returns:
            bool: Whether the block is an air block.
        """
        return not self._exists_block(idx=loc.as_tuple(), structure=structure) and\
            (self._within_hull(loc=loc.scale(1 / structure.grid_size).to_veci(), hull=hull) and hull[loc.scale(1 / structure.grid_size).to_veci().as_tuple()] == block.AIR_BLOCK_VALUE)
    
    def _next_to_target(self,
                        loc: Vec,
                        structure: Structure,
                        direction: Vec) -> bool:
        """Check if the block is next to a target block along a direction.

        Args:
            loc (Vec): The index to check at.
            structure (Structure): The structure.
            direction (Vec): The direction to check at.

        Returns:
            bool: Whether the block is next to a target block.
        """
        dloc = loc.sum(direction)
        if self._exists_block(idx=dloc.as_tuple(), structure=structure):
            obs_block = structure._blocks.get(dloc.as_tuple())
            return any([target.lower() in obs_block.block_type.lower() for target in self.obstruction_targets])
    
    def _remove_in_direction(self,
                             loc: Vec,
                             hull: npt.NDArray[np.float32],
                             direction: Vec) -> npt.NDArray[np.float32]:
        """Remove all blocks in the hull along a direction.

        Args:
            loc (Vec): The starting index.
            hull (npt.NDArray[np.float32]): The hull array.
            direction (Vec): The direction to remove along to.

        Returns:
            npt.NDArray[np.float32]: The modified hull.
        """
        i, j, k = loc.as_tuple()
        di, dj, dk = direction.as_tuple()
        while 0 < i < hull.shape[0] and 0 < j < hull.shape[1] and 0 < k < hull.shape[2]:
            i += di
            j += dj
            k += dk
            if (i, j, k) in self._blocks_set.keys():
                hull[i, j, k] = block.AIR_BLOCK_VALUE
                self._blocks_set.pop((i, j, k))
        return hull
    
    def _remove_obstructing_blocks(self,
                                   hull: npt.NDArray[np.float32],
                                   structure: Structure) -> npt.NDArray[np.float32]:
        """Remove the blocks obstructing a target block.

        Args:
            hull (npt.NDArray[np.float32]): The hull array.
            structure (Structure): The structure.

        Returns:
            npt.NDArray[np.float32]: The modified hull array.
        """
        ii, jj, kk = hull.shape
        scale = structure.grid_size
        for i in range(ii):
            for j in range(jj):
                for k in range(kk):
                    if hull[i, j, k] != block.AIR_BLOCK_VALUE:
                        loc = Vec.from_tuple((scale * i, scale * j, scale * k))
                        for direction in self._orientations:
                            ntt = self._next_to_target(loc=loc,
                                                       structure=structure,
                                                       direction=direction.value.scale(scale))
                            if ntt:
                                hull[i, j, k] = block.AIR_BLOCK_VALUE
                                self._blocks_set.pop((i, j, k))
                                hull = self._remove_in_direction(loc=loc.scale(v=1 / structure.grid_size).to_veci(),
                                                                 hull=hull,
                                                                 direction=direction.value.opposite())
        return hull
    
    def _get_outer_indices(self,
                           arr: npt.NDArray[np.float32],
                           edges_only: bool = False,
                           corners_only: bool = False) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Get the indices of blocks lying on the outer corner/edges/faces of the array.

        Args:
            arr (npt.NDArray[np.float32]): The array.
            edges_only (bool, optional): Get blocks on the edges only. Defaults to False.
            corners_only (bool, optional): Get blocks on the corners only. Defaults to False.

        Returns:
            Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]: The indices of the outer blocks.
        """
        n_neighbours = np.zeros_like(arr)
        for idx, _ in np.ndenumerate(arr):
            if self._blocks_set.get(idx, None):
                for offset in self._orientations:
                    pos = Vec.from_tuple(idx).sum(offset.value).as_tuple()
                    if self._blocks_set.get(pos, None):
                        n_neighbours[idx] = n_neighbours[idx] + 1
        if corners_only:
            return np.nonzero(np.where(n_neighbours <= 2, n_neighbours, 0))
        elif edges_only:
            return np.nonzero(np.where(n_neighbours <= 3, n_neighbours, 0))
        else:
            return np.nonzero(n_neighbours)
    
    def _get_neighbourhood(self,
                           idx: Vec,
                           structure: Structure) -> List[Block]:
        """Get the neighbourhood of a block.

        Args:
            idx (Vec): The index of the block.
            structure (Structure): The structure.

        Returns:
            List[Block]: The neighbourhood of the block.
        """
        n = []
        for di, dj, dk in zip([1, 1, 1, 1, 1,  1,  1,   0, 0, 0, 0,  0,  0,  -1, -1, -1, -1, -1, -1, -1],
                              [0, 1, 0, 1, -1, 0,  -1,  1, 0, 1, -1, 0,  -1, 0,  1,  0,  1,  -1, 0,  -1],
                              [0, 0, 1, 1, 0,  -1, -1,  0, 1, 1, 0,  -1, -1, 0,  0,  1,  1,  0,  -1, -1]):
            offset = Vec(di, dj, dk).scale(structure.grid_size)
            if not offset.is_zero:
                adj = self.try_and_get_block(idx=idx,
                                             offset=offset,
                                             structure=structure)
                if adj:
                    n.append(adj)
        return n
    
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
    
    def _correct_and_centered_rotation(self,
                                       center: Vec,
                                       rotation_matrix: np.typing.NDArray,
                                       vector: Vec) -> Vec:
        """Rotate correctly and center the vector to the center of the block.

        Args:
            center (Vec): The center of the block.
            rotation_matrix (np.typing.NDArray): The rotation matrix of the block.
            vector (Vec): The vector to rotate and center.

        Returns:
            Vec: The rotated and centered vector.
        """
        # Rotation with additional checking as in Space Engineers source code.
        centered_vector = vector.diff(center)
        centered_vector_i = centered_vector.floor()
        rotated_vector = rotate(rotation_matrix=rotation_matrix,
                                vector=centered_vector)
        rotated_vector_i_correct = rotate(rotation_matrix=rotation_matrix,
                                          vector=centered_vector_i)
        rotated_vector_i = rotated_vector.floor()
        correction = rotated_vector_i_correct.diff(rotated_vector_i)
        return rotated_vector.sum(correction)
    
    def _get_mountpoint_limits(self,
                               mountpoints: List[MountPoint],
                               block_center: Vec,
                               rotation_matrix: npt.NDArray[np.float32]) -> Tuple[List[Vec], List[Vec], List[int]]:
        """Get the start and end vectors of the mountpoint, as well as the face area(s).

        Args:
            mountpoints (List[MountPoint]): The list of mountpoints.
            block_center (Vec): The center of the block.
            rotation_matrix (npt.NDArray[np.float32]): The rotation matrix of the block.

        Returns:
            Tuple[List[Vec], List[Vec], List[int]]: The start and end vector, and the face area(s).
        """
        starts, ends = [], []
        for mp in mountpoints:
            rotated_start = self._correct_and_centered_rotation(center=block_center,
                                                                rotation_matrix=rotation_matrix,
                                                                vector=mp.start)
            rotated_end = self._correct_and_centered_rotation(center=block_center,
                                                              rotation_matrix=rotation_matrix,
                                                              vector=mp.end)

            ordered_start = Vec.min(rotated_start, rotated_end)
            ordered_end = Vec.max(rotated_start, rotated_end)
            
            starts.append(ordered_start)
            ends.append(ordered_end)
        
        planes = [end.diff(start).bbox(ignore_zero=False) for start, end in zip(starts, ends)]
        
        return starts, ends, planes
    
    def intersect_planes(self,
                         starts1: List[Vec],
                         ends1: List[Vec],
                         starts2: List[Vec],
                         ends2: List[Vec]) -> float:
        """Compute the intersection surface between mountpoints of different blocks and return the inverse of the intersection as an error to minimize.

        Args:
            starts1 (List[Vec]): The start vectors of the first block.
            ends1 (List[Vec]): The end vectors of the first block.
            starts2 (List[Vec]): The start vectors of the second block.
            ends2 (List[Vec]): The end vectors of the first block.

        Returns:
            float: The intersection as an error.
        """
        intersect_bbox = 0.
        for start1, end1 in zip(starts1, ends1):
            for start2, end2 in zip(starts2, ends2):
                x1 = max(start1.x, start2.x)
                y1 = max(start1.y, start2.y)
                z1 = max(start1.z, start2.z)
                x2 = min(end1.x, end2.x)
                y2 = min(end1.y, end2.y)
                z2 = min(end1.z, end2.z)
                if x2 > x1 and y2 > y1 and z2 < z1:
                    intersect_bbox += (x2 - x1) * (y2 - y1)
                elif x2 > x1 and y2 <= y1 and z2 > z1:
                    intersect_bbox += (x2 - x1) * (z2 - z1)
                elif x2 <= x1 and y2 > y1 and z2 > z1:
                    intersect_bbox += (y2 - y1) * (z2 - z1)
                else:
                    intersect_bbox += 0
        return 1 / intersect_bbox
    
    def _check_valid_placement(self,
                               idx: Tuple[int, int, int],
                               block: Block,
                               direction: Orientation,
                               structure: Structure) -> Tuple[bool, int]:
        """Check if the block could be placed with the given orientation when checking in the specified direction.

        Args:
            idx (Tuple[int, int, int]): The index of the block (hull-scaled).
            block (Block): The block to be checked.
            direction (Orientation): The direction to check placement for.
            structure (Structure): The structure.

        Returns:
            bool: Whether the block could be placed with the given orientation when checking in the specified direction.
        """
        rot_mat = get_rotation_matrix(forward=block.orientation_forward,
                                      up=block.orientation_up)
        mp1 = [mp for mp in block.mountpoints if rotate(rot_mat, mp.face) == direction.value]
        starts1, ends1, planes1 = self._get_mountpoint_limits(mountpoints=mp1,
                                                              block_center=block.center,
                                                              block_size = block.scaled_size,
                                                              rotation_matrix=rot_mat)
        other_block = self.try_and_get_block(idx=idx,
                                             offset=direction.value.scale(structure.grid_size).to_veci(),
                                             structure=structure)
        opposite_direction = orientation_from_vec(direction.value.opposite())
        if other_block is None:
            if mp1 == []:
                mp2 = [mp for mp in block.mountpoints if rotate(rot_mat, mp.face) == opposite_direction.value]
                _, _, planes2 = self._get_mountpoint_limits(mountpoints=mp2,
                                                            block_center=block.center,
                                                            block_size = block.scaled_size,
                                                            rotation_matrix=rot_mat)
                val = sum(planes2)  # error as the surface of mountpoints on opposite face
            else:
                val = sum(planes1)  # erorr as surface of mountpoints of current face
            return True, val
        else:
            if mp1 == []:
                return False, 0
            else:
                rot_mat_other = get_rotation_matrix(forward=other_block.orientation_forward,
                                                    up=other_block.orientation_up)
                mp2 = [mp for mp in other_block.mountpoints if rotate(rot_mat_other, mp.face) == opposite_direction.value]
                if mp2 == []:
                    return False, 0
                starts2, ends2, planes2 = self._get_mountpoint_limits(mountpoints=mp2,
                                                                      block_center=other_block.center,
                                                                      block_size = block.scaled_size,
                                                                      rotation_matrix=rot_mat_other)
                all_valid = []
                for eo1, so1, p1 in zip(ends1, starts1, planes1):
                    assert p1 != 0, f'Mountpoint with empty surface: {mp1} has {so1}-{eo1} (from block {block})'
                    mp_valid = True
                    for eo2, so2, p2 in zip(ends2, starts2, planes2):
                        assert p2 != 0, f'Mountpoint with empty surface: {mp2} has {so2}-{eo2} (from block {other_block})'
                        # NOTE: This check does not take into account exclusions and properties masks (yet)
                        # Sources\VRage.Math\BoundingBoxI.cs#328
                        # (double)this.Max.X >= (double)box.Min.X && (double)this.Min.X <= (double)box.Max.X && ((double)this.Max.Y >= (double)box.Min.Y && (double)this.Min.Y <= (double)box.Max.Y) && ((double)this.Max.Z >= (double)box.Min.Z && (double)this.Min.Z <= (double)box.Max.Z);
                        mp_valid |= eo1.x >= so2.x and so1.x <= eo2.x and eo1.y >= so2.y and so1.y <= eo2.y and eo1.z >= so2.z and so1.z <= eo2.z
                    all_valid.append(mp_valid)
                return all(all_valid), self.intersect_planes(starts1, ends1, starts2, ends2)
        
    def _check_valid_position(self,
                              idx: Tuple[int, int, int],
                              block: Block,
                              hull: np.typing.NDArray,
                              structure: Structure) -> Tuple[bool, int]:
        """Check if the current position is valid for the given block.

        Args:
            idx (Tuple[int, int, int]): The index of the block.
            block (Block): The block to check.
            hull (npt.NDArray[np.float32]): The hull array.
            structure (Structure): The structure.

        Returns:
            Tuple[bool, int]: Whether the position is valid, and the area error.
        """
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
                      hull: npt.NDArray[np.float32],
                      structure: Structure) -> Optional[Block]:
        """Try applying a smoothing pass at the given index.

        Args:
            idx (Tuple[int, int, int]): The index.
            hull (npt.NDArray[np.float32]): The hull array.
            structure (Structure): The structure.

        Returns:
            Optional[Block]: A possible block to replace the current one.
        """
        i, j, k = idx
        block_type = hull[i, j, k]
        block = self._blocks_set[idx]
        # removal check
        valid, curr_err = self._check_valid_position(idx=idx,
                                                     block=block,
                                                     hull=hull,
                                                     structure=structure)
        if not valid:
            return None, block.AIR_BLOCK_VALUE
        # replacement check   
        elif block_type in self._smoothing_order.keys():
            # give priority to surrounding blocks orientations
            neighbourhood = self._get_neighbourhood(idx=idx,
                                                    structure=structure)
            priority_orientations = []
            for other_block in neighbourhood:                
                oo = (orientation_from_vec(other_block.orientation_forward),
                        orientation_from_vec(other_block.orientation_up))
                if oo not in priority_orientations:
                    priority_orientations.append(oo)
            for oo in self._valid_orientations:
                if oo not in priority_orientations:
                    priority_orientations.append(oo)
            for possible_type in self._smoothing_order[block_type]:
                orientation_scores, valids = np.zeros(shape=len(self._valid_orientations), dtype=np.int64), np.zeros(shape=len(self._valid_orientations), dtype=np.bool8)
                # try replacement
                for i, (of, ou) in enumerate(priority_orientations):
                    possible_block = Block(block_type=self.block_value_types[possible_type],
                                           orientation_forward=of,
                                           orientation_up=ou)
                    valid, err = self._check_valid_position(idx=idx,
                                                            block=possible_block,
                                                            hull=hull,
                                                            structure=structure)
                    orientation_scores[i] = err if valid else 9999  # make sure invalid scores are never picked
                    valids[i] = valid
                if any(valids) and min(orientation_scores) < curr_err:
                    of, ou = priority_orientations[np.argmin(orientation_scores)]
                    return Block(block_type=self.block_value_types[possible_type],
                                orientation_forward=of,
                                orientation_up=ou), possible_type
            return None, block_type
        # skip
        else:
            return None, block_type
        
    def add_external_hull(self,
                          structure: Structure) -> None:
        """Add an external hull to the given Structure.
        This process adds the hull blocks directly into the Structure, so it can be used only once per spaceship.

        Args:
            structure (Structure): The spaceship.
        """
        arr = structure.as_grid_array
        air = self._tag_internal_air_blocks(arr=arr)
        hull = self._get_convex_hull(arr=arr)
        hull[np.nonzero(air)] = block.AIR_BLOCK_VALUE
        hull[np.nonzero(arr)] = block.AIR_BLOCK_VALUE
        
        if self.apply_erosion:
            if self.erosion_type == 'grey':
                hull = grey_erosion(input=hull,
                                    footprint=self.footprint,
                                    mode='constant',
                                    cval=1)
                hull = hull.astype(int)
                hull *= block.BASE_BLOCK_VALUE                
            elif self.erosion_type == 'bin':
                mask = np.zeros(arr.shape)
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        for k in range(mask.shape[2]):
                            mask[i, j, k] = block.AIR_BLOCK_VALUE if self._adj_to_spaceship(i=i, j=j, k=k, spaceship=arr) else block.BASE_BLOCK_VALUE
                hull = binary_erosion(input=hull,
                                      mask=mask,
                                      iterations=self.iterations)
                hull = hull.astype(int)
                hull *= block.BASE_BLOCK_VALUE
                
        # add blocks to self._blocks_set
        for i in range(hull.shape[0]):
                for j in range(hull.shape[1]):
                    for k in range(hull.shape[2]):
                        if hull[i, j, k] != block.AIR_BLOCK_VALUE:
                            self._add_block(block_type=self.base_block,
                                            idx=(i, j, k),
                                            pos=Vec.v3i(i, j, k).scale(v=structure.grid_size),
                                            orientation_forward=Orientation.FORWARD,
                                            orientation_up=Orientation.UP)
        
        # remove all blocks that obstruct target block type
        hull = self._remove_obstructing_blocks(hull=hull,
                                               structure=structure)
        
        # initial blocks removal check
        for (i, j, k), v in np.ndenumerate(hull):
            if v != block.AIR_BLOCK_VALUE:
                block = self._blocks_set[(i, j, k)]
                valid, _ = self._check_valid_position(idx=(i, j, k),
                                                        block=block,
                                                        hull=hull,
                                                        structure=structure)
                if not valid:
                    hull[i, j, k] = block.AIR_BLOCK_VALUE
                    self._blocks_set.pop((i, j, k))
        
        if self.apply_smoothing:
            idxs = self._get_outer_indices(arr=hull, edges_only=True)
            ii, jj, kk = idxs
            curr_checking = [(i, j, k) for (i, j, k) in zip(ii, jj, kk)]
            while len(curr_checking) != 0:
                to_rem, to_inspect = [], []
                for (i, j, k) in curr_checking:
                    block = self._blocks_set[(i, j, k)]
                    substitute_block, val = self.try_smoothing(idx=(i, j, k),
                                                               hull=hull,
                                                               structure=structure)
                    if substitute_block is not None and substitute_block.block_type != block.block_type:
                        substitute_block.position = block.position
                        self._blocks_set[(i, j, k)] = substitute_block
                        to_inspect.extend(self.adj_in_hull(idx=(i, j, k), hull=hull))
                    elif substitute_block is None and val == block.AIR_BLOCK_VALUE:
                        to_rem.append((i, j, k))
                        to_inspect.extend(self.adj_in_hull(idx=(i, j, k), hull=hull))
                    hull[i, j, k] = val
                to_inspect = list(set(to_inspect))
                to_rem = list(set(to_rem))
                for r in to_rem:
                    self._blocks_set.pop(r)
                    if r in to_inspect:
                        to_inspect.remove(r)
                curr_checking = to_inspect

        # add blocks to structure
        for k, block in self._blocks_set.items():
            structure.add_block(block=block,
                                grid_position=block.position.as_tuple())
        
        structure.sanify()


def enforce_symmetry(structure: Structure,
                     axis: str = 'z',
                     upper: bool = True,
                     pivot_blocktype: str = 'MyObjectBuilder_Cockpit_OpenCockpitLarge') -> None:
    """Enforce a symmetry along an axis.

    Args:
        structure (Structure): The structure.
        axis (str, optional): The axis to enforce symmetry along to. Defaults to 'z'.
        upper (bool, optional): Whether to keep the upper or the lower half of the structure. Defaults to True.
        pivot_blocktype (str, optional): Block type used as pivot to determine the middle point. Defaults to 'MyObjectBuilder_Cockpit_OpenCockpitLarge'.
    """
    def to_keep(v1: Vec,
                v2: Vec,
                axis: str,
                upper: bool,
                keep_equal: bool) -> bool:
        if axis == 'x':
            if upper:
                return (v1.x > v2.x) or (keep_equal and v1.x == v2.x)
            else:
                return (v1.x < v2.x) or (keep_equal and v1.x == v2.x)
        elif axis == 'y':
            if upper:
                return (v1.y > v2.y) or (keep_equal and v1.y == v2.y)
            else:
                return (v1.y < v2.y) or (keep_equal and v1.y == v2.y)
        elif axis == 'z':
            if upper:
                return (v1.z > v2.z) or (keep_equal and v1.z == v2.z)
            else:
                return (v1.z < v2.z) or (keep_equal and v1.z == v2.z)
    
    midpoint = [x for x in structure._blocks.values() if x.block_type == pivot_blocktype][0].position
    structure._blocks = {k:v for k, v in structure._blocks.items() if to_keep(v1=v.position, v2=midpoint, axis=axis, upper=upper, keep_equal=True)}
    half = [b for b in structure._blocks.values() if to_keep(v1=b.position, v2=midpoint, axis=axis, upper=upper, keep_equal=False)]
    for b in half:
        if axis == 'x':
            b.position.x = midpoint.x - (b.position.x - midpoint.x)
        elif axis == 'y':
            b.position.y = midpoint.y - (b.position.y - midpoint.y)
        elif axis == 'z':
            b.position.z = midpoint.z - (b.position.z - midpoint.z)
        structure.add_block(block=b,
                            grid_position=b.position.as_tuple())
    structure.sanify()