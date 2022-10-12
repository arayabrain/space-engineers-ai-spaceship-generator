import json
import os
from copy import deepcopy
from functools import cached_property
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pcgsepy.common.api_call import block_definitions
from pcgsepy.common.vecs import Orientation, Vec

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
        """Create a mount point.

        Args:
            face (Dict[str, int]): The face normal.
            start (Dict[str, float]): The start vector.
            end (Dict[str, float]): The end vector.
            exclusion_mask (int): The exclusion mask.
            properties_mask (int): The properties mask.
            block_size (Vec): The size of the block.
        """
        self.face = Vec.from_json(face)
        self.start = Vec.from_json(start).dot(block_size)
        self.end = Vec.from_json(end).dot(block_size)
        self.exclusion_mask = exclusion_mask
        self.properties_mask = properties_mask

    def __str__(self) -> str:
        return f'Normal: {self.face}\tStart: {self.start}\tEnd: {self.end}'

    def __repr__(self) -> str:
        return str(self.__dict__)


class Block:
    def __init__(self,
                 block_type: str,
                 orientation_forward: Orientation = Orientation.FORWARD,
                 orientation_up: Orientation = Orientation.UP,
                 position: Vec = Vec.v3f(0., 0., 0.)):
        """Create a Block object.

        Args:
            block_type (str): The type of the block (unique type from Space Engineers API).
            orientation_forward (Orientation, optional): The Forward orientation of the block. Defaults to `Orientation.FORWARD`.
            orientation_up (Orientation, optional): The Up orientation of the block. Defaults to `Orientation.UP`.
            position (Vec, optional): The position the block. Defaults to `Vec.v3f(0., 0., 0.)`.
        """
        self.block_type = block_type
        self.orientation_forward = orientation_forward.value
        self.orientation_up = orientation_up.value
        self.position = position
        self.definition_id = block_definitions[self.block_type]['definition_id']
        self.color = Vec.v3f(x=0.45, y=0.45, z=0.45)  # default block color is #737373
    
    @cached_property
    def cube_size(self) -> float:
        """Get the size of the cube block, as provided by the API.

        Returns:
            float: The size of the cube block.
        """
        return block_definitions[self.block_type]['cube_size']
    
    @cached_property
    def size(self) -> Vec:
        """Get the size of the block, as provided by the API.

        Returns:
            float: The size of the block.
        """
        return Vec.from_json(block_definitions[self.block_type]['size'])
    
    @cached_property
    def mass(self) -> float:
        """Get the mass of the block, as provided by the API.

        Returns:
            float: The mass of the block.
        """
        return float(block_definitions[self.block_type]['mass'])
    
    @cached_property
    def scaled_size(self) -> Vec:
        """Get the scaled size of the block.

        Returns:
            float: The scaled size of the block.
        """
        return self.size.scale(_blocks_sizes[self.cube_size])

    @cached_property
    def volume(self) -> float:
        """Compute the volume of the block.

        Returns:
            float: The volume of the block.
        """
        return self.scaled_size.bbox()
    
    @cached_property
    def center(self) -> Vec:
        """Get the center point of the block.

        Returns:
            Vec: The center point of the vector.
        """
        return self.scaled_size.scale(v=0.5)
    
    @cached_property
    def mountpoints(self) -> List[MountPoint]:
        """Generate the mountpoints of the block.

        Returns:
            List[MountPoint]: The list of mountpoints, one per face.
        """
        return [MountPoint(face=v['Normal'],
                           start=v['Start'],
                           end=v['End'],
                           exclusion_mask=v['ExclusionMask'],
                           properties_mask=v['PropertiesMask'],
                           block_size=self.scaled_size) for v in block_definitions[self.block_type]['mountpoints']]

    def duplicate(self,
                  new_pos: Vec) -> "Block":
        """Duplicate the current block with a new position.

        Args:
            new_pos (Vec): The new position.

        Returns:
            Block: The duplicated block.
        """
        new_block = deepcopy(self)
        new_block.position = new_pos
        return new_block

    def __str__(self) -> str:
        return f'{self.block_type} at {self.position}; OF {self.orientation_forward}; OU {self.orientation_up}'

    def __repr__(self) -> str:
        return str(self)


class IntersectionException(Exception):
    """
    Exception to throw when an intersection occurs.
    """
    pass


def _is_base_block(block_type: str) -> bool:
    """Check if the block is a base block. Base blocks are non-functional, structural blocks.

    Args:
        block_type (str): The type of the block.

    Returns:
        bool: Whether the block is a base block.
    """
    return block_type in ['LargeBlockArmorCorner', 'LargeBlockArmorSlope',
                          'LargeBlockArmorCornerInv', 'LargeBlockArmorBlock']



class Structure:
    __slots__ = ['origin_coords', 'orientation_forward', 'orientation_up', 'grid_size', '_blocks',
                 '_has_intersections', '_scaled_arr', '_air_gridmask', '_arr']
    
    def __init__(self, origin: Vec,
                 orientation_forward: Vec,
                 orientation_up: Vec,
                 grid_size: int = 5) -> None:
        """Create a Structure object. A Structure is similar to the `GridBlocks` in Space Engineers' API.

        Args:
            origin (Vec): The XYZ origin coordinates of the Structure.
            orientation_forward (Vec): The Forward orientation of the structure as a `Vec` object.
            orientation_up (Vec): The Up orientation of the structure as a `Vec` object.
            grid_size (int): The size of the grid. Defaults to `5.`.
        """
        self.origin_coords = origin
        self.orientation_forward = orientation_forward
        self.orientation_up = orientation_up
        self.grid_size = grid_size

        self._blocks: Dict[Tuple(int, int, int), Block] = {}
        self._has_intersections: bool = None
        self._scaled_arr: npt.NDArray[np.uint32] = None
        self._air_gridmask: npt.NDArray[np.bool8] = None
        self._arr: npt.NDArray[np.uint32] = None

    def __repr__(self) -> str:
        return f'{self.grid_size}x Structure with {len(self._blocks.keys())} blocks'
    
    def add_block(self,
                  block: Block,
                  grid_position: Tuple[int, int, int]) -> None:
        """Add a block to the structure.

        Args:
            block (Block): The block to add.
            grid_position (Tuple[int, int, int]): The position in the grid at which the block is placed. Obtained from `Vec.to_tuple()`.
            exit_on_duplicates (bool): Flag to check for existing blocks when adding new ones. May raise an `IntersectionException`.

        Raises:
            IntersectionException: Raised if an intersection with another block occurrs.
        """
        i, j, k = grid_position
        block.position = Vec.v3i(i, j, k)
        
        if grid_position in self._blocks.keys():
            self._has_intersections = True
        
        self._blocks[(i, j, k)] = block
    
    def set_color(self,
                  color: Vec) -> None:
        """Set the color of the base blocks in the structure.

        Args:
            color (Vec): The color as RGB values vector.
        """
        for block in self._blocks.values():
            if _is_base_block(block_type=self._clean_label(a=block.block_type)):
                block.color = color
    
    @property
    def _max_dims(self) -> Tuple[int, int, int]:
        """Compute the maximum dimension of the Structure.

        Returns:
            Tuple[int, int, int]: The XYZ maximum dimensions
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

    @property
    def _min_dims(self) -> Tuple[int, int, int]:
        """Compute the minimum dimension of the Structure.

        Returns:
            Tuple[int, int, int]: The XYZ minimum dimensions.
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
    def as_array(self) -> npt.NDArray[np.uint32]:
        """Convert the structure to its equivalent NumPy array.
        Each point in the XYZ matrix represents the block type.

        Returns:
            npt.NDArray[np.uint32]: The 3D NumPy array.
        """
        if self._scaled_arr is None:
            self._scaled_arr = np.zeros(shape=Vec.from_tuple(self._max_dims).add(v=self.grid_size).as_tuple(), dtype=np.uint32)
            for (i, j, k), block in self._blocks.items():
                r = block.scaled_size
                if np.sum(self._scaled_arr[i:i + r.x, j:j + r.y, k:k + r.z]) != 0:
                    self._has_intersections = True
                self._scaled_arr[i:i + r.x, j:j + r.y, k:k + r.z] = list(block_definitions.keys()).index(block.block_type) + 1
        return self._scaled_arr

    @property
    def as_grid_array(self) -> npt.NDArray[np.uint32]:
        """Convert the structure to the grid-sized array.
        Each point in the XYZ matrix represents the block type.

        Returns:
            npt.NDArray[np.uint32]: The 3D NumPy grid-sized array.
        """
        if self._arr is None:
            self._arr = np.zeros(shape=Vec.from_tuple(self._max_dims).scale(v=1 / self.grid_size).to_veci().add(v=1).as_tuple(), dtype=np.uint32)
            for r, block in self._blocks.items():
                r = Vec.from_tuple(r).scale(v=1 / self.grid_size).to_veci().as_tuple()
                if np.sum(self._arr[r]) != 0:
                    self._has_intersections = True
                self._arr[r] = list(block_definitions.keys()).index(block.block_type) + 1
        return self._arr

    @property
    def has_intersections(self) -> bool:
        """Check if the Structure contains an intersection between blocks.

        Returns:
            bool: Whether there is an intersection.
        """
        if self._has_intersections is None:
            _ = self.as_array
            if self._has_intersections is None:
                self._has_intersections = False
        return self._has_intersections
    
    @property
    def total_volume(self) -> float:
        """Compute the volume of the grid.

        Returns:
            float: The volume of the grid.
        """
        return sum([b.volume for b in self._blocks.values()])
    
    @property
    def mass(self) -> float:
        """Compute the mass of the grid.

        Returns:
            float: The mass of the grid.
        """
        return np.round(sum([b.mass for b in self._blocks.values()]), 2)
    
    @property
    def blocks_count(self) -> Tuple[int, int]:
        """Count armor blocks and non-armor blocks contained in the grid.

        Returns:
            Tuple[int, int]: The number of armor and non-armor blocks.
        """
        armor_blocks = sum([1 if 'armor' in x.block_type.lower() else 0 for x in self._blocks.values()])
        return armor_blocks, len(self._blocks) - armor_blocks
    
    def unique_blocks_count(self,
                            block_type: str) -> int:
        """Count the number of blocks with the given block type.

        Args:
            block_type (str): The block type.

        Returns:
            int: The number of blocks with the given block type.
        """
        return sum([1 if x.block_type == block_type else 0 for x in self._blocks.values()])
    
    @property
    def air_blocks_gridmask(self) -> npt.NDArray[np.bool8]:
        """Get the grid array of internal air blocks in the structure.

        Returns:
            npt.NDArray[np.bool8]: A boolean array where `True` elements are internal air blocks in the grid array.
        """
        if self._air_gridmask is None:
            self._air_gridmask = np.zeros_like(self.as_grid_array, dtype=np.bool8)
            # Old code, reliable but very slow
            # ds = [Vec.v3i(1, 0, 0), Vec.v3i(0, 1, 0), Vec.v3i(0, 0, 1),
            #     Vec.v3i(-1, 0, 0), Vec.v3i(0, -1, 0), Vec.v3i(0, 0, -1)]
            # # get existing blocks indices
            # blocks_idxs = [Vec.from_tuple(k).scale(v=1 / self.grid_size).to_veci() for k in self._blocks.keys()]
            # # get all indices attached to the blocks indices
            # next_to_idxs = [b.sum(d) for d in ds for b in blocks_idxs if b.sum(d) not in blocks_idxs]
            # # internal air cotner blocks have at least 3 blocks next to them
            # internal_air = [Vec.from_tuple(t) for t in  list({b.as_tuple() : next_to_idxs.count(b) for b in next_to_idxs if next_to_idxs.count(b) > 2}.keys())]
            # # checking loop
            # past = blocks_idxs
            # while internal_air:
            #     to_check = []
            #     for idx in internal_air:
            #         if 0 <= idx.x < self._air_gridmask.shape[0] and 0 <= idx.y < self._air_gridmask.shape[1] and 0 <= idx.z < self._air_gridmask.shape[2] and\
            #             not self._air_gridmask[idx.as_tuple()]:
            #                 past.append(idx)
            #                 if self._blocks.get(idx.scale(self.grid_size).as_tuple(), None) is None:
            #                     self._air_gridmask[idx.as_tuple()] = True
            #                     to_check.extend([idx.sum(d) for d in ds if idx.sum(d) not in past])
            #     internal_air = list(set(to_check))
            # new code, faster but less readable
            i1, j1, k1 = self.as_grid_array.shape
            for (i, j, k) in zip(*np.nonzero(self.as_grid_array == 0)):
                # basically, check for every empty block index if it's surrounded on all 6 sides by at least a block
                self._air_gridmask[i, j, k] = np.sum(self.as_grid_array[0:i, j, k]) != 0 and \
                    np.sum(self.as_grid_array[i:i1, j, k]) != 0 and \
                    np.sum(self.as_grid_array[i, 0:j, k]) != 0 and \
                    np.sum(self.as_grid_array[i, j:j1, k]) != 0 and \
                    np.sum(self.as_grid_array[i, j, 0:k]) != 0 and \
                    np.sum(self.as_grid_array[i, j, k:k1]) != 0
        return self._air_gridmask
    
    def sanify(self) -> None:
        """Correct the structure's blocks to be >=0 on every axis."""
        min_x, min_y, min_z = self._min_dims
        updated_blocks = {}
        for x, y, z in self._blocks.keys():
            block = self._blocks[(x, y, z)]
            new_pos = Vec.v3i(x=x - min_x, y=y - min_y, z=z - min_z)
            block.position = self.origin_coords.sum(new_pos)
            updated_blocks[new_pos.as_tuple()] = block
        self._blocks = updated_blocks
        self._scaled_arr = None
        self._arr = None

    def update(self, origin: Vec,
               orientation_forward: Vec,
               orientation_up: Vec) -> None:
        """Update the structure position and orientations.

        Args:
            origin (Vec): The new XYZ origin coordinates of the Structure.
            orientation_forward (Vec): The new Forward orientation of the structure as a `Vec` object.
            orientation_up (Vec): The new Up orientation of the structure as a `Vec` object.
        """
        # update structure position and orientation
        self.origin_coords = origin
        self.orientation_forward = orientation_forward
        self.orientation_up = orientation_up
        # update all blocks accordingly
        self.sanify()

    def get_all_blocks(self,
                       to_place: bool = True,
                       scaled: bool = False) -> List[Block]:
        """Get all the blocks in the structure.

        Args:
            to_place (bool): Flag for placement position correction. Dafaults to `True`.
            scaled (bool): Flag for block grid position correction. Defaults to `False`.

        Returns:
            List[Block]: The list of all blocks.
        """
        all_blocks = list(self._blocks.values())
        if to_place:
            return [b.duplicate(b.position.scale(grid_to_coords)) for b in all_blocks]
        elif scaled:
            return [b.duplicate(b.position.scale(1 / self.grid_size)) for b in all_blocks]
        else:
            return all_blocks

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
            'MyObjectBuilder_InteriorLight_'
        ]:
            a = a.replace(d, '')
        return a

    def show(self,
             title: str,
             title_len: int = 90,
             save: bool = False) -> plt.Axes:
        """Plot the structure.

        Args:
            title (str): Title of the plot.
            title_len (int, optional): Maximum length of the title. Defaults to `90`.
            save (bool, optional): Flag to salve the plot as picture. Defaults to `False`.

        Returns:
            plt.Axes: The figure object.
        """
        structure = self.as_grid_array
        ax = plt.axes(projection='3d')
        x, y, z = np.nonzero(structure)
        cs = [structure[i, j, k] for i, j, k in zip(x, y, z)]
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        scatter = ax.scatter(x, y, z, c=cs, cmap='jet', linewidth=0.1)
        legend = scatter.legend_elements(num=len(np.unique(structure[(x, y, z)])) - 1)
        for i, v in zip(range(len(legend[1])), np.unique(structure[(x, y, z)])):
            legend[1][i] = self._clean_label(list(block_definitions.keys())[v - 1])
        ax.legend(*legend,
                  bbox_to_anchor=(1.2, 1),
                  loc="upper left",
                  title="Block types")
        axis_limit = Vec.from_tuple(self._max_dims).scale(1 / self.grid_size).as_tuple()
        ax.set_xlim3d(0, axis_limit[0])
        ax.set_ylim3d(0, axis_limit[1])
        ax.set_zlim3d(0, axis_limit[2])
        ax.set_xlabel("$\\vec{x}$")
        ax.set_ylabel("$\\vec{y}$")
        ax.set_zlabel("$\\vec{z}$")
        plt.title(title if len(title) <= title_len else f'{title[:title_len - 3]}...')
        plt.autoscale(enable=True,
                      axis='x',
                      tight=True)
        if save:
            plt.savefig('content_plot.png',
                        transparent=True,
                        bbox_inches='tight')
        plt.show()
        return ax
