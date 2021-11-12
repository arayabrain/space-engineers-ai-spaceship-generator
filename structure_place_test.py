from common.api_call import get_base_values
from common.vecs import Orientation
from structure import *

def main():
    base_position, orientation_forward, orientation_up = get_base_values()
    structure = Structure(origin=base_position,
                        orientation_forward=orientation_forward,
                        orientation_up=orientation_up,
                        dimensions=(10, 10, 10))
    structure.add_block(block=Block(block_type='LargeBlockSmallGenerator',
                                    orientation_forward=Orientation.UP,
                                    orientation_up=Orientation.UP),
                        grid_position=(0, 0, 0)
                        )
    structure.add_block(block=Block(block_type='LargeBlockCockpitSeat',
                                    orientation_forward=Orientation.BACKWARD,
                                    orientation_up=Orientation.UP
                                    ),
                                    grid_position=(0, 5, 0))

    place_blocks(structure.get_all_blocks())

if __name__=="__main__":
    main()
