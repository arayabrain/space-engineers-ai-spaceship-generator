from pcgsepy.common.api_call import GameMode, get_base_values, toggle_gamemode
from pcgsepy.common.vecs import Orientation
from pcgsepy.structure import *

def main():
    toggle_gamemode(mode=GameMode.PLACING)
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
    toggle_gamemode(mode=GameMode.EVALUATING)

if __name__=="__main__":
    main()
