import os
import random
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple

from .common.vecs import Orientation, orientation_from_vec, Vec
from .structure import Block, Structure

to_orientation = {
    'Up': Orientation.UP,
    'Down': Orientation.DOWN,
    'Left': Orientation.LEFT,
    'Right': Orientation.RIGHT,
    'Forward': Orientation.FORWARD,
    'Backward': Orientation.BACKWARD
}

orientations_str = {
    Orientation.FORWARD: 'F',
    Orientation.BACKWARD: 'B',
    Orientation.RIGHT: 'R',
    Orientation.LEFT: 'L',
    Orientation.UP: 'U',
    Orientation.DOWN: 'D',
}

grid_enum_to_offset = {'Small': 1, 'Normal': 2, 'Large': 5}


def convert_xml_to_structure(root_node: ET.Element,
                             struct_dim: int = 100) -> Structure:
    """
    Convert the XML-defined structure to a `Structure` object.

    Parameters
    ----------
    root_node : ET.Element
        The XML root node.
    struct_dim : int
        The dimension of the Structure.

    Returns
    -------
    Structure
        The converted Structure object.
    """
    structure = Structure(origin=Vec.v3f(0., 0., 0.),
                          orientation_forward=Orientation.FORWARD.value,
                          orientation_up=Orientation.UP.value)
    for grid in root_node.findall('.//CubeGrid'):
        grid_size = None
        for child in grid:
            if child.tag == 'GridSizeEnum':
                grid_size = grid_enum_to_offset[child.text]
            elif child.tag == 'CubeBlocks':
                for block_node in child:  # CubeBlocks node
                    if block_node.tag.startswith('MyObjectBuilder_'):
                        block_type = ''
                        position = (0, 0, 0)
                        orientations = {
                            'Forward': Orientation.FORWARD,
                            'Up': Orientation.UP
                        }
                        for p in block_node:
                            if p.tag == 'SubtypeName':
                                block_type = '_'.join([block_node.attrib['{http://www.w3.org/2001/XMLSchema-instance}type'],
                                                       p.text if p.text else ''])
                            elif p.tag == 'Min':
                                position = (grid_size * int(p.attrib['x']),
                                            grid_size * int(p.attrib['y']),
                                            grid_size * int(p.attrib['z']))
                            elif p.tag == 'BlockOrientation':
                                orientations = {
                                    'Forward':
                                        to_orientation[p.attrib['Forward']],
                                    'Up':
                                        to_orientation[p.attrib['Up']]
                                }
                        if not block_type:
                            continue
                        block = Block(
                            block_type=block_type,
                            orientation_forward=orientations['Forward'],
                            orientation_up=orientations['Up'])
                        structure.add_block(block=block, grid_position=position)
    return structure


def _at_same_x(x: int,
               blocks: List[Block]) -> List[Block]:
    r = []
    for b in blocks:
        if b.position.x == x:
            r.append(b)
    return r


def _at_same_y(y: int,
               blocks: List[Block]) -> List[Block]:
    r = []
    for b in blocks:
        if b.position.y == y:
            r.append(b)
    return r


def _at_same_z(z: int,
               blocks: List[Block]) -> List[Block]:
    r = []
    for b in blocks:
        if b.position.z == z:
            r.append(b)
    return r


# TODO: refactor/reduce complexity & return offset
def extract_rule(bp_dir: str,
                 title: str = '') -> Tuple[str, Tuple[int, int, int]]:
    bp = os.path.join(bp_dir, 'bp.sbc')
    root = ET.parse(bp).getroot()
    structure = convert_xml_to_structure(root_node=root)
    structure.sanify()
    structure.show(title)
    blocks = structure.get_all_blocks()

    max_x, max_y, max_z = 0., 0., 0.
    for block in blocks:
        x, y, z = block.position.as_tuple()
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
        if z > max_z:
            max_z = z

    min_x, min_y, min_z = max_x, max_y, max_z
    for block in blocks:
        x, y, z = block.position.as_tuple()
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if z < min_z:
            min_z = z

    for block in blocks:
        new_p = block.position.sum(Vec.v3f(-min_x, -min_y, -min_z))
        block.position = Vec.v3f(x=np.round(new_p.x, 1),
                                 y=np.round(new_p.y, 1),
                                 z=np.round(new_p.z, 1))

    max_x -= min_x
    max_y -= min_y
    max_z -= min_z

    ordered_blocks = []
    x, y, z = 0., 0., 0.
    while z <= max_z:
        bs1 = _at_same_z(z, blocks)
        while y <= max_y:
            bs2 = _at_same_y(y, bs1)
            while x <= max_x:
                b = _at_same_x(x, bs2)
                if b:
                    ordered_blocks.append(b[0])
                x += 0.5
            x = 0.
            y += 0.5
        x = 0.
        y = 0.
        z += 0.5

    rule = ''
    x, y, z = 0., 0., 0.
    for block in ordered_blocks:
        if block.position.z != z:
            if block.position.z > z:
                dz = block.position.z - z
                rule += f'<({int(dz // 0.5)})'
                z = block.position.z
            else:
                dz = z - block.position.z
                rule += f'>({int(dz // 0.5)})'
                z = block.position.z
        if block.position.y != y:
            if block.position.y > y:
                dy = block.position.y - y
                rule += f'!({int(dy // 0.5)})'
                y = block.position.y
            else:
                dy = y - block.position.y
                rule += f'?({int(dy // 0.5)})'
                y = block.position.y
        if block.position.x != x:
            if block.position.x > x:
                dx = block.position.x - x
                rule += f'+({int(dx // 0.5)})'
                x = block.position.x
            else:
                dx = x - block.position.x
                rule += f'-({int(dx // 0.5)})'
                x = block.position.x
        of = orientations_str[orientation_from_vec(block.orientation_forward)]
        ou = orientations_str[orientation_from_vec(block.orientation_up)]
        rule += f'{block.block_type}({of},{ou})'

    if x != 0.:
        if x > 0.:
            rule += f'-({int(x // 0.5)})'
        if x < 0.:
            rule += f'+({int(-x // 0.5)})'
    if y != 0.:
        if y > 0.:
            rule += f'?({int(y // 0.5)})'
        if y < 0.:
            rule += f'!({int(-y // 0.5)})'
    if z != 0.:
        if z > 0.:
            rule += f'>({int(z // 0.5)})'
        if z < 0.:
            rule += f'<({int(-z // 0.5)})'

    x, y, z = structure._max_dims
    # TODO: change this from the grid size
    x += 5
    y += 5
    z += 5

    return rule, (x, y, z)


def structure_xml_converter(structure: Structure,
                            name: str) -> str:
    
    builder_id = str(random.getrandbits(64)).zfill(20)
    
    def armour_blocks(block: Block) -> str:
        builder, xsi, block_type = block.block_type.split('_')
        pos = block.position.scale(v=1 / structure.grid_size).to_veci()
        return f"""<MyObjectBuilder_CubeBlock xsi:type="{builder}_{xsi}">
            <SubtypeName>{block_type}</SubtypeName>
            <Min x = "{pos.x}" y="{pos.y}" z="{pos.z}" />
            <BlockOrientation Forward="{orientations_str[orientation_from_vec(block.orientation_forward)]}" Up="{orientations_str[orientation_from_vec(block.orientation_up)]}" />
            <ColorMaskHSV x="0" y="-0.8" z="0.55" />
        </MyObjectBuilder_CubeBlock>
        """
    
    def reactor_block(block: Block) -> str:
        builder, xsi, block_type = block.block_type.split('_')
        pos = block.position.scale(v=1 / structure.grid_size).to_veci()
        return f"""<MyObjectBuilder_CubeBlock xsi:type="{builder}_{xsi}">
              <SubtypeName>{block_type}</SubtypeName>
              <EntityId>{str(random.getrandbits(32)).zfill(16)}</EntityId>
              <Min x = "{pos.x}" y="{pos.y}" z="{pos.z}" />
              <BlockOrientation Forward="{orientations_str[orientation_from_vec(block.orientation_forward)]}" Up="{orientations_str[orientation_from_vec(block.orientation_up)]}" />
              <ColorMaskHSV x="0.575" y="0" z="0" />
              <BuiltBy>{builder_id}</BuiltBy>
              <ComponentContainer>
                <Components>
                  <ComponentData>
                    <TypeId>MyInventoryBase</TypeId>
                    <Component xsi:type="MyObjectBuilder_Inventory">
                      <Items>
                        <MyObjectBuilder_InventoryItem>
                          <Amount>19.996409</Amount>
                          <PhysicalContent xsi:type="MyObjectBuilder_Ingot">
                            <SubtypeName>Uranium</SubtypeName>
                          </PhysicalContent>
                          <ItemId>0</ItemId>
                        </MyObjectBuilder_InventoryItem>
                      </Items>
                      <nextItemId>1</nextItemId>
                      <Volume>1</Volume>
                      <Mass>9223372036854.775807</Mass>
                      <MaxItemCount>2147483647</MaxItemCount>
                      <Size xsi:nil="true" />
                      <InventoryFlags>CanReceive</InventoryFlags>
                      <RemoveEntityOnEmpty>false</RemoveEntityOnEmpty>
                    </Component>
                  </ComponentData>
                  <ComponentData>
                    <TypeId>MyTimerComponent</TypeId>
                    <Component xsi:type="MyObjectBuilder_TimerComponent">
                      <Repeat>true</Repeat>
                      <TimeToEvent>0</TimeToEvent>
                      <SetTimeMinutes>0</SetTimeMinutes>
                      <TimerEnabled>false</TimerEnabled>
                      <RemoveEntityOnTimer>false</RemoveEntityOnTimer>
                      <TimerType>Frame100</TimerType>
                      <FramesFromLastTrigger>0</FramesFromLastTrigger>
                      <TimerTickInFrames>900</TimerTickInFrames>
                      <IsSessionUpdateEnabled>false</IsSessionUpdateEnabled>
                    </Component>
                  </ComponentData>
                </Components>
              </ComponentContainer>
              <CustomName>{block_type}</CustomName>
              <ShowOnHUD>false</ShowOnHUD>
              <ShowInTerminal>true</ShowInTerminal>
              <ShowInToolbarConfig>true</ShowInToolbarConfig>
              <ShowInInventory>true</ShowInInventory>
              <Enabled>true</Enabled>
              <Capacity>19.9964085</Capacity>
            </MyObjectBuilder_CubeBlock>"""
    
    def container_block(block: Block) -> str:
        builder, xsi, block_type = block.block_type.split('_')
        pos = block.position.scale(v=1 / structure.grid_size).to_veci()
        return f"""<MyObjectBuilder_CubeBlock xsi:type="{builder}_{xsi}">
              <SubtypeName>{block_type}</SubtypeName>
              <EntityId>{str(random.getrandbits(32)).zfill(16)}</EntityId>
              <Min x="{pos.x}" y="{pos.y}" z="{pos.z}" />
              <BlockOrientation Forward="{orientations_str[orientation_from_vec(block.orientation_forward)]}" Up="{orientations_str[orientation_from_vec(block.orientation_up)]}" />
              <ColorMaskHSV x="0.575" y="0" z="0" />
              <BuiltBy>{builder_id}</BuiltBy>
              <ComponentContainer>
                <Components>
                  <ComponentData>
                    <TypeId>MyInventoryBase</TypeId>
                    <Component xsi:type="MyObjectBuilder_Inventory">
                      <Items />
                      <nextItemId>17</nextItemId>
                      <Volume>15.625</Volume>
                      <Mass>9223372036854.775807</Mass>
                      <MaxItemCount>2147483647</MaxItemCount>
                      <Size xsi:nil="true" />
                      <InventoryFlags>CanReceive CanSend</InventoryFlags>
                      <RemoveEntityOnEmpty>false</RemoveEntityOnEmpty>
                    </Component>
                  </ComponentData>
                </Components>
              </ComponentContainer>
              <CustomName>{block_type}</CustomName>
              <ShowOnHUD>false</ShowOnHUD>
              <ShowInTerminal>true</ShowInTerminal>
              <ShowInToolbarConfig>true</ShowInToolbarConfig>
              <ShowInInventory>true</ShowInInventory>
            </MyObjectBuilder_CubeBlock>"""
    
    def thruster_block(block: Block) -> str:
        builder, xsi, block_type = block.block_type.split('_')
        pos = block.position.scale(v=1 / structure.grid_size).to_veci()
        return f"""<MyObjectBuilder_CubeBlock xsi:type="{builder}_{xsi}">
              <SubtypeName>{block_type}</SubtypeName>
              <EntityId>{str(random.getrandbits(32)).zfill(16)}</EntityId>
              <Min x= "{pos.x}" y="{pos.y}" z="{pos.z}" />
              <BlockOrientation Forward="{orientations_str[orientation_from_vec(block.orientation_forward)]}" Up="{orientations_str[orientation_from_vec(block.orientation_up)]}" />
              <BuiltBy>{builder_id}</BuiltBy>
              <ComponentContainer>
                <Components>
                  <ComponentData>
                    <TypeId>MyTimerComponent</TypeId>
                    <Component xsi:type="MyObjectBuilder_TimerComponent">
                      <Repeat>true</Repeat>
                      <TimeToEvent>0</TimeToEvent>
                      <SetTimeMinutes>0</SetTimeMinutes>
                      <TimerEnabled>true</TimerEnabled>
                      <RemoveEntityOnTimer>false</RemoveEntityOnTimer>
                      <TimerType>Frame100</TimerType>
                      <FramesFromLastTrigger>0</FramesFromLastTrigger>
                      <TimerTickInFrames>100</TimerTickInFrames>
                      <IsSessionUpdateEnabled>false</IsSessionUpdateEnabled>
                    </Component>
                  </ComponentData>
                </Components>
              </ComponentContainer>
              <CustomName>{block_type}</CustomName>
              <ShowOnHUD>false</ShowOnHUD>
              <ShowInTerminal>true</ShowInTerminal>
              <ShowInToolbarConfig>true</ShowInToolbarConfig>
              <ShowInInventory>true</ShowInInventory>
              <Enabled>true</Enabled>
            </MyObjectBuilder_CubeBlock>"""
    
    def collector_block(block: Block) -> str:
        builder, xsi, block_type = block.block_type.split('_')
        pos = block.position.scale(v=1 / structure.grid_size).to_veci()
        return f"""<MyObjectBuilder_CubeBlock xsi:type="{builder}_{xsi}">
              <SubtypeName>{block_type}</SubtypeName>
              <EntityId>{str(random.getrandbits(32)).zfill(16)}</EntityId>
              <Min x="{pos.x}" y="{pos.y}" z="{pos.z}" />
              <BlockOrientation Forward="{orientations_str[orientation_from_vec(block.orientation_forward)]}" Up="{orientations_str[orientation_from_vec(block.orientation_up)]}" />
              <ColorMaskHSV x="0.575" y="0" z="0" />
              <BuiltBy>{builder_id}</BuiltBy>
              <ComponentContainer>
                <Components>
                  <ComponentData>
                    <TypeId>MyInventoryBase</TypeId>
                    <Component xsi:type="MyObjectBuilder_Inventory">
                      <Items />
                      <nextItemId>0</nextItemId>
                      <Volume>1.575</Volume>
                      <Mass>9223372036854.775807</Mass>
                      <MaxItemCount>2147483647</MaxItemCount>
                      <Size xsi:nil="true" />
                      <InventoryFlags>CanSend</InventoryFlags>
                      <RemoveEntityOnEmpty>false</RemoveEntityOnEmpty>
                    </Component>
                  </ComponentData>
                </Components>
              </ComponentContainer>
              <CustomName>{block_type}</CustomName>
              <ShowOnHUD>false</ShowOnHUD>
              <ShowInTerminal>true</ShowInTerminal>
              <ShowInToolbarConfig>true</ShowInToolbarConfig>
              <ShowInInventory>true</ShowInInventory>
              <Enabled>true</Enabled>
            </MyObjectBuilder_CubeBlock>"""
    
    def gyro_block(block: Block) -> str:
        builder, xsi, block_type = block.block_type.split('_')
        pos = block.position.scale(v=1 / structure.grid_size).to_veci()
        return f"""<MyObjectBuilder_CubeBlock xsi:type="{builder}_{xsi}">
              <SubtypeName>{block_type}</SubtypeName>
              <EntityId>{str(random.getrandbits(32)).zfill(16)}</EntityId>
              <Min x="{pos.x}" y="{pos.y}" z="{pos.z}" />
              <BlockOrientation Forward="{orientations_str[orientation_from_vec(block.orientation_forward)]}" Up="{orientations_str[orientation_from_vec(block.orientation_up)]}" />
              <ColorMaskHSV x="0.575" y="0" z="0" />
              <BuiltBy>{builder_id}</BuiltBy>
              <CustomName>{block_type}</CustomName>
              <ShowOnHUD>false</ShowOnHUD>
              <ShowInTerminal>true</ShowInTerminal>
              <ShowInToolbarConfig>true</ShowInToolbarConfig>
              <ShowInInventory>true</ShowInInventory>
              <Enabled>true</Enabled>
            </MyObjectBuilder_CubeBlock>"""
    
    def oxygen_block(block: Block) -> str:
        builder, xsi, block_type = block.block_type.split('_')
        pos = block.position.scale(v=1 / structure.grid_size).to_veci()
        return f"""<MyObjectBuilder_CubeBlock xsi:type="{builder}_{xsi}">
              <SubtypeName>{block_type}</SubtypeName>
              <EntityId>{str(random.getrandbits(32)).zfill(16)}</EntityId>
              <Min x="{pos.x}" y="{pos.y}" z="{pos.z}" />
              <BlockOrientation Forward="{orientations_str[orientation_from_vec(block.orientation_forward)]}" Up="{orientations_str[orientation_from_vec(block.orientation_up)]}" />
              <ColorMaskHSV x="0.575" y="0" z="0" />
              <BuiltBy>{builder_id}</BuiltBy>
              <ComponentContainer>
                <Components>
                  <ComponentData>
                    <TypeId>MyInventoryBase</TypeId>
                    <Component xsi:type="MyObjectBuilder_Inventory">
                      <Items>
                        <MyObjectBuilder_InventoryItem>
                          <Amount>999.149691</Amount>
                          <PhysicalContent xsi:type="MyObjectBuilder_Ore">
                            <SubtypeName>Ice</SubtypeName>
                          </PhysicalContent>
                          <ItemId>0</ItemId>
                        </MyObjectBuilder_InventoryItem>
                      </Items>
                      <nextItemId>1</nextItemId>
                      <Volume>1</Volume>
                      <Mass>9223372036854.775807</Mass>
                      <MaxItemCount>2147483647</MaxItemCount>
                      <Size xsi:nil="true" />
                      <InventoryFlags>CanReceive</InventoryFlags>
                      <RemoveEntityOnEmpty>false</RemoveEntityOnEmpty>
                    </Component>
                  </ComponentData>
                  <ComponentData>
                    <TypeId>MyTimerComponent</TypeId>
                    <Component xsi:type="MyObjectBuilder_TimerComponent">
                      <Repeat>true</Repeat>
                      <TimeToEvent>0</TimeToEvent>
                      <SetTimeMinutes>0</SetTimeMinutes>
                      <TimerEnabled>false</TimerEnabled>
                      <RemoveEntityOnTimer>false</RemoveEntityOnTimer>
                      <TimerType>Frame10</TimerType>
                      <FramesFromLastTrigger>240</FramesFromLastTrigger>
                      <TimerTickInFrames>300</TimerTickInFrames>
                      <IsSessionUpdateEnabled>false</IsSessionUpdateEnabled>
                    </Component>
                  </ComponentData>
                </Components>
              </ComponentContainer>
              <CustomName>{block_type}</CustomName>
              <ShowOnHUD>false</ShowOnHUD>
              <ShowInTerminal>true</ShowInTerminal>
              <ShowInToolbarConfig>true</ShowInToolbarConfig>
              <ShowInInventory>true</ShowInInventory>
              <Enabled>true</Enabled>
            </MyObjectBuilder_CubeBlock>"""
    
    def lightcorner_block(block: Block) -> str:
        builder, xsi, block_type = block.block_type.split('_', maxsplit=2)
        pos = block.position.scale(v=1 / structure.grid_size).to_veci()
        return f"""<MyObjectBuilder_CubeBlock xsi:type="{builder}_{xsi}">
              <SubtypeName>{block_type}</SubtypeName>
              <EntityId>{str(random.getrandbits(32)).zfill(16)}</EntityId>
              <Min x="{pos.x}" y="{pos.y}" z="{pos.z}" />
              <BlockOrientation Forward="{orientations_str[orientation_from_vec(block.orientation_forward)]}" Up="{orientations_str[orientation_from_vec(block.orientation_up)]}" />
              <BuiltBy>{builder_id}</BuiltBy>
              <CustomName>{block_type}</CustomName>
              <ShowOnHUD>false</ShowOnHUD>
              <ShowInTerminal>true</ShowInTerminal>
              <ShowInToolbarConfig>true</ShowInToolbarConfig>
              <ShowInInventory>true</ShowInInventory>
              <Enabled>true</Enabled>
              <Radius>2</Radius>
              <ReflectorRadius>120</ReflectorRadius>
              <Falloff>1</Falloff>
              <Intensity>4</Intensity>
              <BlinkIntervalSeconds>0</BlinkIntervalSeconds>
              <BlinkLenght>10</BlinkLenght>
              <BlinkOffset>0</BlinkOffset>
              <Offset>0.5</Offset>
            </MyObjectBuilder_CubeBlock>"""
    
    def light_block(block: Block) -> str:
        builder, xsi, block_type = block.block_type.split('_')
        pos = block.position.scale(v=1 / structure.grid_size).to_veci()
        return f"""<MyObjectBuilder_CubeBlock xsi:type="{builder}_{xsi}">
              <SubtypeName>{block_type}</SubtypeName>
              <!-- <EntityId>{str(random.getrandbits(32)).zfill(16)}</EntityId> -->
              <EntityId>{random.randint(1, 99999999999)}</EntityId>
              <Min x="{pos.x}" y="{pos.y}" z="{pos.z}" />
              <BlockOrientation Forward="{orientations_str[orientation_from_vec(block.orientation_forward)]}" Up="{orientations_str[orientation_from_vec(block.orientation_up)]}" />
              <BuiltBy>{builder_id}</BuiltBy>
              <CustomName>{block_type}</CustomName>
              <ShowOnHUD>false</ShowOnHUD>
              <ShowInTerminal>true</ShowInTerminal>
              <ShowInToolbarConfig>true</ShowInToolbarConfig>
              <ShowInInventory>true</ShowInInventory>
              <Enabled>true</Enabled>
              <Radius>3.6</Radius>
              <ReflectorRadius>120</ReflectorRadius>
              <Falloff>1.3</Falloff>
              <Intensity>5</Intensity>
              <BlinkIntervalSeconds>0</BlinkIntervalSeconds>
              <BlinkLenght>10</BlinkLenght>
              <BlinkOffset>0</BlinkOffset>
              <Offset>0.5</Offset>
            </MyObjectBuilder_CubeBlock>"""
    
    def cockpit_block(block: Block) -> str:
      builder, xsi, block_type = block.block_type.split('_')
      pos = block.position.scale(v=1 / structure.grid_size).to_veci()
      return f"""<MyObjectBuilder_CubeBlock xsi:type="{builder}_{xsi}">
        <SubtypeName>{block_type}</SubtypeName>
        <EntityId>{str(random.getrandbits(32)).zfill(16)}</EntityId>
        <Min x="{pos.x}" y="{pos.y}" z="{pos.z}" />
        <BlockOrientation Forward="{orientations_str[orientation_from_vec(block.orientation_forward)]}" Up="{orientations_str[orientation_from_vec(block.orientation_up)]}" />
        <Owner>{builder_id}</Owner>
        <BuiltBy>{builder_id}</BuiltBy>
        <ShareMode>Faction</ShareMode>
        <ComponentContainer>
        <Components>
          <ComponentData>
          <TypeId>MyInventoryBase</TypeId>
          <Component xsi:type="MyObjectBuilder_Inventory">
            <Items />
            <nextItemId>0</nextItemId>
            <Volume>1</Volume>
            <Mass>9223372036854.775807</Mass>
            <MaxItemCount>2147483647</MaxItemCount>
            <Size xsi:nil="true" />
            <InventoryFlags>CanReceive CanSend</InventoryFlags>
            <RemoveEntityOnEmpty>false</RemoveEntityOnEmpty>
          </Component>
          </ComponentData>
        </Components>
        </ComponentContainer>
        <ShowOnHUD>false</ShowOnHUD>
        <ShowInTerminal>true</ShowInTerminal>
        <ShowInToolbarConfig>true</ShowInToolbarConfig>
        <ShowInInventory>true</ShowInInventory>
        <NumberInGrid>1</NumberInGrid>
        <UseSingleWeaponMode>false</UseSingleWeaponMode>
        <Toolbar>
        <ToolbarType>Character</ToolbarType>
        <SelectedSlot xsi:nil="true" />
        <Slots />
        <SlotsGamepad />
        </Toolbar>
        <SelectedGunId xsi:nil="true" />
        <BuildToolbar>
        <ToolbarType>Character</ToolbarType>
        <SelectedSlot xsi:nil="true" />
        <Slots />
        <SlotsGamepad />
        </BuildToolbar>
        <OnLockedToolbar>
        <ToolbarType>Character</ToolbarType>
        <SelectedSlot xsi:nil="true" />
        <Slots />
        <SlotsGamepad />
        </OnLockedToolbar>
        <PilotRelativeWorld xsi:nil="true" />
        <PilotGunDefinition xsi:nil="true" />
        <IsInFirstPersonView>true</IsInFirstPersonView>
        <OxygenLevel>0</OxygenLevel>
        <PilotJetpackEnabled xsi:nil="true" />
        <TextPanels>
        <MySerializedTextPanelData>
          <ChangeInterval>0</ChangeInterval>
          <Font Type="MyObjectBuilder_FontDefinition" Subtype="Debug" />
          <FontSize>1</FontSize>
          <ShowText>NONE</ShowText>
          <FontColor>
          <PackedValue>4294967295</PackedValue>
          <X>255</X>
          <Y>255</Y>
          <Z>255</Z>
          <R>255</R>
          <G>255</G>
          <B>255</B>
          <A>255</A>
          </FontColor>
          <BackgroundColor>
          <PackedValue>4278190080</PackedValue>
          <X>0</X>
          <Y>0</Y>
          <Z>0</Z>
          <R>0</R>
          <G>0</G>
          <B>0</B>
          <A>255</A>
          </BackgroundColor>
          <CurrentShownTexture>0</CurrentShownTexture>
          <ContentType>SCRIPT</ContentType>
          <SelectedScript>TSS_ArtificialHorizon</SelectedScript>
          <TextPadding>2</TextPadding>
          <ScriptBackgroundColor>
          <PackedValue>4288108544</PackedValue>
          <X>0</X>
          <Y>88</Y>
          <Z>151</Z>
          <R>0</R>
          <G>88</G>
          <B>151</B>
          <A>255</A>
          </ScriptBackgroundColor>
          <ScriptForegroundColor>
          <PackedValue>4294962611</PackedValue>
          <X>179</X>
          <Y>237</Y>
          <Z>255</Z>
          <R>179</R>
          <G>237</G>
          <B>255</B>
          <A>255</A>
          </ScriptForegroundColor>
          <Sprites>
          <Length>0</Length>
          </Sprites>
        </MySerializedTextPanelData>
        <MySerializedTextPanelData>
          <ChangeInterval>0</ChangeInterval>
          <Font Type="MyObjectBuilder_FontDefinition" Subtype="Debug" />
          <FontSize>1</FontSize>
          <ShowText>NONE</ShowText>
          <FontColor>
          <PackedValue>4294967295</PackedValue>
          <X>255</X>
          <Y>255</Y>
          <Z>255</Z>
          <R>255</R>
          <G>255</G>
          <B>255</B>
          <A>255</A>
          </FontColor>
          <BackgroundColor>
          <PackedValue>4278190080</PackedValue>
          <X>0</X>
          <Y>0</Y>
          <Z>0</Z>
          <R>0</R>
          <G>0</G>
          <B>0</B>
          <A>255</A>
          </BackgroundColor>
          <CurrentShownTexture>0</CurrentShownTexture>
          <ContentType>SCRIPT</ContentType>
          <SelectedScript>TSS_EnergyHydrogen</SelectedScript>
          <TextPadding>2</TextPadding>
          <ScriptBackgroundColor>
          <PackedValue>4288108544</PackedValue>
          <X>0</X>
          <Y>88</Y>
          <Z>151</Z>
          <R>0</R>
          <G>88</G>
          <B>151</B>
          <A>255</A>
          </ScriptBackgroundColor>
          <ScriptForegroundColor>
          <PackedValue>4294962611</PackedValue>
          <X>179</X>
          <Y>237</Y>
          <Z>255</Z>
          <R>179</R>
          <G>237</G>
          <B>255</B>
          <A>255</A>
          </ScriptForegroundColor>
          <Sprites>
          <Length>0</Length>
          </Sprites>
        </MySerializedTextPanelData>
        <MySerializedTextPanelData>
          <ChangeInterval>0</ChangeInterval>
          <Font Type="MyObjectBuilder_FontDefinition" Subtype="Debug" />
          <FontSize>1</FontSize>
          <ShowText>NONE</ShowText>
          <FontColor>
          <PackedValue>4294967295</PackedValue>
          <X>255</X>
          <Y>255</Y>
          <Z>255</Z>
          <R>255</R>
          <G>255</G>
          <B>255</B>
          <A>255</A>
          </FontColor>
          <BackgroundColor>
          <PackedValue>4278190080</PackedValue>
          <X>0</X>
          <Y>0</Y>
          <Z>0</Z>
          <R>0</R>
          <G>0</G>
          <B>0</B>
          <A>255</A>
          </BackgroundColor>
          <CurrentShownTexture>0</CurrentShownTexture>
          <ContentType>SCRIPT</ContentType>
          <SelectedScript>TSS_Gravity</SelectedScript>
          <TextPadding>2</TextPadding>
          <ScriptBackgroundColor>
          <PackedValue>4288108544</PackedValue>
          <X>0</X>
          <Y>88</Y>
          <Z>151</Z>
          <R>0</R>
          <G>88</G>
          <B>151</B>
          <A>255</A>
          </ScriptBackgroundColor>
          <ScriptForegroundColor>
          <PackedValue>4294962611</PackedValue>
          <X>179</X>
          <Y>237</Y>
          <Z>255</Z>
          <R>179</R>
          <G>237</G>
          <B>255</B>
          <A>255</A>
          </ScriptForegroundColor>
          <Sprites>
          <Length>0</Length>
          </Sprites>
        </MySerializedTextPanelData>
        <MySerializedTextPanelData>
          <ChangeInterval>0</ChangeInterval>
          <Font Type="MyObjectBuilder_FontDefinition" Subtype="Debug" />
          <FontSize>1</FontSize>
          <ShowText>NONE</ShowText>
          <FontColor>
          <PackedValue>4294967295</PackedValue>
          <X>255</X>
          <Y>255</Y>
          <Z>255</Z>
          <R>255</R>
          <G>255</G>
          <B>255</B>
          <A>255</A>
          </FontColor>
          <BackgroundColor>
          <PackedValue>4278190080</PackedValue>
          <X>0</X>
          <Y>0</Y>
          <Z>0</Z>
          <R>0</R>
          <G>0</G>
          <B>0</B>
          <A>255</A>
          </BackgroundColor>
          <CurrentShownTexture>0</CurrentShownTexture>
          <ContentType>SCRIPT</ContentType>
          <SelectedScript>TSS_ClockAnalog</SelectedScript>
          <TextPadding>2</TextPadding>
          <ScriptBackgroundColor>
          <PackedValue>4288108544</PackedValue>
          <X>0</X>
          <Y>88</Y>
          <Z>151</Z>
          <R>0</R>
          <G>88</G>
          <B>151</B>
          <A>255</A>
          </ScriptBackgroundColor>
          <ScriptForegroundColor>
          <PackedValue>4294962611</PackedValue>
          <X>179</X>
          <Y>237</Y>
          <Z>255</Z>
          <R>179</R>
          <G>237</G>
          <B>255</B>
          <A>255</A>
          </ScriptForegroundColor>
          <Sprites>
          <Length>0</Length>
          </Sprites>
        </MySerializedTextPanelData>
        <MySerializedTextPanelData>
          <ChangeInterval>0</ChangeInterval>
          <Font Type="MyObjectBuilder_FontDefinition" Subtype="Debug" />
          <FontSize>1</FontSize>
          <ShowText>NONE</ShowText>
          <FontColor>
          <PackedValue>4294967295</PackedValue>
          <X>255</X>
          <Y>255</Y>
          <Z>255</Z>
          <R>255</R>
          <G>255</G>
          <B>255</B>
          <A>255</A>
          </FontColor>
          <BackgroundColor>
          <PackedValue>4278190080</PackedValue>
          <X>0</X>
          <Y>0</Y>
          <Z>0</Z>
          <R>0</R>
          <G>0</G>
          <B>0</B>
          <A>255</A>
          </BackgroundColor>
          <CurrentShownTexture>0</CurrentShownTexture>
          <SelectedScript />
          <TextPadding>2</TextPadding>
          <ScriptBackgroundColor>
          <PackedValue>4288108544</PackedValue>
          <X>0</X>
          <Y>88</Y>
          <Z>151</Z>
          <R>0</R>
          <G>88</G>
          <B>151</B>
          <A>255</A>
          </ScriptBackgroundColor>
          <ScriptForegroundColor>
          <PackedValue>4294962611</PackedValue>
          <X>179</X>
          <Y>237</Y>
          <Z>255</Z>
          <R>179</R>
          <G>237</G>
          <B>255</B>
          <A>255</A>
          </ScriptForegroundColor>
          <Sprites>
          <Length>0</Length>
          </Sprites>
        </MySerializedTextPanelData>
        </TextPanels>
        <TargetData>
        <TargetId>0</TargetId>
        <IsTargetLocked>false</IsTargetLocked>
        <LockingProgress>0</LockingProgress>
        </TargetData>
      </MyObjectBuilder_CubeBlock>"""
    
    grid_sizes = {
        1: 'Small',
        2: 'Normal',
        5: 'Large'
    }
    orientations_str = {
        Orientation.FORWARD: 'Forward',
        Orientation.BACKWARD: 'Backward',
        Orientation.RIGHT: 'Right',
        Orientation.LEFT: 'Left',
        Orientation.UP: 'Up',
        Orientation.DOWN: 'Down'
    }
    
    block_xml = {
        'MyObjectBuilder_CubeBlock_LargeBlockArmorCornerInv': armour_blocks,
        'MyObjectBuilder_CubeBlock_LargeBlockArmorCorner': armour_blocks,
        'MyObjectBuilder_CubeBlock_LargeBlockArmorSlope': armour_blocks,
        'MyObjectBuilder_CubeBlock_LargeBlockArmorBlock': armour_blocks,
        'MyObjectBuilder_Gyro_LargeBlockGyro': gyro_block,
        'MyObjectBuilder_Reactor_LargeBlockSmallGenerator': reactor_block,
        'MyObjectBuilder_CargoContainer_LargeBlockSmallContainer': container_block,
        'MyObjectBuilder_Cockpit_OpenCockpitLarge': cockpit_block,
        'MyObjectBuilder_Thrust_LargeBlockSmallThrust': thruster_block,
        'MyObjectBuilder_InteriorLight_SmallLight': light_block,
        'MyObjectBuilder_CubeBlock_Window1x1Slope': armour_blocks,
        'MyObjectBuilder_CubeBlock_Window1x1Flat': armour_blocks,
        'MyObjectBuilder_InteriorLight_LargeBlockLight_1corner': lightcorner_block
    }
    
    header = f"""<?xml version="1.0"?>
    <Definitions xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <ShipBlueprints>
            <ShipBlueprint xsi:type="MyObjectBuilder_ShipBlueprintDefinition">
                <Id Type="MyObjectBuilder_ShipBlueprintDefinition" Subtype="{name.lower().strip()}"/>
                <DisplayName>{name}</DisplayName>
                <CubeGrids>
                    <CubeGrid>
                        <SubtypeName/>
                        <EntityId>{str(random.getrandbits(32)).zfill(16)}</EntityId>
                        <PersistentFlags>CastShadows InScene</PersistentFlags>
                        <PositionAndOrientation>
                            <Position x ="0" y="0" z="0" />
                            <Forward x="0" y="0" z="0" />
                            <Up x="0" y="0" z="0" />
                            <Orientation>
                                <X>0</X>
                                <Y>0</Y>
                                <Z>0</Z>
                                <W>0</W>
                            </Orientation>
                        </PositionAndOrientation>
                        <GridSizeEnum>{grid_sizes[structure.grid_size]}</GridSizeEnum>
                        <CubeBlocks>
                    """
    footer = f"""        </CubeBlocks>
                         <DisplayName>{name.strip()}</DisplayName>
                         <DestructibleBlocks>true</DestructibleBlocks>
                         <IsRespawnGrid>false</IsRespawnGrid>
                         <LocalCoordSys>0</LocalCoordSys>
                         <TargetingTargets />
                     </CubeGrid>
                 </CubeGrids>
                 <WorkshopId>0</WorkshopId>
                 <OwnerSteamId>{builder_id}</OwnerSteamId>
                 <Points>0</Points>
             </ShipBlueprint>
         </ShipBlueprints>
     </Definitions>"""
    
    cube_blocks = ""      
    for block in structure._blocks.values():
        
        if block_xml[block.block_type] is not None:
            cube_blocks += block_xml[block.block_type](block)
    
    return header + cube_blocks + footer