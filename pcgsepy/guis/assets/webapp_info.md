How to use the program:

### Plots
There are two plots that you can interact with:

* Collection plot: here you can select spaceships based on their characteristics. Only spaceships marked with the "▣" symbol can be selected. The color of the spaceship is determined by its *Fitness*, which is a measure of how "good" the spaceship is. You can zoom using the scrollwheel and pan by keeping the left mouse button pressed and dragging the plot around.
* Spaceship plot: here you can explore a preview of the spaceship you selected. Each sphere in the plot represents a game block and you can tell which block it is simply by hovering over it with the mouse cursor. The legend on the right lists all the game blocks present in the spaceship (you can click once a block type in the legend to hide it from the preview or double click it to show only that block type).

### Properties & Download
Once you select a spacehsip, its properties are displayed in the table on the right. You can also download the currently selected spaceship as a `.zip` file by clicking the **Download** button. The compressed folder contains the files needed to load the spaceship in Space Engineers as a blueprint. Simply place the unzipped folder in `...\AppData\Roaming\SpaceEngineers\Blueprints\local` and load Space Engineers. In a scenario world, press `Ctrl+F10` to bring up the **Blueprints** window and you will see the spaceship listed among the local blueprints.

### Generating new spaceships
To generate new spaceships from the currently selected one, simply press the **Evolve from spaceship** button. A progress bar will appear on the left during the generation and will disappear once the process is completed. The new spaceships are automatically added to the "Collection plot" at the end of the generation.

### Log
All log messages are relayed here. As some operations may take some time to complete, progress messages are also reported.