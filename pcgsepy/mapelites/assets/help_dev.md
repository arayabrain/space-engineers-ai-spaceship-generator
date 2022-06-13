How to use the program:
###### Plot settings
Here it is possible to change the data that is being visualized in the MAP-Elites plot (left) and content plot (right).

It is possible to visualize behavior characteristics for both the feasible and infeasible population. It is possible to plot either the Fitness, the Age, and the Coverage metric for all solutions. Additionally, it is possible to plot metrics for only the elite or the bin population.

By selecting a bin in the MAP-Elites plot, the elite content will be displayed in the content plot (right) and the content properties (spaceship size, number of blocks used, and string representation) will be shown next to it. Additionally, the spaceship(s) can be downloaded by clicking the "DOWNLOAD CONTENT" button.

###### Experiment settings & control
Here there are both information on the experiment and buttons that allow the user to interact with the evolution procedure.

In particular:
- **Experiment settings** shows the bins that are valid for evolution, the current generation, and the bins selected by the user. There are also different controls that the user can modify:
    - **Choose feature descriptors (X, Y)**: here you can select which behaviour characteristics to use in MAP-Elites.
    - **Toggle L-system modules**: here you can select which parts of the spaceship are allowed to mutate.
    - **Control fitness weights**: here you can choose how much impact each metric has on the overall fitness. Slider values go from 0.0 to 1.0, and the MAP-Elites preview is updated accordingly.
    - **Select emitter**: here you can select which emitter to use during the experiment. Note that changing the emitter will create a new one, so all emitter data collected thus far will be lost!

- **Experiment controls** is comprised of different buttons:
    - **APPLY STEP**: Executes a step of FI-2Pop with the selected bin(s) populations. If no bin is selected, or if the selected bin(s) is invalid, an error is thrown and no step is executed.
    - **INITIALIZE/RESET**: Either initializes the entire population of solutions (if empty) or resets it.
    - **CLEAR SELECTION**: Clears the selected bin(s).
    - **TOGGLE BIN SELECTION**: Toggles evolution on a single bin or multiple bins. If toggled to false and more than one bin were selected, only the last bin will remain selected.
    - **SUBDIVIDE SELECTED BIN(S)**: Subdivides the selected bin(s) in half, reassigning the solutions to the correct bin.
    - **DOWNLOAD MAP-ELITES**: Downloads the MAP-Elites object. This is only possible after a certain number of generations has elapsed.

###### High-level rules
Here it is possible to inspect and update the high-level rules used by the L-system. When updating a rule, a check is always performed to ensure the expansion probability of the left-hand side rule sums up to 1.

###### Log
All log messages are relayed here. As some operations may take some time to complete, progress messages are also reported.