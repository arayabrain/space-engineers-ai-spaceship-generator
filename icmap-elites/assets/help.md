How to use the program:
###### Plot settings
Here it is possible to change the data that is being visualized in the MAP-Elites plot (left) and content plot (right).

It is possible to visualize behavior characteristics for both the feasible and infeasible population. It is possible to plot either the Fitness, the Age, and the Coverage metric for all solutions. Additionally, it is possible to plot metrics for only the elitist or the bin population.

By selecting a bin in the MAP-Elites plot, the elitist content will be displayed in the content plot (right) and the content string will be printed in the box (bottom-right).

###### Experiment settings & control
Here there are both information on the experiment and buttons that allow the user to interact with the evolution procedure.

In particular:
* Experiment settings shows the bins that are valid for evolution, the current generation, and the bins selected by the user.
* Experiment controls is comprised of different buttons:
    - "Initialize/Reset": Either initializes the entire population of solutions (if empty) or resets it.
    - "Toggle single bin selection": This button allows to toggle evolution on a single bin or multiple bins. If toggled to false and more than one bin were selected, only the last bin will remain selected.
    - "Clear selection": Removes the selections entirely.
    - "Apply step": Applies an evolution step (with FI-2Pop) on the selected bin(s).

###### Log
All log messages are relayed here. As some operations may take some time to complete, progress messages are also reported.