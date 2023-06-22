# CD_plasticity
Codes to simulate, explore and graph model built during M1 project in Montpellier

## Single run
<<single_run.py>> file can be run with desirable paramaters to observe evolution of a single run.

## Parameter exploration
All files begining with <<Exploration>> in their file name deal with paramater exploration. The pipeline used is that of Pypet (dccumentation: https://pypet.readthedocs.io/en/latest/)
The following steps were used:
1) Establishing paramaters. Usually done by some function of the form "add_param" or something similar. Defines default values of the paramaters and their names
2) Establishing exploration. This function adds paramaters and their range of values to be explored while others stay constant.
3) Adding main simulation. These functions are of the type "run_main" or some other variation. Multiple variants of this function are present. The ones used commonly are defined in helper_exploration, and the ones that had tiny modifications are present under their respective exploration file.
4) Running main simulation with parallel processing and storing the results
5) Post processing: Parsing results to a more suitable file such as a csv file instead of hdf5. Denoted by functions of the form "post_proc"

## Graphing
the previous results are then graphed with the files starting witt "graphing_"
