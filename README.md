The code here is to replicate the study:
Evolvable robots: understanding how the need to locomote defines robot bodies - targeted locomotion

Here includes the targeted locomotion scenarios: normal, slippery, and sticky.

It is based on **plasticoding_v2** branch of Revolve2.
Installation and more information can be found:
https://github.com/karinemiras/revolve2/tree/plasticoding_v2

Directed locomotion can use the branch listed above, speed_y would be the fitness in this case, with 150 generations, population size of 100, offspring size of 50, parent selection tournament size of 3, survivor selection tournament size of 5, and simulation time of 30, in accordance with the targeted locomotion scenarios.

Changed files include: _cpg.py, _measure.py, geometry.py, environment_actor_controller.py, _targetObject.py, _brain_cpg_network_neighbour.py, _local_runner.py, and most files in target_study folder.

Changing scenarios to slippery or sticky can be done in optimizer.py

