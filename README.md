# ring-attractor-analysis-tools
Suite of analysis tools for Wang &amp; Compte-style neural network model of WM. WM as persistent activity in a ring attractor. Core ring attractor software is not public.

decode.py is a suite of decoding functions to interpret the shape/center of the bump of WM activation. Included both gaussian and von Mises decoders, as well as tooling to translate from network space to "visual field" space.

run_ring1D_simulations.py is a piece of code written by Nathan and adapted by Sarah to run the ring attractor nsims times.

analyze_ring1D_simulations.py is a piece of code written by Sarah to load & analyze pre-saved simulation outputs.

run_parameter_grid_search.py is a piece of code written by Sarah to trst the robustness of the effects of changing parameters and the stability of the network after these changes in parameters.
