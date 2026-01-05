!/bin/bash

# ffmpeg -framerate 20 -pattern_type glob -i '/home/robel/energetic-primordial-soup/frames_1767317228/sim_0/*.ppm' -c:v libx264 -pix_fmt yuv420p -y sim_0_20fps.mp4
ffmpeg -framerate 15 -pattern_type glob -i 'frames_1767564394/mega_epoch_*.png' -c:v libx264 -pix_fmt yuv420p -y 1767564394_mega_simulation_15fps.mp4