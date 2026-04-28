!/bin/bash

# ffmpeg -framerate 20 -pattern_type glob -i 'frames/sim_0/*.ppm' -c:v libx264 -pix_fmt yuv420p -y sim_0_20fps.mp4
ffmpeg -framerate 15 -pattern_type glob -i 'frames/mega_epoch_*.png' -c:v libx264 -pix_fmt yuv420p -y mega_simulation_15fps.mp4