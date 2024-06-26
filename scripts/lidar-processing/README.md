Step 1: install lastools64 and rstudio. for WSL you can use rstudio-server and port-forward localhost:8787 in your ssh client, otherwise on windows rstudio via rdp. also install cloudcompare for looking at LAS files.

Step 2: check if input las is in meters or longitude/latitude via "description" field of lasinfo64 in.las
if not: ensure input LAS is in meters with: las2las64 -i in.las -wgs84 -longlat -elevation_meter -target_utm 48N -target_meter -target_elevation_meter -o out.las

Step 3: check LAS point density with lasinfo64 in.las -cd . you want it to be ~200 points/m2. if its more like 2000 points/m2: discard 9/10 points w: las2las64 -i in.las -keep_every_nth 10 -o out.las. verify that it looks good by opening the LAS in cloudcompare.

Step 4: cut up the LAS file into chunks with lastile64 -i input.las -tile_size 50 -odir tiles -o tile.las (feel free to adjust the number, its to save memory by not having to load the whole LAS into RAM).

Step 5: in rstudio, install lidR, terra, and ggplot2 packages.

Step 6: use the attached script in rstudio. it will produce heightmaps at 10cm spatial resolution and an RGB raster at m spatial resolution.

Step 7: follow the instructions at https://github.com/manaakiwhenua/pycrown to install pycrown.

Step 8. use the PyCrown_example.ipynb notebook to perform PyCrown segmentation.
