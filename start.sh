#For object recognition on the COCO-SEARCH-18
#Uncomment the following lines to generate the cvs files containing the list of objects. Here, gpuids refer to the number of processes to make inference.
#Note: There is a risk of broken process pipe failure or any other failure if you try to run all the commands below at once by uncommenting them. 

# python main.py --imgpath="data/coco_search_18/images/bowl" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="bowl"
# python main.py --imgpath="data/coco_search_18/images/car" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="car"
# python main.py --imgpath="data/coco_search_18/images/chair" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="chair"
# python main.py --imgpath="data/coco_search_18/images/clock" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="clock"
# python main.py --imgpath="data/coco_search_18/images/cup" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="cup"
# python main.py --imgpath="data/coco_search_18/images/fork" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="fork"
# python main.py --imgpath="data/coco_search_18/images/keyboard" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="keyboard"
# python main.py --imgpath="data/coco_search_18/images/knife" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="knife"
# python main.py --imgpath="data/coco_search_18/images/laptop" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="laptop"
# python main.py --imgpath="data/coco_search_18/images/microwave" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="microwave"
# python main.py --imgpath="data/coco_search_18/images/mouse" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="mouse"
# python main.py --imgpath="data/coco_search_18/images/oven" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="oven"
# python main.py --imgpath="data/coco_search_18/images/potted plant" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="plotted plant"
# python main.py --imgpath="data/coco_search_18/images/sink" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="sink"
# python main.py --imgpath="data/coco_search_18/images/stop sign" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="stop sign"
# python main.py --imgpath="data/coco_search_18/images/toilet" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="toilet"
# python main.py --imgpath="data/coco_search_18/images/tv" --gpuids="1,2,3,4,5" --outpath="data/coco_search_18/detected" --target="tv"

# uncomment the following lines to concat the files generated using multiple processes(above) into a single object file.

# python ./concat.py --dir="data/coco_search_18/detected/bowl"
# python ./concat.py --dir="data/coco_search_18/detected/car"
# python ./concat.py --dir="data/coco_search_18/detected/chair"
# python ./concat.py --dir="data/coco_search_18/detected/clock"
# python ./concat.py --dir="data/coco_search_18/detected/cup"
# python ./concat.py --dir="data/coco_search_18/detected/fork"
# python ./concat.py --dir="data/coco_search_18/detected/keyboard"
# python ./concat.py --dir="data/coco_search_18/detected/knife"
# python ./concat.py --dir="data/coco_search_18/detected/laptop"
# python ./concat.py --dir="data/coco_search_18/detected/microwave"
# python ./concat.py --dir="data/coco_search_18/detected/oven"
# python ./concat.py --dir="data/coco_search_18/detected/mouse"
# python ./concat.py --dir="data/coco_search_18/detected/plotted plant"
# python ./concat.py --dir="data/coco_search_18/detected/sink"
# python ./concat.py --dir="data/coco_search_18/detected/stop sign"
# python ./concat.py --dir="data/coco_search_18/detected/toilet"
# python ./concat.py --dir="data/coco_search_18/detected/tv"

#uncomment the following lines to run ACT-R simulations

# python ./multiactrsim.py --path="data/coco_search_18/detected/bowl/concat/detected_objects.csv" --target="bowl" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/car/concat/detected_objects.csv" --target="car" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/chair/concat/detected_objects.csv" --target="chair" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/clock/concat/detected_objects.csv" --target="clock" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/cup/concat/detected_objects.csv" --target="cup" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/fork/concat/detected_objects.csv" --target="fork" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/keyboard/concat/detected_objects.csv" --target="keyboard" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/knife/concat/detected_objects.csv" --target="knife" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/laptop/concat/detected_objects.csv" --target="laptop" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/microwave/concat/detected_objects.csv" --target="microwave" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/oven/concat/detected_objects.csv" --target="oven" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/mouse/concat/detected_objects.csv" --target="mouse" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/plotted plant/concat/detected_objects.csv" --target="potted plant" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/sink/concat/detected_objects.csv" --target="sink" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/stop sign/concat/detected_objects.csv" --target="stop sign" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/toilet/concat/detected_objects.csv" --target="toilet" --subjects="20"
# python ./multiactrsim.py --path="data/coco_search_18/detected/tv/concat/detected_objects.csv" --target="tv" --subjects="20"

# compare all
# python compare.py --dir="data/coco_search_18/simulations"

# compare individualy
# python ./compare.py --dir="data/coco_search_18/simulations" --target="bottle"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="bowl"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="car"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="chair"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="clock"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="cup"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="fork"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="keyboard"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="knife"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="laptop"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="microwave"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="oven"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="mouse"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="potted plant"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="sink"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="stop sign"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="toilet"
# python ./compare.py --dir="data/coco_search_18/simulations" --target="tv"


# For SALICON
# takes 1.8hrs to run
# python main.py --imgpath="data/salicon/images/val" --gpuids="1,2,3,4,5" --outpath="data/salicon/detected/val" 
# takes 2.3hrs to run
# python main.py --imgpath="data/salicon/images/train" --gpuids="1,2,3,4,5" --outpath="data/salicon/detected/train"


# python ./concat.py --dir="data/salicon/detected/train"
# python ./concat.py --dir="data/salicon/detected/val"


# python ./multiactrsim.py --path="data/salicon/detected/val/concat/detected_objects.csv" --subjects="20"
# python ./compare.py --dir="data/salicon/simulations"  --run="salicon"