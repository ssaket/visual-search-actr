from multiprocessing import Process, Queue
import os
import argparse, time
from yolov3_worker import YoloWorker


class Scheduler:
    def __init__(self, gpuids, outpath):
        self._queue = Queue()
        self._gpuids = gpuids
        self._outpath = outpath

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(YoloWorker(gpuid, self._queue, self._outpath))


    def start(self, xfilelst):

        # put all of files into queue
        for xfile in xfilelst:
            self._queue.put(xfile)

        #add a None into queue to indicate the end of task
        self._queue.put(None)

        #start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        print("all of workers have been done")

                

def run(img_path, gpuids, tdtype):
    #scan all files under img_path
    xlist = list()
    for xfile in os.listdir(img_path):
        xlist.append(os.path.join(img_path, xfile))
    print("--- total files to process %s" %(len(xlist)))
    
    #init scheduler
    x = Scheduler(gpuids, tdtype)
    
    #start processing and wait for complete
    start_time = time.time() 
    x.start(xlist)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", help="path to images to be proceed")
    parser.add_argument("--gpuids",  type=str, help="gpu ids to run" )
    parser.add_argument("--outpath",  type=str, help="path to output folder" )
    parser.add_argument("--target",  type=str, help="target for coco-search-18" )

    args = parser.parse_args()

    gpuids = [int(x) for x in args.gpuids.strip().split(',')]

    print(args.imgpath)
    print(gpuids)
    
    if not args.outpath:
        args.outpath = args.imgpath
    if args.target:
        args.outpath = os.path.join(args.outpath, args.target)
        if not os.path.isdir(args.outpath):
            os.makedirs(args.outpath)

    run(args.imgpath, gpuids, args.outpath)
    