from multiprocessing import Process, Queue
import os
import argparse, time
from yolov3_worker import YoloWorker


class Scheduler:
    def __init__(self, gpuids, tdtype):
        self._queue = Queue()
        self._gpuids = gpuids
        self.tdtype = tdtype

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(YoloWorker(gpuid, self._queue, self.tdtype))


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
    parser.add_argument("--imgpath", help="path to your images to be proceed")
    parser.add_argument("--gpuids",  type=str, help="gpu ids to run" )
    parser.add_argument("--type",  type=str, help="output folder, test | train | val" )

    args = parser.parse_args()

    gpuids = [int(x) for x in args.gpuids.strip().split(',')]

    print(args.imgpath)
    print(gpuids)
    
    if not args.type:
        args.type = "default"

    run(args.imgpath, gpuids, args.type)
    