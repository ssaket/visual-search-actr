from multiprocessing import Queue, Process
import numpy as np
import os

class YoloWorker(Process):
    def __init__(self, gpuid, queue, tdset):
        Process.__init__(self, name='ModelProcessor')
        self._gpuid = gpuid
        self._queue = queue
        self._tdset = tdset

    def run(self):

        #set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        #load models
        import yolov3, csv
        yolo = yolov3.yolov3('yolov3.weights')

        print('yoloV3 init done', self._gpuid)

        with open('data/salicon/detected/%s/detected_objects_salicon_test_%s.csv' %(self._tdset, self._gpuid), mode='w', newline='') as image_file:
            image_writer = csv.writer(image_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            while True:
                xfile = self._queue.get()
                if xfile == None:
                    self._queue.put(None)
                    break
                image_data = self.predict(yolo, xfile)
                print('woker', self._gpuid, ' xfile ', xfile)
                image_writer.writerows(image_data)

        print('yoloV3 done ', self._gpuid)

    def predict(self, yolo, imgfile):
        input_w, input_h = 416, 416

        import utils
        photo_filename = imgfile
        # load and prepare image
        image, image_w, image_h = utils.load_image_pixels(photo_filename, (input_w, input_h))
        # make prediction
        yhat = yolo.predict(image)
        # summarize the shape of the list of arrays
        # print([a.shape for a in yhat])
        # define the anchors
        anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        # define the probability threshold for detected objects
        class_threshold = 0.6
        image_data = []
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += utils.decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
        # correct the sizes of the bounding boxes for the shape of the image
        utils.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        # suppress non-maximal boxes
        utils.do_nms(boxes, 0.5)
        # define the labels
        labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        # get the details of the detected objects
        v_boxes, v_labels, v_scores = utils.get_boxes(boxes, labels, class_threshold)
        # summarize what we found
        for i in range(len(v_boxes)):
            image_data.append([ photo_filename.split('\\')[-1], v_labels[i], v_scores[i], v_boxes[i].xmax, v_boxes[i].xmin, v_boxes[i].ymax, v_boxes[i].ymin])
        return image_data
