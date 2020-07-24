
import time
import zmq
import cv2
import base64
import numpy as np
from object_tracker import ObjectTracker

from absl import app, flags, logging
from absl.flags import FLAGS



flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://*:5555')
footage_socket.setsockopt_string(zmq.SUBSCRIBE, str(''))

socket =  context.socket(zmq.REP)
socket.bind('tcp://*:5556')

def send_array(socket, A, flags=0, copy=True, track=False):
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def main(_argv):

    object_tracker = ObjectTracker()
    frame = None    

    while True:
	    
        frame = footage_socket.recv_string()
        img = base64.b64decode(frame)
        npimg = np.fromstring(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)
       
        message = socket.recv()    

        object_tracker.run(source)
        
        send_array(socket, np.array(object_tracker.get_last_tracked()))

        object_tracker.clear_last_tracked()
    

if __name__ == '__main__':
    try:
        app.run(main)

    except SystemExit:
        pass





