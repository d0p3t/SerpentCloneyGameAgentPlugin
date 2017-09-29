import numpy as np
import tensorflow as tf

import os.path
import sys

sys.path.append(os.path.dirname(__file__))

import label_map_util


# OBJECT DETECTION - EXAMPLE RETURN DATA
# ==========================================
# score = Probability the detection is an object
# bb = bounding box coordinates in %
# bb_o = bounding box coordinates of corners (y1, x1, y2, x2)
# class = object label
#
# [{'score': 0.99999976,
# 'bb': array([ 0.23609778,  0.3672961 ,  0.38386199,  0.61785924], dtype=float32),
# 'bb_o': [135, 376, 221, 632],
# 'class': 'leaves'},
# {'score': 0.99999976,
# 'bb': array([ 0.24311924,  0.6509043 ,  0.48661894,  0.79301977], dtype=float32),
# 'bb_o': [140, 666, 280, 812],
# 'class': 'leaves'},
# {'score': 0.99985301,
# 'bb': array([ 0.03044974,  0.20537947,  0.27158079,  0.35302216], dtype=float32),
# 'bb_o': [17, 210, 156, 361],
# 'class': 'leaves'}]
# ==============================================



class ObjectDetector:
    def __init__(self, graph_fp, labels_fp, num_classes=1, threshold=0.6):
        self.graph_fp = graph_fp
        self.labels_fp = labels_fp
        self.num_classes = num_classes

        self.graph = None
        self.label_map = None
        self.categories = None
        self.category_index = None

        self.bb = None
        self.bb_origin = None
        self.image_tensor = None
        self.boxes = None
        self.scores = None
        self.classes = None
        self.num_detections = None

        self.in_progress = False
        self.session = None
        self.threshold = threshold
        with tf.device('/gpu:0'):
            self._load_graph()
            self._load_labels()
            self._init_predictor()

    def _load_labels(self):
        self.label_map = label_map_util.load_labelmap(self.labels_fp)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def _load_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_fp, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                tf.get_default_graph().finalize()

    def _init_predictor(self):
        tf_config = tf.ConfigProto(device_count={'gpu': 0}, log_device_placement=False)
        tf_config.gpu_options.allow_growth = True
        with self.graph.as_default():
            self.session = tf.Session(config=tf_config, graph=self.graph)
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    def predict(self, frame):
        self.in_progress = True

        with self.graph.as_default():
            height, width, _ = frame.shape

            #frame_resized = frame.reshape((height, width, 3)).astype(np.uint8)

            frame_np_expanded = np.expand_dims(frame, axis=0)

            (boxes, scores, classes, num_detections) = self.session.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={
                    self.image_tensor: frame_np_expanded
                })

            filtered_results = []
            for i in range(0, num_detections[0].astype(np.uint8)):
                score = scores[0][i]
                if score >= self.threshold:
                    y1, x1, y2, x2 = boxes[0][i]
                    y1_o = int(y1 * height)
                    x1_o = int(x1 * width)
                    y2_o = int(y2 * height)
                    x2_o = int(x2 * width)
                    predicted_class = self.category_index[classes[0][i]]['name']
                    filtered_results.append({
                        "score": score,
                        "bb": boxes[0][i],
                        "bb_o": [y1_o, x1_o, y2_o, x2_o],
                        "class": predicted_class
                    })

            return filtered_results

        self.in_progress = False

    def get_status(self):
        return self.in_progress

    def kill_predictor(self):
        self.session.close()
        self.session = None
