import serpent.cv

from serpent.game_agent import GameAgent
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

from .helpers.ml import *

import offshoot
import tensorflow as tf
import numpy as np

import time

from datetime import datetime


class SerpentCloneyGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None

    def setup_play(self):
        self.plugin_path = offshoot.config["file_paths"]["plugins"]

        context_classifier_path = f"{self.plugin_path}/SerpentCloneyGameAgentPlugin/files/ml_models/cloney_context_classifier.model"

        context_classifier = CNNInceptionV3ContextClassifier(input_shape=(288, 512, 3))
        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

        self.current_run = 0
        self.current_run_started_at = None

        self.last_run_duration = 0

        # Leaves object detection

        self.ml_categories = {1: {'id': 1, 'name': 'leaves'}}
        self.detection_graph = self.load_leaves_model()

    def handle_play(self, game_frame):
        context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)

        if context is None:
            return

        if context == "GAME_WORLD_1":
            self.handle_play_context_game_world(game_frame)
        elif context == "GAME_OVER":
            self.handle_play_context_game_over(game_frame)
        elif context == "MAIN_MENU":
            self.input_controller.click_screen_region(screen_region="MAIN_MENU_PLAY")
            time.sleep(3.5)
            self.current_run_started_at = datetime.utcnow()
        elif context == "GAME_PAUSE":
            self.handle_play_context_game_pause(game_frame)

    def handle_play_context_game_world(self, game_frame):
        # self.input_controller.click(y=288, x=512)
        self.display_game_agent_state(context="GAME_WORLD")
        if len(self.game_frame_buffer.frames) > 1:
            self.detect_leaves(detection_graph=self.detection_graph,category_index=self.ml_categories)

    def handle_play_context_game_over(self, game_frame):
        for i, game_frame in enumerate(self.game_frame_buffer.frames):
            self.visual_debugger.store_image_data(
                game_frame.frame,
                game_frame.frame.shape,
                str(i)
            )
        self.last_run_duration = (datetime.utcnow() - self.current_run_started_at).seconds if self.current_run_started_at else 0

        print("\033c")
        print("GAME: Coney         PLATFORM: Steam\n")
        print("CURRENT CONTEXT: GAME_OVER\n")

        print(f"LAST RUN: {self.current_run}")
        print(f"LAST RUN DURATION: {self.last_run_duration}")

        time.sleep(2)

        # Click PLAY button to start a new run
        self.input_controller.click_screen_region(screen_region="GAME_OVER_PLAY")
        print("\033c")
        time.sleep(0.5)
        print("READY..")
        time.sleep(1)
        print("SET..")
        time.sleep(1)
        print("TAP!")
        time.sleep(0.5)

        # time.sleep(4) # Wait for "Ready, Set, Tap"

        self.current_run += 1
        self.current_run_started_at = datetime.utcnow()

    def handle_play_context_main_menu(self, game_frame):

        # Send the input to start a new run
        self.input_controller.click_screen_region(screen_region="MAIN_MENU_PLAY")

    def handle_play_context_game_pause(self, game_frame):
        time.sleep(2)
        self.input_controller.click_screen_region(screen_region="GAME_PAUSE")

    def display_game_agent_state(self, context):
        print("\033c")
        print(f"GAME: Coney         PLATFORM: Steam\n")
        print(f"CURRENT CONTEXT: {context}\n")
        print(f"CURRENT RUN: {self.current_run}")
        print(f"CURRENT RUN DURATION: {(datetime.utcnow() - self.current_run_started_at)} seconds\n")
        print("")
        print(f"\nLAST RUN: {self.current_run - 1}")
        print(f"LAST RUN DURATION: {self.last_run_duration} seconds")

    def load_leaves_model(self):
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(f'{self.plugin_path}/SerpentCloneyGameAgentPlugin/files/ml_models/leaves_detection/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def detect_leaves(self, detection_graph=None, category_index=None):
        if detection_graph is None:
            return

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                #for i, game_frame in enumerate(self.game_frame_buffer.frames):
                    # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                #image_np = load_image_into_numpy_array(frame)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image = np.expand_dims(self.game_frame_buffer.frames[0].frame, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image})
                # Visualization of the results of a detection.

                for i,b in enumerate(boxes[0]):
                    if classes[0][i] == 1:
                        class_name = category_index[classes[0][i]]['name']
                        if scores[0][i] >= 0.5:
                            ymin = boxes[0][i][0] * 576
                            xmin = boxes[0][i][1] * 1024
                            ymax = boxes[0][i][2] * 576
                            xmax = boxes[0][i][3] * 1024
                            # If x1 > x_of_cloney
                            # If cloney is inside box then x
                            print(f"DETECTED | SCORE { scores[0][i] } | POSITION {ymin}, {xmin}, {ymax}, {xmax}\n")
