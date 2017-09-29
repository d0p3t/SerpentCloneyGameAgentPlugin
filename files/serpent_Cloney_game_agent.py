import serpent.cv

from serpent.game_agent import GameAgent
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace



from .helpers.ml import *

import offshoot


import numpy as np

import time
import gc
import collections

from datetime import datetime


class SerpentCloneyGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        #self.frame_handlers["PLAY_DDQN"] = self.handle_play_ddqn

        self.frame_handler_setups["PLAY"] = self.setup_play
        #self.frame_handler_setups["PLAY_DDQN"] = self.setup_play_ddqn

        self.analytics_client = None
        #self._reset_game_state()

        #self.game_state = None

    def setup_play(self):
        self.plugin_path = offshoot.config["file_paths"]["plugins"]

        # Context Classifier
        context_classifier_path = f"{self.plugin_path}/SerpentCloneyGameAgentPlugin/files/ml_models/cloney_context_classifier.model"

        context_classifier = CNNInceptionV3ContextClassifier(input_shape=(288, 512, 3))
        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

        # Object Detection of leaves
        self.object_detector = ObjectDetector(graph_fp=f'{self.plugin_path}/SerpentCloneyGameAgentPlugin/files/ml_models/cloney_detection/frozen_inference_graph.pb',
                           labels_fp=f'{self.plugin_path}/SerpentCloneyGameAgentPlugin/files/ml_models/cloney_detection/cloney-detection.pbtxt',
                           num_classes=2,
                           threshold=0.6)

        self.object_predictions = []
        self.warning = ""
        self.dragon_object = []
        self.leaf_object = []

        self.positions = {
            'leaf_pos_mid_y'    : 0,
            'leaf_pos_right_x'  : 0,
            'leaf_pos_left_x'   : 0,
            'leaf_pos_top_y'    : 0,
            'leaf_pos_bottom_y' : 0,
            'dragon_pos_right_x': 0,
            'dragon_pos_left_x' : 0,
            'dragon_pos_mid_y'  : 0,
            'dragon_pos_mid_x'  : 0
        }

        # Variables for Display
        self.current_run = 0
        self.current_run_started_at = datetime.utcnow()
        self.current_run_duration = 0

        self.last_run_duration = 0
        self.last_run = 0

    def setup_play_ddqn(self):

        input_mapping = {
            "U": [self.input_controller.tap_key(key="u", duration=0.1)]
        }

        action_space = KeyboardMouseActionSpace(
            directional_keys=[None, "U"]
        )

        model_file_path = None

        self.dqn_movement = DDQN(
            model_file_path=model_file_path if os.path.isfile(model_file_path) else None,
            input_shape=(72, 128, 4),
            input_mapping=input_mapping,
            action_space=movement_action_space,
            replay_memory_size=50000,
            max_steps=500000,
            observe_steps=10000,
            batch_size=64,
            initial_epsilon=0.1,
            final_epsilon= 0.0001,
            override_epsilon=False
            )

    def handle_play_ddqn(self, game_frame):
        gc.disable()

        context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)

        if self.dqn_movement.first_run:
            if context == "GAME_OVER":
                self.input_controller.click_screen_region(screen_region="GAME_OVER_PLAY")

            self.dqn_movement.first_run = False

            return None

    def handle_play(self, game_frame):
        context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)

        if context is None:
            return

        if context == "GAME_WORLD_1":
            self.handle_play_context_game_world(game_frame=game_frame, context=context)
        elif context == "GAME_OVER":
            self.handle_play_context_game_over(game_frame=game_frame, context=context)
        elif context == "MAIN_MENU":
            self.input_controller.click_screen_region(screen_region="MAIN_MENU_PLAY")
            time.sleep(3.5)
            self.current_run_started_at = datetime.utcnow()
        elif context == "GAME_PAUSE":
            self.handle_play_context_game_pause(game_frame)



    def handle_play_context_game_world(self, game_frame, context):

        # Object detection
        with tf.device('/gpu:0'):
            self.object_predictions = self.object_detector.predict(frame=game_frame.frame)

            for prediction in self.object_predictions:
                if prediction['class'] == "dragon":
                    self.positions['dragon_pos_right_x'] = prediction['bb_o'][3]
                    self.positions['dragon_pos_left_x'] = prediction['bb_o'][1]
                    self.positions['dragon_pos_mid_y'] = (prediction['bb_o'][0] + prediction['bb_o'][2]) / 2
                    self.positions['dragon_pos_mid_x'] = (prediction['bb_o'][1] + prediction['bb_o'][3]) / 2
                    self.dragon_object = prediction
                elif prediction['class'] == "leaves":
                    self.positions['leaf_pos_mid_y'] = (prediction['bb_o'][0] + prediction['bb_o'][2]) / 2
                    self.positions['leaf_pos_top_y'] = prediction['bb_o'][0]
                    self.positions['leaf_pos_bottom_y'] = prediction['bb_o'][2]
                    self.positions['leaf_pos_right_x'] = prediction['bb_o'][3]
                    self.positions['leaf_pos_left_x']= prediction['bb_o'][1]
                    self.leaf_object = prediction

                if (self.positions['dragon_pos_mid_y'] > (self.positions['leaf_pos_top_y'] - 25) and self.positions['dragon_pos_mid_y'] < (self.positions['leaf_pos_bottom_y']) + 25) and (self.positions['dragon_pos_right_x'] + 100) > self.positions['leaf_pos_left_x']: # Same height
                    self.warning = "WARNING"
                    # self.input_controller.tap_key(key="s")
                    # self.input_controller.tap_key(key="a")
                    break
                else:
                    self.warning = "SAFE"
                    # self.input_controller.tap_key(key="s")
                    # time.sleep(0.3)

        self.display_game_agent_state(context=context)

    def handle_play_context_game_over(self, game_frame, context):
        self.display_game_agent_state(context=context)

        self.last_run_duration = (datetime.utcnow() - self.current_run_started_at).seconds if self.current_run_started_at else 0
        self.last_run = self.current_run

        time.sleep(2)

        # Click PLAY button to start a new run
        self.input_controller.click_screen_region(screen_region="GAME_OVER_PLAY")

        time.sleep(3) # Wait for "Ready, Set, Tap"


        self.current_run += 1
        self.current_run_started_at = datetime.utcnow()

    def handle_play_context_main_menu(self, game_frame):

        # Send the input to start a new run
        self.input_controller.click_screen_region(screen_region="MAIN_MENU_PLAY")

    def handle_play_context_game_pause(self, game_frame):
        time.sleep(2)
        self.input_controller.click_screen_region(screen_region="GAME_PAUSE")

    def display_game_agent_state(self, context):
        self.current_run_duration = (datetime.utcnow() - self.current_run_started_at).seconds

        print("\033c")
        print(f"GAME: Cloney         PLATFORM: Steam\n")

        print("")

        print(f"CURRENT CONTEXT: {context}")

        print("")

        print(f"CURRENT RUN: {self.current_run}")
        print(f"CURRENT RUN DURATION: {self.current_run_duration} seconds")

        print("")

        print(f"LAST RUN: {self.last_run}")
        print(f"LAST RUN DURATION: {self.last_run_duration} seconds")

        print("")

        print(f"COLLISSION: {self.warning}")
        print(f"DRAGON: {self.dragon_object}")
        print(f"LEAF: {self.leaf_object}")
        #print(f"LEAF PROBABILITY: {round(leaf_score, 5) * 100}%")

    def _reset_game_state(self):
        self.game_state = {
            "seed_entered": False,
            "health": collections.deque(np.full((8,), 6), maxlen=8),
            "coins": 0,
            "game_context": None,
            "current_run": 1,
            "current_run_steps": 0,
            "average_aps": 0,
            "run_reward_movement": 0,
            "run_reward_projectile": 0,
            "run_future_rewards": 0,
            "run_predicted_actions": 0,
            "run_timestamp": datetime.utcnow(),
            "last_run_duration": 0,
            "record_time_alive": dict(),
            "record_distance_travelled": dict(),
            "record_coins_collected": dict(),
            "random_time_alive": None,
            "random_time_alives": list(),
            "random_distance_travelled": None,
            "random_boss_hps": list()
            }
