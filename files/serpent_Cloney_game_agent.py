import serpent.cv
import serpent.ocr as ocr
import serpent.utilities

from serpent.game_agent import GameAgent

from serpent.frame_grabber import FrameGrabber

from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace

from serpent.input_controller import KeyboardKey

from .helpers.ml import ObjectDetector

from datetime import datetime

import offshoot
import xtermcolor
import skimage
import numpy as np

import time
import gc
import os

import collections

class SerpentCloneyGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handlers["PLAY_DDQN"] = self.handle_play_ddqn

        # self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_setups["PLAY_DDQN"] = self.setup_play_ddqn

        self.analytics_client = None

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

        # Reset Variables
        self._reset_game_state()

    # =============================
    # -----------DQN TODO ---------
    # =============================
    def setup_play_ddqn(self):

        self._reset_game_state()

        input_mapping = {
            "UP": [KeyboardKey.KEY_SPACE]
        }

        self.key_mapping = {
            KeyboardKey.KEY_SPACE.name: "UP"
        }

        movement_action_space = KeyboardMouseActionSpace(
            default_keys=[None, "UP"]
        )

        movement_model_file_path = "datasets/cloney_direction_dqn_0_1_.hp5".replace("/", os.sep)

        self.dqn_movement = DDQN(
            model_file_path=movement_model_file_path if os.path.isfile(movement_model_file_path) else None,
            input_shape=(100, 100, 4),
            input_mapping=input_mapping,
            action_space=movement_action_space,
            replay_memory_size=5000,
            max_steps=1000000,
            observe_steps=1000,
            batch_size=32,
            model_learning_rate=1e-4,
            initial_epsilon=1,
            final_epsilon=0.01,
            override_epsilon=False
            )

    def handle_play_ddqn(self, game_frame):
        gc.disable()

        if self.dqn_movement.first_run:
            self.input_controller.tap_key(KeyboardKey.KEY_W)

            self.dqn_movement.first_run = False

            time.sleep(5)

            return None

        dragon_alive = self._measure_dragon_alive(game_frame)
        # dragon_coins = self._measure_dragon_coins(game_frame)

        self.game_state["alive"].appendleft(dragon_alive)
        # self.game_state["coins"].appendleft(dragon_coins)

        if self.dqn_movement.frame_stack is None:
            # pipeline_game_frame = FrameGrabber.get_frames(
            #     [0],
            #     frame_shape=game_frame.frame.shape,
            #     frame_type="MINI"
            # ).frames[0]

            self.dqn_movement.build_frame_stack(game_frame.ssim_frame)
        else:
            game_frame_buffer = FrameGrabber.get_frames(
                [0, 4, 8, 12],
                frame_shape=game_frame.frame.shape,
                frame_type="MINI"
                )

            if self.dqn_movement.mode == "TRAIN":
                reward = self._calculate_reward()

                self.game_state["run_reward"] += reward

                self.dqn_movement.append_to_replay_memory(
                    game_frame_buffer,
                    reward,
                    terminal=self.game_state["alive"] == 0
                )
                # Every 2000 steps, save latest weights to disk
                if self.dqn_movement.current_step % 2000 == 0:
                    self.dqn_movement.save_model_weights(
                        file_path_prefix=f"datasets/cloney_movement"
                    )

                # Every 20000 steps, save weights checkpoint to disk
                if self.dqn_movement.current_step % 20000 == 0:
                    self.dqn_movement.save_model_weights(
                        file_path_prefix=f"datasets/cloney_movement",
                        is_checkpoint=True
                    )

            elif self.dqn_movement.mode == "RUN":
                self.dqn_movement.update_frame_stack(self.game_frame_buffer)

            run_time = datetime.now() - self.started_at

            print("\033c" + f"SESSION RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours, {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} seconds")
            print("")

            print("MOVEMENT NEURAL NETWORK:\n")
            self.dqn_movement.output_step_data()

            print("")
            print(f"CURRENT RUN: {self.game_state['current_run']}")
            print(f"CURRENT RUN REWARD: {round(self.game_state['run_reward'], 2)}")
            print(f"CURRENT RUN PREDICTED ACTIONS: {self.game_state['run_predicted_actions']}")
            print(f"CURRENT DRAGON ALIVE: {self.game_state['alive'][0]}")
            # print(f"CURRENT DRAGON COINS: {self.game_state['coins'][0]})

            print("")
            # print(f"AVERAGE ACTIONS PER SECOND: {round(self.game_state['average_aps'], 2)}")
            print("")
            print(f"LAST RUN DURATION: {self.game_state['last_run_duration']} seconds")
            # print(f"LAST RUN COINS: {self.game_state['last_run_coins'][0]})

            print("")
            print(f"RECORD TIME ALIVE: {self.game_state['record_time_alive'].get('value')} seconds (Run {self.game_state['record_time_alive'].get('run')}, {'Predicted' if self.game_state['record_time_alive'].get('predicted') else 'Training'})")
            # print(f"RECORD COINS COLLECTED: {self.game_state['record_coins_collected'].get('value')} coins (Run {self.game_state['record_coins_collected'].get('run')}, {'Predicted' if self.game_state['record_coins_collected'].get('predicted') else 'Training'})")
            print("")
            print(f"RANDOM AVERAGE TIME ALIVE: {self.game_state['random_time_alive']} seconds")

            if self.game_state["alive"][1] <= 0:
                serpent.utilities.clear_terminal()
                timestamp = datetime.utcnow()

                gc.enable()
                gc.collect()
                gc.disable()

                # Set display stuff TODO
                timestamp_delta = timestamp - self.game_state["run_timestamp"]
                self.game_state["last_run_duration"] = timestamp_delta.seconds

                if self.dqn_movement.mode in ["TRAIN", "RUN"]:
                    # Check for Records
                    if self.game_state["last_run_duration"] > self.game_state["record_time_alive"].get("value", 0):
                        self.game_state["record_time_alive"] = {
                            "value": self.game_state["last_run_duration"],
                            "run": self.game_state["current_run"],
                            "predicted": self.dqn_movement.mode == "RUN"
                        }

                    # if self.game_state["coins"][0] < self.game_state["record_coins_collected"].get("value", 1000):
                    #     self.game_state["record_coins_collected"] = {
                    #         "value": self.game_state["coins"][0],
                    #         "run": self.game_state["current_run"],
                    #         "predicted": self.dqn_movement.mode == "RUN"
                    #     }
                else:
                    self.game_state["random_time_alives"].append(self.game_state["last_run_duration"])
                    self.game_state["random_time_alive"] = np.mean(self.game_state["random_time_alives"])

                self.game_state["current_run_steps"] = 0

                self.input_controller.release_key(KeyboardKey.KEY_SPACE)

                if self.dqn_movement.mode == "TRAIN":
                    for i in range(8):
                        serpent.utilities.clear_terminal()
                        print(f"TRAINING ON MINI-BATCHES: {i + 1}/8")
                        print(f"NEXT RUN: {self.game_state['current_run'] + 1} {'- AI RUN' if (self.game_state['current_run'] + 1) % 20 == 0 else ''}")

                        self.dqn_movement.train_on_mini_batch()

                self.game_state["run_timestamp"] = datetime.utcnow()
                self.game_state["current_run"] += 1
                self.game_state["run_reward_movement"] = 0
                self.game_state["run_predicted_actions"] = 0
                self.game_state["alive"] = collections.deque(np.full((8,), 4), maxlen=8)
                # self.game_state["coins"] = collections.deque(np.full((8,), 0), maxlen=8)

                if self.dqn_movement.mode in ["TRAIN", "RUN"]:
                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 100 == 0:
                        if self.dqn_movement.type == "DDQN":
                            self.dqn_movement.update_target_model()

                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 20 == 0:
                        self.dqn_movement.enter_run_mode()
                    else:
                        self.dqn_movement.enter_train_mode()

                self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
                time.sleep(5)

                return None

        self.dqn_movement.pick_action()
        self.dqn_movement.generate_action()

        keys = self.dqn_movement.get_input_values()
        print("")
        print(" + ".join(list(map(lambda k: self.key_mapping.get(k.name), keys))))

        self.input_controller.handle_keys(keys)

        if self.dqn_movement.current_action_type == "PREDICTED":
            self.game_state["run_predicted_actions"] += 1

        self.dqn_movement.erode_epsilon(factor=2)

        self.dqn_movement.next_step()

        self.game_state["current_run_steps"] += 1

    def handle_play(self, game_frame):
        context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)

        if context is None:
            return

        if context == "GAME_WORLD_1":
            self.display_game_agent_state(context=context)
            self.handle_play_context_game_world(game_frame=game_frame)
            self.in_progress_game_over = False
        elif context == "GAME_OVER":
            self.display_game_agent_state(context=context)
            time.sleep(2)
            if self.in_progress_game_over is False:
                self.handle_play_context_game_over(game_frame=game_frame)
        elif context == "MAIN_MENU":
            self.input_controller.click_screen_region(screen_region="MAIN_MENU_PLAY")
            time.sleep(3.5)
            self.current_run_started_at = datetime.utcnow()
        elif context == "GAME_PAUSE":
            self.handle_play_context_game_pause(game_frame)

    def handle_play_context_game_world(self, game_frame):
        # Only predict if object_detector is idle
        if self.object_detector.get_status() is False:
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

                if (self.positions['dragon_pos_mid_y'] > (self.positions['leaf_pos_top_y'] - 50) and self.positions['dragon_pos_mid_y'] < (self.positions['leaf_pos_bottom_y']) + 50) and (self.positions['dragon_pos_right_x'] + 100) > self.positions['leaf_pos_left_x']: # Same height
                    self.warning = "HIGH"
                    if self.positions['dragon_pos_right_x'] + 100 > self.positions['leaf_pos_left_x']:
                        self.input_controller.tap_key(KeyboardKey.KEY_S, duration=0.025)
                        time.sleep(0.1)
                    elif self.positions['dragon_pos_mid_y'] - 50 < self.positions['leaf_pos_bottom_y']:
                        time.sleep(0.225)
                        self.input_controller.tap_key(KeyboardKey.KEY_S, duration=0.025)
                    elif self.positions['dragon_pos_mid_y'] + 50 > self.positions['leaf_pos_top_y']:
                        self.input_controller.tap_key(KeyboardKey.KEY_S, duration=0.025)
                        time.sleep(0.1)
                    break
                else:
                    self.warning = "SAFE"
                    self.input_controller.tap_key(KeyboardKey.KEY_S, duration=0.026)
                    time.sleep(0.23)
                    break

    def handle_play_context_game_over(self, game_frame):
            self.in_progress_game_over = True

            time.sleep(4)

            self.game_state['last_run_duration'] = (datetime.utcnow() - self.game_state['current_run_started_at']).seconds if self.game_state['current_run_started_at'] else 0
            self.game_state['last_run'] = self.game_state['current_run']

            if self.game_state['record_duration'] is not None:
                if self.game_state['last_run_duration'] > self.game_state['record_duration']:
                    self.game_state['record_duration'] = self.game_state['last_run_duration']
                    self.game_state['record_run'] = self.game_state['last_run']
            else:
                self.game_state['record_duration'] = self.game_state['last_run_duration']

            # Process Image for OCR
            frame = game_frame.frame
            gray_frame = skimage.color.rgb2gray(frame)
            frame_coins = gray_frame[190:300, 250: 780]
            frame_distance = gray_frame[355:410, 550:760]
            frame_time = gray_frame[300:355, 550:760]

            # Find Coins
            text_coins = ocr.perform_ocr(image=frame_coins, scale=2, order=5, horizontal_closing=2, vertical_closing=3)

            # Find Distance
            text_distance = ocr.perform_ocr(image=frame_distance, scale=2, order=5, horizontal_closing=2, vertical_closing=3)
            text_time = ocr.perform_ocr(image=frame_time, scale=2, order=5, horizontal_closing=2, vertical_closing=3)
            print(text_coins)
            print(text_time)
            print(text_distance)

            # if "$" in coins:
            #     num_coins = coins.replace('$', '')
            #     self.game_state['last_run_coins_collected'] = int(num_coins)
            #
            # if self.game_state['last_run_coins_collected'] > self.game_state['record_coins_collected']:
            #     self.game_state['record_coins_collected'] = self.game_state['last_run_coins_collected']

            # Find Distance and Time
            #candidates, regions = ocr.extract_ocr_candidates(image=frame, gradient_size=3, closing_size=10, minimum_area=100, minimum_aspect_ratio=2)

            #print(regions)

            #gray_frame = skimage.color.rgb2gray(frame)

            # for region in regions:
            #     crop = gray_frame[region[0]:region[2], region[1]:region[3]]
            #     read = ocr.perform_ocr(image=crop, scale=1, order=5, horizontal_closing=1, vertical_closing=1)
            #     print(read)
            #     if "Distance" in read or "Time":
            #         self.pos_d = regions.index(region) + 1
            #     elif "Time" in read:
            #         self.pos_t = regions.index(region) + 1
            #
            #     if regions.index(region) == self.pos_d:
            #         self.game_state['last_run_distance'] = read.replace('m', '')
            #         if self.game_state['last_run_distance'] > self.game_state['record_distance']:
            #             self.game_state['record_distance'] = self.game_state['last_run_distance']
            #     elif regions.index(region) == self.pos_t:
            #         self.game_state['last_run_duration_actual'] = read
                    # Have to still check for record. Find out about time formatting

            # Click PLAY button to start a new run
            #self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
            self.input_controller.click_screen_region(screen_region="GAME_OVER_PLAY")

            # Wait for "Ready, Set, Tap"
            time.sleep(3)
            self.input_controller.tap_key(KeyboardKey.KEY_S)
            time.sleep(0.2)

            self.game_state['current_run'] += 1
            self.game_state['current_run_started_at'] = datetime.utcnow()

    def handle_play_context_main_menu(self, game_frame):
        self.input_controller.click_screen_region(screen_region="MAIN_MENU_PLAY")

    def handle_play_context_game_pause(self, game_frame):
        time.sleep(1)
        self.input_controller.click_screen_region(screen_region="GAME_PAUSE")

    def display_game_agent_state(self, context):
        self.game_state['current_run_duration'] = (datetime.utcnow() - self.game_state['current_run_started_at']).seconds

        print("\033c")
        print("======================================================")
        print(f"GAME: Cloney    PLATFORM: Steam    VERSION: v0.0.1")
        print("======================================================")

        print("")

        print(xtermcolor.colorize("OBJECT DETECTION", ansi=9))
        print(f"Detected:                   {len(self.object_predictions)} objects")
        if self.warning == "HIGH":
            print(xtermcolor.colorize(f"Danger Level:               {self.warning}", ansi=1))
        elif self.warning == "SAFE":
            print(xtermcolor.colorize(f"Danger Level:               {self.warning}", ansi=2))
        # print(f"DRAGON POS: {self.dragon_object['bb_o'][0]}, {self.dragon_object['bb_o'][1]}, {self.dragon_object['bb_o'][2]}, {self.dragon_object['bb_o'][3]}")
        # print(f"LAST LEAF POS: {self.leaf_object['bb_o'][0]}, {self.leaf_object['bb_o'][1]}. {self.leaf_object['bb_o'][2]}. {self.leaf_object['bb_o'][3]}")

        print("")

        print(xtermcolor.colorize("GAME STATISTICS", ansi=9))
        print(f"Current Context:            {context}\n")
        print(f"Current Run:                #{self.game_state['current_run']}")
        print(f"Current Run Duration:       {self.game_state['current_run_duration']}s")
        print("")
        print(f"Last Run:                   #{self.game_state['last_run']}")
        print(f"Last Run Duration:          {self.game_state['last_run_duration']}s")
        print(f"Last Run Duration Actual:   {self.game_state['last_run_duration_actual']}")
        print(f"Last Run Distance:          {self.game_state['last_run_distance']}m")
        print(f"Last Run Coins Collected:   {self.game_state['last_run_coins_collected']}")
        print(f"Record Duration:            {self.game_state['record_duration']}s (Run #{self.game_state['record_run']})")

    def _reset_game_state(self):
        # Display Variables
        self.game_state = {
            "alive": collections.deque(np.full((8,), 4), maxlen=8),
            "coins": collections.deque(np.full((8,), 0), maxlen=8),
            "current_run": 1,
            "current_run_started_at": datetime.utcnow(),
            "current_run_duration": None,
            "current_run_steps": 0,
            "run_reward": 0,
            "run_future_rewards": 0,
            "run_predicted_actions": 0,
            "run_timestamp": datetime.utcnow(),
            "last_run": 0,
            "last_run_duration": 0,
            "last_run_duration_actual": None,
            "last_run_distance": 0.0,
            "last_run_coins_collected": 0,
            "record_duration": None,
            "record_duration_actual": 0,
            "record_run": 0,
            "record_distance": 0.0,
            "record_coins_collected": 0,
            "record_time_alive": dict(),
            "random_time_alive": None,
            "random_time_alives": list(),
            "random_distance_travelled": 0.0
            }

        # Object Detection Variables
        self.object_predictions = []
        self.warning = ""
        self.dragon_object = []
        self.leaf_object = []
        self.positions = {
            'leaf_pos_mid_y': 0,
            'leaf_pos_right_x': 0,
            'leaf_pos_left_x': 0,
            'leaf_pos_top_y': 0,
            'leaf_pos_bottom_y': 0,
            'dragon_pos_right_x': 0,
            'dragon_pos_left_x': 0,
            'dragon_pos_mid_y': 0,
            'dragon_pos_mid_x': 0
        }

        # Other Variables
        self.pos_d = -1
        self.pos_t = -1
        self.in_progress_game_over = False
    def _measure_dragon_alive(self, game_frame):
        dollar_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["DOLLAR_AREA"])

        dragon_alive = None
        max_ssim = 0

        for name, sprite in self.game.sprites.items():
            print(name)
            print(name[-1])
            for i in range(sprite.image_data.shape[3]):
                ssim = skimage.measure.compare_ssim(dollar_area_frame, np.squeeze(sprite.image_data[..., :3, i]), multichannel=True)

                if ssim > max_ssim:
                    max_ssim = ssim
                    dragon_alive = 1 # int(name[-1])

        return dragon_alive

    def _measure_dragon_coins(self, game_frame):
        coins_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["COINS_AREA"])

        return coins_area_frame[coins_area_frame[..., 2] > 150].size

    def _calculate_reward(self):
        reward = 0

        reward += (-0.5 if self.game_state["alive"][0] < self.game_state["alive"][1] else 0.05)
        # reward += (0.5 if (self.game_state["coins"][0] - self.game_state["coins"][1]) >= 1 else -0.05)

        return reward
