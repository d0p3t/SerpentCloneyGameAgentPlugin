import serpent.cv

from serpent.game_agent import GameAgent

from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

import offshoot

import time

from datetime import datetime


class SerpentCloneyGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None

    def setup_play(self):
        plugin_path = offshoot.config["file_paths"]["plugins"]

        context_classifier_path = f"{plugin_path}/SerpentCloneyGameAgentPlugin/files/ml_models/cloney_context_classifier.model"

        context_classifier = CNNInceptionV3ContextClassifier(input_shape=(288, 512, 3))
        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

        self.current_run = 0
        self.current_run_started_at = None

        self.last_run_duration = 0

    def handle_play(self, game_frame):
        context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)

        if context is None:
            return

        if context == "GAME_WORLD_1":
            self.handle_play_context_game_world(game_frame)
        elif context == "GAME_OVER":
            self.handle_play_context_game_over(game_frame)
        elif context == "MAIN_MENU":
            print(f"{context}")
        elif context == "GAME_PAUSE":
            self.handle_play_context_game_pause(game_frame)

    def handle_play_context_game_world(self, game_frame):
        self.input_controller.click_screen_region(screen_region="GAME_CENTER", game=self.game)
        self.display_game_agent_state(context="GAME_WORLD")

    def handle_play_context_game_over(self, game_frame):
        self.last_run_duration = (datetime.utcnow() - self.current_run_started_at).seconds if self.current_run_started_at else 0

        print("\033c")
        print("GAME: Coney         PLATFORM: Steam\n")
        print("CURRENT CONTEXT: GAME_OVER\n")

        print(f"LAST RUN: {self.current_run}")
        print(f"LAST RUN DURATION: {self.last_run_duration}")

        time.sleep(2)

        # Click PLAY button to start a new run
        self.input_controller.click_screen_region(screen_region="GAME_OVER_PLAY", game=self.game)
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
        self.input_controller.click_screen_region(screen_region="MAIN_MENU_PLAY", game=self.game)

    def handle_play_context_game_pause(self, game_frame):
        time.sleep(2)
        self.input_controller.click_screen_region(screen_region="GAME_PAUSE", game=self.game)

    def display_game_agent_state(self, context):
        print("\033c")
        print(f"GAME: Coney         PLATFORM: Steam\n")
        print(f"CURRENT CONTEXT: {context}\n")
        print(f"CURRENT RUN: {self.current_run}")
        print(f"CURRENT RUN DURATION: {(datetime.utcnow() - self.current_run_started_at)} seconds\n")
        print("")
        print(f"\nLAST RUN: {self.current_run - 1}")
        print(f"LAST RUN DURATION: {self.last_run_duration} seconds")
