# config.py
import yaml
import logging
import os

class Config:
    def __init__(self, config_file, y_map_file, logger=None):
        self.config_file = config_file
        self.y_map_file = y_map_file
        self.logger = logger if logger else Logger().logger
        self.config = self.load_config()
        self.y_map = self.load_y_map()

    def load_config(self):
        try:
            with open(self.config_file, "r") as f:
                config = yaml.safe_load(f)
                self.logger.info("Successfully loaded the config YAML file.")
                return config
        except Exception as e:
            self.logger.error(f"Failed to load the config YAML file due to {str(e)}")
            return None

    def load_y_map(self):
        try:
            with open(self.y_map_file, "r") as f:
                y_map = yaml.safe_load(f)
                self.logger.info("Successfully loaded the y-map YAML file.")
                return y_map
        except Exception as e:
            self.logger.error(f"Failed to load the y-map YAML file due to {str(e)}")
            return None

class Logger:
    def __init__(self, show_message=True):
        msg=f"""
        #######################################################################################################
            For controlling verbosity of messages set LOG_LEVEL environment variable (e.g., INFO, DEBUG)
        #######################################################################################################
        """
        if show_message: print(msg)

        # Create a logger
        self.logger = logging.getLogger(__name__)

        # Set the log level based on the LOG_LEVEL environment variable
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.logger.setLevel(log_level)

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(ch)
        