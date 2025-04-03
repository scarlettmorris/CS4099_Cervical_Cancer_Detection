import time
import logging

class Logger:
    def __init__(self, model):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        log_filename = f"{model}_{timestamp}.log"
        
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        self.logger = logging.getLogger(model)

    def get_logger(self):
        return self.logger