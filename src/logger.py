import os
from pathlib import Path
import logging

dir_path =Path(__file__).parent
filepath  = os.path.join(dir_path, 'loggings.log')  #all the logs will go inside this file 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename=filepath)

