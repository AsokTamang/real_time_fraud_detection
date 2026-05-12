import os
from pathlib import Path
import logging

dir_path =Path(__file__).parent
filepath  = os.path.join(dir_path, 'loggings.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename=filepath)

