import sys
import os
sys.path.append('.')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path    = os.path.join(BASE_DIR, 'data', 'pay_sim.csv')
model_path   = os.path.join(BASE_DIR, 'artifacts', 'model.pkl')
