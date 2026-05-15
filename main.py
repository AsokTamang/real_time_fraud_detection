import sys
import os
sys.path.append('.')
from src.utils import load_object



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path    = os.path.join(BASE_DIR, 'data', 'pay_sim.csv')
model_path   = os.path.join(BASE_DIR, 'artifacts', 'model.pkl')


best_model_info = load_object(os.path.join('artifacts','best_model_info.pkl'))
print(best_model_info['threshold'])
