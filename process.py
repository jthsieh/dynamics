import os
from glob import glob
import numpy as np
import json

root = os.path.join(os.environ['HOME'], 'slowbro/bouncing_balls/balls_n4_t60_ex50000/')

all_trajs = []

for fname in glob(os.path.join(root, 'jsons/*.json')):
  data = json.load(open(fname, 'r'))
  for traj in data['trajectories']:
    # [tid, n_frames, n_balls, [x, y, mass, radius]]
    balls = []
    for ball in traj:
      info = [[f['position']['x'], f['position']['y'], f['mass'], f['sizemul']] for f in ball]
      balls += info,
    balls = np.array(balls).transpose([1,0,2])
    all_trajs += balls,

all_trajs = np.array(all_trajs)
print('shape:', all_trajs.shape)

fout = os.path.join(root, 'dataset_info.npy')
np.save(fout, all_trajs)
