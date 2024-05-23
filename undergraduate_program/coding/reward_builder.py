"""
完整代码：https://github.com/wty-yy/KataCR/blob/master/katacr/policy/perceptron/reward_builder.py
"""
...  # import packages, define constants

class RewardBuilder:
  ...  # Define other functions

  def get_reward(self, verbose=False):
    ### Update King Tower ###
    ### Update Tower ###
    ### Calculate Reward ###
    ...
  
  def update(self, info):
    """
    Args:
      info (dict): The return in `VisualFusion.process()`,
        which has keys=[time, arena, cards, elixir]
    """
    self.time: int = info['time'] if not np.isinf(info['time']) else self.time
    self.arena: CRResults = info['arena']
    self.elixir: int = info['elixir']
    self.img = self.arena.get_rgb()
    self.box = self.arena.get_data()  # xyxy, track_id, conf, cls, bel
    self.frame_count += 1
