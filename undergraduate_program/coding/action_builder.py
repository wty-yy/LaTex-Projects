"""
完整代码：https://github.com/wty-yy/KataCR/blob/master/katacr/policy/perceptron/action_builder.py
"""
...  # import packages, define constants

class ActionBuilder:
  ...  # define other functions

  def get_action(self, verbose=False):
    ...  # Get from actions queue
  
  def update(self, info):
    """
    Args:
      info (dict): The return in `VisualFusion.process()`,
        which has keys=[time, arena, cards, elixir]
    """
    self.time: int = info['time'] if not np.isinf(info['time']) else self.time
    self.arena: CRResults = info['arena']
    self.cards: dict = info['cards']
    self.elixir: int = info['elixir']
    self.card2idx: dict = info['card2idx']
    self.box = self.arena.get_data()  # xyxy, track_id, conf, cls, bel
    self.img = self.arena.get_rgb()
    self.frame_count += 1
    ### Step 1: Update elixir history ###
    self._update_elixir()
    # print("Elixirs:", self.elixirs)
    ### Step 2: Update card memory ###
    self._update_cards()
    ### Step 3: Update last elixir ###
    self._update_mutation_elixir_num()
    ### Step 4: Find new action ###
    self._find_action()
