"""
完整代码：https://github.com/wty-yy/KataCR/blob/master/katacr/policy/perceptron/state_builder.py
"""
...  # import packages, define constants

class StateBuilder:
  def get_state(self, verbose=False):
    ...  # Build state

  def update(self, info: dict, deploy_cards: set):
    """
    Args:
      info (dict): The return in `VisualFusion.process()`,
        which has keys=[time, arena, cards, elixir]
      deploy_cards (set): Get deploy_cards from action_builder.
    """
    self.time: int = info['time'] if not np.isinf(info['time']) else self.time
    self.arena: CRResults = info['arena']
    self.cards: List[str] = info['cards']
    self.elixir: int = info['elixir']
    self.card2idx: dict = info['card2idx']
    self.parts_pos: np.ndarray = info['parts_pos']  # shape=(3, 4), part1,2,3, (x,y,w,h)
    self.box = self.arena.get_data()  # xyxy, track_id, conf, cls, bel
    self.img = self.arena.get_rgb()
    self.frame_count += 1
    ### Step 0: Update belong memory ###
    self._update_bel_memory()
    ### Step 1: Find text information ###
    self._find_text_info(deploy_cards)
    ### Step 2: Build bar items ###
    self._build_bar_items()
    ### Step 3: Combine units and bar2 with their BarItem ###
    self._combine_bar_items()
    ### Step 4: Update bar history ###
    self._update_bar_items_history()
    ### Step 5: Update class memory, if body exists ###
    self._update_cls_memory()
  