"""
完整代码：https://github.com/wty-yy/KataCR/blob/master/katacr/build_dataset/generator.py
"""
...  # import packages, define constants

class Generator:
  def __init__(
      self, background_index: int | None = None,
      unit_list: Tuple[Unit,...] = None,
      seed: int | None = None,
      intersect_ratio_thre: float = 0.5,
      map_update: dict = {'mode': 'naive','size': 5},
      augment: bool = True,
      dynamic_unit: bool = True,
      avail_names: Sequence[str] = None,
      noise_unit_ratio: float = 0.0,
    ):
    """
    Args:
      background_index: Use image file name in `dataset/images/segment/backgrounds/background{index}.jpg` as current background.
      unit_list: The list of units will be generated in arean.
      seed: The random seed.
      intersect_ratio_thre: The threshold to filte overlapping units.
      map_update_size: The changing size of dynamic generation distribution (SxS).
      augment: If taggled, the mask augmentation will be used.
      dynamic_unit: If taggled, the frequency of each unit will tend to average.
      avail_names: Specify the generation classes.
      noise_unit_ratio: The ratio of inavailable unit (noise unit) in whole units.
    Variables:
      map_cfg (dict):
        'ground': The 0/1 ground unit map in `katacr/build_dataset/generation_config.py`.
        'fly': The 0/1 fly unit map in `katacr/build_dataset/generation_config.py`.
        'update_size': The size of round squra.
    """
    ...
  
  def build(self, save_path="", verbose=False, show_box=False, box_format='cxcywh', img_size=None):
    ... # Build image and bounding box

  def add_tower(self, king=True, queen=True):
    ... # Add king and queen tower

  def add_unit(self, n=1):
    ... # Add unit in [ground, flying, others] randomly. Unit list looks at `katacr/constants/label_list.py`

  def reset(self):
    ... # Reset generator

if __name__ == '__main__':
  generator = Generator(seed=42, background_index=25, intersect_ratio_thre=0.5, augment=True, map_update={'mode': 'naive', 'size': 5}, avail_names=None)
  for i in range(10):
    generator.add_tower()
    generator.add_unit(n=40)
    x, box, _ = generator.build(verbose=False, show_box=True)
    generator.reset()