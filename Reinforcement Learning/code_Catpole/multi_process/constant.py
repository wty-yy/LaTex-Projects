from pathlib import Path

PATH_LOGS = Path("./logs")
PATH_LOGS.mkdir(parents=True, exist_ok=True)
PATH_FIGURES = Path("./figures")
PATH_FIGURES.mkdir(parents=True, exist_ok=True)
PATH_CHECKPOINT= Path(r"./checkpoints")

N_PLAYER = 6
STANDARD_STEP = 450
# STANDARD_STEP = 1

LEARNING_RATE = 1e-3
BATCH_SIZE = 32

EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

def get_logs():
    return f"n_player={N_PLAYER},lr={LEARNING_RATE},batch={BATCH_SIZE},epsilon_min={EPSILON_MIN}"

def mkdir_logs():
    path = PATH_LOGS.joinpath(get_logs())
    path.mkdir(parents=True, exist_ok=True)

def mkdir_figures():
    path = PATH_FIGURES.joinpath(get_logs())
    path.mkdir(parents=True, exist_ok=True)

def mkdir_checkpoints():
    path = PATH_CHECKPOINT.joinpath(get_logs())
    path.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    mkdir_logs()
    mkdir_figures()
    mkdir_checkpoints()