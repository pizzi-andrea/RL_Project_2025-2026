import subprocess
import sys

doom_scenarios = [
    "VizdoomBasic-v0",
    "VizdoomCorridor-v0",
    "VizdoomDeathmatch-v0",
    "VizdoomDefendCenter-v0",
    "VizdoomDefendLine-v0",
    "VizdoomHealthGatheringSupreme-v0",
    "VizdoomMyWayHome-v0",
]

episodes = [1, 3, 5, 1, 1, 5, 1]
dataset_name = "Doom4VAE"

PY = sys.executable  # usa quello dell'env attivo

def run(args):
    print("\n>>>", " ".join(args))
    subprocess.check_call(args)

for env, n in zip(doom_scenarios, episodes):
    run([PY, "doomGen.py", "-d", dataset_name, "-n", str(n), "-s", "train", "-e", env, "-r", "True", "-m", "False"])
    run([PY, "doomGen.py", "-d", dataset_name, "-n", "1",      "-s", "validation", "-e", env, "-r", "True", "-m", "False"])
    run([PY, "doomGen.py", "-d", dataset_name, "-n", "1",      "-s", "test", "-e", env, "-r", "True", "-m", "False"])

print("\nDONE.")
