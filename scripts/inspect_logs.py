import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def inspect_latest_log():
    log_root = "logs"
    if not os.path.exists(log_root):
        log_root = "tb_logs"
    
    subdirs = [os.path.join(log_root, d) for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]
    if not subdirs:
        print("No subdirectories found.")
        return

    latest_subdir = max(subdirs, key=os.path.getmtime)
    print(f"Inspecting: {latest_subdir}")
    
    event_files = glob.glob(os.path.join(latest_subdir, "events.out.tfevents.*"))
    if not event_files:
        print("No event files found.")
        return

    event_file = max(event_files, key=os.path.getsize)
    ea = EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags()['scalars']
    print("Found tags:")
    for t in tags:
        print(f"- {t}")

if __name__ == "__main__":
    inspect_latest_log()
