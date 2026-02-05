import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def find_valid_logs():
    log_root = "logs"
    if not os.path.exists(log_root):
        log_root = "tb_logs"
    
    if not os.path.exists(log_root):
        print("No log directory found.")
        return

    subdirs = [os.path.join(log_root, d) for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]
    
    # Sort by time
    subdirs.sort(key=os.path.getmtime, reverse=True)
    
    print(f"Checking {len(subdirs)} directories for 'rollout/ep_rew_mean'...")
    
    for d in subdirs:
        event_files = glob.glob(os.path.join(d, "events.out.tfevents.*"))
        if not event_files:
            continue
            
        event_file = max(event_files, key=os.path.getsize)
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            tags = ea.Tags()['scalars']
            if 'rollout/ep_rew_mean' in tags:
                print(f"[FOUND] {d} has reward data!")
                # Print some stats
                events = ea.Scalars('rollout/ep_rew_mean')
                print(f"   - Steps: {len(events)}")
                print(f"   - Last Step: {events[-1].step}")
                print(f"   - Last Value: {events[-1].value}")
                return # Stop after finding the latest valid one
            else:
                print(f"[MISSING] {d} - Tags: {tags[:5]}...")
        except Exception as e:
            print(f"[ERROR] {d}: {e}")

if __name__ == "__main__":
    find_valid_logs()
