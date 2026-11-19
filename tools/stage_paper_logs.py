import os
import subprocess

def main():
    root_dir = "runs/uav_paper_ablation"
    files_to_add = []
    for root, dirs, files in os.walk(root_dir):
        if "weights" in root:
            continue
        for f in files:
            if f.endswith(".pt") or f.endswith(".weights"):
                continue
            fp = os.path.join(root, f).replace("\\", "/")
            files_to_add.append(fp)
    
    print(f"Adding {len(files_to_add)} log files from {root_dir}...")
    batch_size = 25
    for i in range(0, len(files_to_add), batch_size):
        chunk = files_to_add[i:i + batch_size]
        cmd = ["git", "add", "-f"] + chunk
        subprocess.check_call(cmd)
    print("All log files staged successfully!")

if __name__ == "__main__":
    main()
