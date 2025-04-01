#!/usr/bin/env python3
import os
import glob
import subprocess
import time

start_time = time.time()

# ------------------ HARD-CODED CONFIG ------------------
# List of GPUs you want to use (by ID/index)
GPUs = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
]  # For example, skip GPU 0 for interactive use

# Folder containing all your .txt anti-bias statements
BIAS_TXT_FOLDER = "./prompts/generated_anti_bias_statements"

# Path to your main script
MAIN_SCRIPT = "mypkg/main_paper_dataset.py"

# Any extra arguments you want to pass to each run
EXTRA_ARGS = "--downsample 100 --gpu_inference --political_orientation"
# ------------------ END OF CONFIG ------------------

# Find all .txt files in the folder
txt_files = sorted(glob.glob(os.path.join(BIAS_TXT_FOLDER, "*.txt")))
if not txt_files:
    raise RuntimeError(f"No .txt files found in {BIAS_TXT_FOLDER}")

n_gpus = len(GPUs)
n_files = len(txt_files)

txt_files = [prompt.split("/")[-1] for prompt in txt_files]


print(txt_files)

# Figure out how many files per GPU (round up)
chunk_size = (n_files + n_gpus - 1) // n_gpus

processes = []
index = 0

for gpu in GPUs:
    # Grab a subset of .txt files for this GPU
    subset = txt_files[index : index + chunk_size]
    index += chunk_size
    if not subset:
        break  # No more files left to process

    # Create log filename for this GPU
    log_file = f"GPU_{gpu}_inference_log.txt"

    # Build a single shell command that runs each file *sequentially*
    # on the assigned GPU, separated by semicolons.
    commands = []
    for txt_file in subset:
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} python {MAIN_SCRIPT} "
            f"--anti_bias_statement_file {txt_file} {EXTRA_ARGS}"
        )
        commands.append(cmd)

    # Join them with semicolons so they run sequentially on that GPU
    combined_command = " ; ".join(commands)
    print(f"Launching on GPU {gpu} with these files: {subset}")
    print(f"Output will be logged to: {log_file}")
    print(f"Final shell command:\n{combined_command}\n")

    # Spawn a subprocess that executes the entire combined command with output redirected to log file
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            combined_command, shell=True, stdout=f, stderr=subprocess.STDOUT
        )
    processes.append(proc)

# Wait for all GPUs' processes to complete
for p in processes:
    p.wait()

print("\nAll jobs are done!")

end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds")
