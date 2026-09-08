


import os


QAPLIB_DIR = "data/QAPLIB/"
#  read all data file from QAPLIB folder into a list of instance name and size
instances = []
for f in os.listdir(QAPLIB_DIR):
    if f.endswith('.dat'):
        with open(os.path.join(QAPLIB_DIR, f), 'r') as file:
            n = int(file.readline().split()[0])
        instances.append((f, n))

# sort the instances by size increasingly
instances.sort(key=lambda x: x[1])
if not os.path.exists("results/sdp_volume"):
    os.makedirs("results/sdp_volume")
    

# run the benchmark for each instance and each formulation and write each output to a log file
formulations = ["formulation3", "formulation4", "formulation2"]
for instance, size in instances:
    for formulation in formulations:
        log_file = f"results/sdp_volume/{formulation}_{instance}.log"
        # before running the benchmark, check if the log file already exists, 
        # if it does, check if it contains the line "Solving time" and 
        # if it does, skip this instance and formulation else run the benchmark and write the output to the log file
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                if any("Volume Algorithm Complete" in line for line in f):
                    print(f"Skipping {formulation} on {instance} (size {size}) as log file already exists and contains results.")
                    continue
        
        
        n_threads = size if size < 128 else 128
        print(f"Running {formulation} on {instance} (size {size})")
        # run python -u tools/qap_cli.py run --module sdp_volume --config config_json and redirect output to log_file
        os.system(f"python -u tools/qap_cli.py run --module sdp_volume --formulation {formulation} --instance {QAPLIB_DIR}{instance} --threads {n_threads} > {log_file} 2>&1")
        
# formulation = "formulation1"
# for instance, size in instances:
#     log_file = f"results/sdp_volume/{formulation}_{instance}.log"
#     n_threads = size if size < 128 else 128
#     print(f"Running {formulation} on {instance} (size {size})")
#     # run python -u tools/qap_cli.py run --module sdp_volume --config config_json and redirect output to log_file
#     os.system(f"python -u tools/qap_cli.py run --module sdp_volume --formulation {formulation} --instance {QAPLIB_DIR}{instance} --threads {n_threads} > {log_file} 2>&1")