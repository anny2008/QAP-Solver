import os
import csv
import re

# Define paths
LOG_FOLDER = "results/sdp_volume"
OUTPUT_CSV = "results/sdp_volume_results.csv"

# Prepare the CSV file
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    # Write header columns
    writer.writerow(["Instance Name", "Formulation", "Eigen Solver", "Subproblem Time", "Objective Value", "N Iterations", "Solving Time"])

    result = []
    # Loop through all files in the folder
    for filename in os.listdir(LOG_FOLDER):
        if filename.endswith(".log"):  # Only process log files
            file_path = os.path.join(LOG_FOLDER, filename)
            # formulation1_bur26a.dat.log
            # extract the instance name from the filename
            instance_name = filename.split("_")[1]  # This will give you 'bur26a.dat.log'
            # remove the .log extension
            instance_name = instance_name.replace(".log", "")  # This will give you 'bur26a.dat'
            # remove the .dat extension
            instance_name = instance_name.replace(".dat", "")  # This will give you 'bur26a'
            # Now instance_name contains the instance name without extensions
            # extract the formulation from the filename
            formulation = filename.split("_")[0]  # This will give you 'formulation1'
            # extract n from the instance_name

            match = re.search(r'\d+', instance_name)
            n = int(match.group()) if match else None  # Returns None if no number is found

            try:
                # check if the file end with this format:
                # {instance_name}.dat & {formulation} & {eigen_solver} & {subproblem_time} & {objective_value} & {n_iterations} & {solving_time}\\
                # read file from the last line to the first line and find the line that starts with "{instance_name}.dat & {formulation} &"
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in reversed(f.readlines()):
                        if line.startswith(f"{instance_name}.dat & {formulation} &"):
                            # remove "\\"
                            line = line.replace("\\", "").strip()
                            # split the line by "&" and extract the values
                            tokenized_line = line.split("&")
                            eigen_solver = tokenized_line[2].strip()
                            subproblem_time = tokenized_line[3].strip()
                            objective_value = tokenized_line[4].strip()
                            n_iterations = tokenized_line[5].strip()
                            solving_time = tokenized_line[6].strip()
                            result.append(
                                {
                                    "instance_name": instance_name,
                                    "n": n,
                                    "formulation": formulation,
                                    "eigen_solver": eigen_solver,
                                    "subproblem_time": subproblem_time,
                                    "objective_value": objective_value,
                                    "n_iterations": n_iterations,
                                    "solving_time": solving_time
                                }
                            )
                            break  # Stop after finding the first matching line
            except Exception as e:
                print(f"Could not read {filename}: {e}")
    print(f"Collected results from {len(result)} instances. Writing to CSV...")
    # sort the results by n,instance name and formulation
    sorted_results = sorted(result, key=lambda x: (x["n"], x["instance_name"], x["formulation"]))
    # sorted_results = sorted(result.items(), key=lambda x: (x[1][0]["n"], x[1][0]["formulation_number"], x[0]))
    # sorted_results = result.items()
    for record in sorted_results:
        # write the results to the csv file
        instance_name = record["instance_name"]
        formulation = record["formulation"]
        eigen_solver = record["eigen_solver"]
        subproblem_time = record["subproblem_time"]
        objective_value = record["objective_value"]
        n_iterations = record["n_iterations"]
        solving_time = record["solving_time"]
        writer.writerow([instance_name, formulation, eigen_solver, subproblem_time, objective_value, n_iterations, solving_time])
print(f"Done! Results collected into {OUTPUT_CSV}")
