import re
import pandas as pd

class QAPLogAnalyzer:
    def __init__(self, log_file):
        self.log_file = log_file
        self.ils_iterations = []
        self.ils_objectives = []
        self.ils_2opt_stats = []
        self._parse_log()

    def _parse_log(self):
        current_ils = None
        current_2opt = []
        with open(self.log_file) as f:
            for line in f:
                m_ils = re.search(r"\[ILS\] Iteration (\d+), objective = ([\d\.eE+-]+)", line)
                if m_ils:
                    if current_ils is not None and current_2opt:
                        initial_val = current_2opt[0]["objective"]
                        final_val = current_2opt[-1]["objective"]
                        self.ils_2opt_stats.append({
                            "ils_iteration": current_ils,
                            "initial_2opt": initial_val,
                            "final_2opt": final_val,
                            "improvement": initial_val - final_val,
                            "num_2opt_iter": len(current_2opt)
                        })
                    current_ils = int(m_ils.group(1))
                    self.ils_iterations.append(current_ils)
                    self.ils_objectives.append(float(m_ils.group(2)))
                    current_2opt = []
                m_2opt = re.search(r"\[2-opt\] Iteration (\d+), objective = ([\d\.eE+-]+), time = ([\d\.eE+-]+)s", line)
                if m_2opt and current_ils is not None:
                    current_2opt.append({
                        "iteration": int(m_2opt.group(1)),
                        "objective": float(m_2opt.group(2)),
                        "time": float(m_2opt.group(3))
                    })
            if current_ils is not None and current_2opt:
                initial_val = current_2opt[0]["objective"]
                final_val = current_2opt[-1]["objective"]
                self.ils_2opt_stats.append({
                    "ils_iteration": current_ils,
                    "initial_2opt": initial_val,
                    "final_2opt": final_val,
                    "improvement": initial_val - final_val,
                    "num_2opt_iter": len(current_2opt)
                })

    def get_ils_objectives(self):
        """Return DataFrame of ILS iteration and objective values."""
        return pd.DataFrame({
            "iteration": self.ils_iterations,
            "objective": self.ils_objectives
        })

    def get_2opt_stats(self):
        """Return DataFrame of 2-opt statistics per ILS iteration."""
        return pd.DataFrame(self.ils_2opt_stats)

    def get_ils_2opt_stats(self):
        """Return DataFrame with initial/final 2-opt values, improvement, and count for each ILS iteration."""
        return pd.DataFrame(self.ils_2opt_stats)

    def plot_ils_objectives(self, ax=None):
        import matplotlib.pyplot as plt
        df = self.get_ils_objectives()
        if ax is None:
            plt.figure(figsize=(10,6))
            ax = plt.gca()
        ax.plot(df["iteration"], df["objective"], marker='o')
        ax.set_xlabel("ILS Iteration")
        ax.set_ylabel("Objective Value")
        ax.set_title("Objective Value over ILS Iterations")
        ax.grid(True)
        plt.tight_layout()
        if ax is None:
            plt.show()
        return ax

    def summary(self):
        print("ILS Objective Summary:")
        display(self.get_ils_objectives().describe())
        print("\n2-opt Inner Loop Summary:")
        display(self.get_ils_2opt_stats().describe())

        return pd.DataFrame(self.ils_2opt_stats)

    def plot_ils_objectives(self, ax=None):
        import matplotlib.pyplot as plt
        df = self.get_ils_objectives()
        if ax is None:
            plt.figure(figsize=(10,6))
            ax = plt.gca()
        ax.plot(df["iteration"], df["objective"], marker='o')
        ax.set_xlabel("ILS Iteration")
        ax.set_ylabel("Objective Value")
        ax.set_title("Objective Value over ILS Iterations")
        ax.grid(True)
        plt.tight_layout()
        if ax is None:
            plt.show()
        return ax

    def summary(self):
        print("ILS Objective Summary:")
        display(self.get_ils_objectives().describe())
        print("\n2-opt Inner Loop Summary:")
        display(self.get_ils_2opt_stats().describe())
