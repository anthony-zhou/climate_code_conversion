from importlib.machinery import ModuleSpec
import os
import time
import importlib.util
import pandas as pd
import plotly.graph_objects as go

ci = 35
lmr_z = 4
par_z = 500
gb_mol = 50_000
je = 40
cair = 45
oair = 21000
rh_can = 0.40
p = 1
iv = 1
c = 1


def run_and_record(path):
    for filename in os.listdir(path):
        if filename.endswith(".py"):
            module_name = filename[:-3]  # Remove .py extension
            file_path = os.path.join(path, filename)

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "main"):
                results = []
                for n in range(0, 1001, 50):
                    total_runtime = 0
                    ci_val, gs_mol = 0, 0
                    for _ in range(5):  # Loop for 5 runs
                        start_time = time.time()
                        for _ in range(0, n):
                            ci_val, gs_mol = module.main(
                                ci,
                                lmr_z,
                                par_z,
                                gb_mol,
                                je,
                                cair,
                                oair,
                                rh_can,
                                p,
                                iv,
                                c,
                            )
                        end_time = time.time()
                        runtime = end_time - start_time
                        total_runtime += runtime
                    average_runtime = total_runtime / 5  # Compute the average runtime
                    results.append((ci_val, gs_mol, n, average_runtime))

                with open(f"{path}/{module_name}_runtime.txt", "w") as f:
                    for ci_val, gs_mol, n, runtime in results:
                        f.write(f"{ci_val}, {gs_mol}, {n}, {runtime}\n")


def plot_runtimes(path):
    dataframes = {}

    for filename in os.listdir(path):
        if filename.endswith("_runtime.txt"):
            with open(os.path.join(path, filename), "r") as f:
                dataframes[filename] = pd.read_csv(
                    f, names=["ci", "gs_mol", "Trials", "Runtime"]
                )

    fig = go.Figure()

    for file in dataframes.keys():
        file_df = dataframes[file]
        fig.add_trace(
            go.Scatter(
                # Ignore the first trial, which is when numba gets initialized.
                x=file_df["Trials"][2:],
                y=file_df["Runtime"][2:],
                mode="lines",
                name=file,
            )
        )

    fig.update_layout(
        title="Runtime of ci solver (averaged across 5 runs)",
        xaxis_title="Number of Trials",
        yaxis_title="Runtime (s)",
    )

    # fig.update_xaxes(type="log")
    # fig.update_yaxes(type="log")

    fig.show()


if __name__ == "__main__":
    run_and_record("comparisons")
    plot_runtimes("comparisons")
