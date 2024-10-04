# Preliminary setup of execution environment
import os
from pathlib import Path
import subprocess

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Limit CUDA visibility
os.environ["OMP_NUM_THREADS"] = "1" # Disable BLAS parallelism
os.environ["PYTHONPATH"] = str(nntile_dir / "build" / "wrappers" / "python") # Path to a binary dir of NNTile Python wrappers

# All StarPU environment variables are available at https://files.inria.fr/starpu/doc/html/ExecutionConfigurationThroughEnvironmentVariables.html
os.environ["STARPU_NCPU"] = "1" # Use only 1 CPU core
os.environ["STARPU_NCUDA"] = "1" # Use only 1 CUDA device
os.environ["STARPU_SILENT"] = "1" # Do not show lots of StarPU outputs
os.environ["STARPU_SCHED"] = "dmdasd" # Name StarPU scheduler to be used
os.environ["STARPU_FXT_TRACE"] = "0" # Do not generate FXT traces
os.environ["STARPU_WORKERS_NOBIND"] = "1" # Do not bind workers (it helps if several instances of StarPU run in parallel)
os.environ["STARPU_PROFILING"] = "1" # This enables logging performance of workers and bandwidth of memory nodes
os.environ["STARPU_HOME"] = str(Path.cwd() / "starpu") # Main directory in which StarPU stores its configuration files
os.environ["STARPU_PERF_MODEL_DIR"] = str(Path(os.environ["STARPU_HOME"]) / "sampling") # Main directory in which StarPU stores its performance model files
os.environ["STARPU_PERF_MODEL_HOMOGENEOUS_CPU"] = "1" # Assume all CPU cores are equal
os.environ["STARPU_PERF_MODEL_HOMOGENEOUS_CUDA"] = "1" # Assume all CUDA devices are equal
os.environ["STARPU_HOSTNAME"] = "GPT2_example" # Force the hostname to be used when managing performance model files
os.environ["STARPU_FXT_PREFIX"] = str(Path(os.environ["STARPU_HOME"]) / "fxt") # Directory to store FXT traces if enabled