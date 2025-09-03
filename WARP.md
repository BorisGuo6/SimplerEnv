# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Overview
- Purpose: Simulated manipulation policy evaluation environments for real robot setups, built on SAPIEN and ManiSkill2, with optional integration to Octo, RT-1, and MolmoAct policies.
- Primary Python version: 3.10 (see setup.py and .pre-commit-config.yaml).
- Key entry points:
  - simpler_env/main_inference.py: CLI driver that selects a policy adapter, applies GPU/runtime guards, and invokes the evaluator.
  - simpler_env/evaluation/maniskill2_evaluator.py: Core evaluation loop and logging; builds environments, executes episodes, saves videos and action plots.
  - simpler_env/__init__.py: Task-to-env mapper and simpler_env.make() for prepackaged ManiSkill2 configs.
  - simpler_env/policies/*: Policy adapters (RT-1, Octo, MolmoAct) that normalize outputs into a common action dictionary.
  - simpler_env/utils/env/*: Environment builder and observation utilities.

Install and Build
- Minimal local install (recommended for development):
  ```bash path=null start=null
  # Python 3.10 environment
  conda create -n simpler_env python=3.10
  conda activate simpler_env

  # Clone with submodules
  git clone https://github.com/allenai/SimplerEnv --recurse-submodules
  cd SimplerEnv

  # Pin numpy before installing ManiSkill2_real2sim to avoid IK issues
  pip install numpy==1.24.4

  # Install the embedded ManiSkill2 real-to-sim environments
  pip install -e ./ManiSkill2_real2sim

  # Install this package
  pip install -e .
  ```
- Full install for bundled policy inference (RT-1, Octo, env building):
  ```bash path=null start=null
  sudo apt install -y ffmpeg
  pip install tensorflow==2.15.0
  pip install -r requirements_full_install.txt
  pip install "tensorflow[and-cuda]==2.15.1"
  pip install git+https://github.com/nathanrooy/simulated-annealing
  ```
- Docker image (reproducible environment):
  ```bash path=null start=null
  # Prereq (see README: issue #64). Copy host NVIDIA ICDs into build context:
  mkdir -p docker/usr_share_nvidia
  sudo cp -r /usr/share/nvidia/* docker/usr_share_nvidia/

  # Build
  docker build -t simplerenv:latest docker
  ```

Common Commands
- Linting and formatting (configured via pyproject.toml and .flake8):
  ```bash path=null start=null
  # One-time hook install
  pre-commit install

  # Run all hooks on entire repo
  pre-commit run --all-files

  # Or run individually
  black .
  isort .
  flake8 .
  ```
- List available high-level tasks (ENVIRONMENTS):
  ```python path=null start=null
  import simpler_env
  print(simpler_env.ENVIRONMENTS)
  ```
- Quick single-episode evaluation on prepackaged visual-matching envs:
  ```bash path=null start=null
  # RT-1 (set your checkpoint path)
  python simpler_env/simple_inference_visual_matching_prepackaged_envs.py \
    --policy rt1 \
    --ckpt-path ./checkpoints/rt_1_tf_trained_for_000400120 \
    --task google_robot_pick_coke_can \
    --logging-root ./results_simple_eval \
    --n-trajs 1

  # Octo (no checkpoint path required; picks huggingface model by name)
  python simpler_env/simple_inference_visual_matching_prepackaged_envs.py \
    --policy octo-base \
    --ckpt-path None \
    --task widowx_spoon_on_towel \
    --logging-root ./results_simple_eval \
    --n-trajs 1
  ```
- Full evaluator (single run) using main_inference.py:
  ```bash path=null start=null
  # Example: RT-1 on Google Robot pick-coke-can visual matching setup
  CUDA_VISIBLE_DEVICES=0 python simpler_env/main_inference.py \
    --policy-model rt1 \
    --ckpt-path ./checkpoints/rt_1_tf_trained_for_000400120 \
    --robot google_robot_static \
    --env-name GraspSingleOpenedCokeCanInScene-v0 \
    --scene-name google_pick_coke_can_1_v4 \
    --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --obj-init-x -0.20 -0.20 1 --obj-init-y 0.10 0.10 1 \
    --logging-dir ./results
  ```
  Notes:
  - Use --additional-env-build-kwargs to parameterize env variants (e.g., urdf_version=recolor_tabletop_visual_matching_1).
  - For Octo, use --policy-model octo-base and --ckpt-path None.
- Batch sweeps (prebuilt experiment scripts):
  ```bash path=null start=null
  # Examples (see scripts/ for more):
  bash scripts/rt1_pick_coke_can_visual_matching.sh
  bash scripts/rt1_move_near_visual_matching.sh
  bash scripts/octo_move_near_visual_matching.sh
  ```
- Metrics for SIMPLER paper (reproduces summary numbers):
  ```bash path=null start=null
  python tools/calc_metrics.py
  ```

Running a “single test”
- There is no formal pytest suite. To validate code changes quickly, run a single episode using either of the two patterns above:
  - Minimal prepackaged envs: set --n-trajs 1 on simpler_env/simple_inference_visual_matching_prepackaged_envs.py.
  - Full evaluator: constrain --robot-init-*, --obj-init-* to single values (… a a 1) as shown to produce exactly one episode.

Architecture and Data Flow
- simpler_env/__init__.py
  - Exposes ENVIRONMENTS and a task registry ENVIRONMENT_MAP mapping high-level task names (e.g., google_robot_pick_coke_can) to ManiSkill2 env IDs (e.g., GraspSingleOpenedCokeCanInScene-v0) plus default kwargs.
  - simpler_env.make(task, **overrides) sets obs_mode="rgbd" and prepackaged_config=True, then gym.make().
- Policy adapters (simpler_env/policies/)
  - RT-1 (rt1/rt1_model.py):
    - Wraps a TF-Agents SavedModel policy; embeds task language via TFHub USE-Large; resizes camera frames; normalizes/filters actions; converts rotation to axis-angle; handles embodiment differences via policy_setup (google_robot vs widowx_bridge).
  - Octo (octo/octo_model.py):
    - Loads rail-berkeley/octo-* from HuggingFace; maintains image history and pad mask; uses JAX RNG per step; ensembles action horizon; converts to a common action dict including sticky gripper logic tuned per embodiment.
  - MolmoAct (policies/molmoact/):
    - Two variants: HF and vLLM (molmoact_model_vllm.py). The vLLM path registers a custom multimodal model with vLLM, parses generated text into depth/trajectory/action via MolmoActParser, then emits normalized actions with google_robot/widowx_bridge gripper handling.
- Evaluation loop (simpler_env/evaluation/maniskill2_evaluator.py)
  - Builds env via utils.env.env_builder.build_maniskill2_env(env_name, …) including optional RGB overlay and ray-tracing.
  - Resets with robot_init and obj_init options; obtains per-step images via utils.env.observation_utils.
  - Drives the selected policy adapter’s reset()/step() until terminate/timeout, handling multi-subtask progression via env.is_final_subtask()/advance_to_next_subtask().
  - Logs annotated videos and action plots under args.logging_dir, with a structured subdirectory encoding model, scene, control mode, env variants, and pose.
- Environment utilities (simpler_env/utils/env/)
  - env_builder.py selects default overlay cameras based on robot type and returns gym.make() environments; get_robot_control_mode() picks control strings per robot/policy.
  - observation_utils.py selects camera streams by robot type (overhead_camera for google_robot; 3rd_view_camera for widowx).
- Metrics (simpler_env/utils/metrics.py; tools/calc_metrics.py)
  - Implements mean_maximum_rank_violation and pearson_correlation; includes published REAL_PERF/SIMPLER_PERF tables and a script to print per-task metrics.
- External environments (ManiSkill2_real2sim/)
  - Embedded codebase providing task/robot/object assets and scenes used by the evaluator. Refer to README “Code Structure” for an overview.

Project-specific notes
- GPU/runtime guards:
  - main_inference.py sets DISPLAY="", XLA_PYTHON_CLIENT_PREALLOCATE=false, and optionally a TensorFlow logical GPU memory limit (via --tf-memory-limit).
  - Critical ordering for MolmoAct vLLM: PyTorch/vLLM must initialize before TensorFlow to avoid GPU device registration conflicts (handled in main_inference.py).
- Octo with CUDA 12: ensure CUDA >= 12.2 for JAX. Example run with explicit PATH/LD_LIBRARY_PATH (per README):
  ```bash path=null start=null
  PATH=/usr/local/cuda-12.3/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH \
  bash scripts/octo_move_near_visual_matching.sh
  ```
- Adding new policies or environments: see README sections “Adding New Policies” and “Adding New Real-to-Sim Evaluation Environments and Robots” (ADDING_NEW_ENVS_ROBOTS.md).

Style and configuration
- Tooling:
  - Black (line-length 120) and isort (profile=black) configured in pyproject.toml.
  - Flake8 configured in .flake8 (line length 120; explicit ignore list).
  - Pre-commit includes black, isort, flake8, and basic hygiene hooks; default python is 3.10.

Where outputs go
- By default, results are written under --logging-dir (default ./results) with nested directories capturing checkpoint/model, scene, control mode, env variant tags, and robot/object poses. Episodes save:
  - Annotated video (.mp4) per episode.
  - Action time-series plots under an actions/ subdirectory.

