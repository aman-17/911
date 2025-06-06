BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
echo "Beaker user: $BEAKER_USER"


cat << 'EOF' > /data/input/amanr/911_train.py
#!/usr/bin/env python3
import sys
from beaker import Beaker, ExperimentSpec, TaskSpec, TaskContext, ResultSpec, TaskResources, ImageSource, Priority, Constraints, EnvVar

b = Beaker.from_env(default_workspace="ai2/oe-support")

pipeline_cmd = "python train.py"
commands = []
commands.extend([
    "git clone https://github.com/aman-17/911.git",
    "cd 911",
    "pip install -e .",
    "git checkout parallelism",
    pipeline_cmd,
])

task_spec_args = {
    "name": "MLA-Training",
    "image": ImageSource(beaker="ai2/cuda11.8-ubuntu20.04"),
    "command": [
        "bash", "-c",
        " && ".join(commands)
    ],
    "env_vars": [
        EnvVar(name="NCCL_SOCKET_IFNAME", value="ib"),
        EnvVar(name="NCCL_IB_HCA", value="^=mlx5_bond_0"),
        EnvVar(name="NCCL_DEBUG", value="INFO"),
    ],
    "context": TaskContext(
        priority=Priority.normal,
        preemptible=True,
    ),
    "resources": TaskResources(gpu_count=2),
    "constraints": Constraints(cluster=["ai2/titan-cirrascale","ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]),
    "result": ResultSpec(path="/noop-results"),
}

experiment_spec = ExperimentSpec(
    description=f"MLA test on OLMo",
    budget="ai2/oe-training",
    tasks=[TaskSpec(**task_spec_args)],
)

experiment = b.experiment.create(spec=experiment_spec, workspace="ai2/oe-support")
print(f"Created experiment: {experiment.id}")
print(f"View at: https://beaker.org/ex/{experiment.id}")
EOF
chmod +x /data/input/amanr/911_train.py
echo "Creating Beaker experiment..."
$PYTHON /data/input/amanr/911_train.py

rm /data/input/amanr/911_train.py

echo "Training experiment submitted successfully!"