"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import json
import os
import sys
import argparse
import traceback
import joblib
from azureml.core import Run, Experiment, Workspace, Dataset, Datastore
from azureml.core.authentication import AzureCliAuthentication

from dotenv import load_dotenv
# For local development, set values in this section
load_dotenv()


def prepare_env(ws, env_name, conda_file):
    from azureml.core.environment import Environment
    from azureml.core.conda_dependencies import CondaDependencies

    # TODO: Get this in a seperate file
    env = Environment(env_name)
    cd = CondaDependencies.create(
        pip_packages=['azureml-dataset-runtime[pandas,fuse]',
                      'azureml-defaults', 'keras', 'tensorflow', 'matplotlib'],
        conda_packages=['scikit-learn==0.22.1']
    )

    env.python.conda_dependencies = cd

    # Register environment to re-use later
    env.register(workspace=ws)

    return env


def prepare_machines(ws):
    # If machine not yet ready, create !
    from azureml.core.compute import AmlCompute, ComputeTarget

    # choose a name for your cluster
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
    compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES")
    compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES")

    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_CPU_SKU")

    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print("Found compute target, will use this one: " + compute_name)
    else:
        print("creating new compute target...")
        provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                    min_nodes=compute_min_nodes,
                                                                    max_nodes=compute_max_nodes)

        compute_target = ComputeTarget.create(
            ws, compute_name, provisioning_config)

        compute_target.wait_for_completion(
            show_output=True, min_node_count=None, timeout_in_minutes=20)

    return compute_target


def prepare_training(dataset, script_folder, compute_target, env):
    from azureml.core import ScriptRunConfig

    # TODO: Dataset!!
    # TODO: compute_target,
    # TODO: env
    # TODO: Dynamic args
    epochs = os.environ.get("EPOCHS")
    batch_size = os.environ.get("BATCH_SIZE")
    conv_layers = os.environ.get("CONV_LAYERS")
    maxpooling = os.environ.get("MAXPOOLING")
    pool_size = os.environ.get("POOL_SIZE")
    conv_dense = os.environ.get("CONV_DENSE")
    val_dense = os.environ.get("VAL_DENSE")
    comb_dense = os.environ.get("COMB_DENSE")
    dropout = os.environ.get("DROPOUT")
    model_name = os.environ.get("MODEL_NAME")
    model_file = os.environ.get("MODEL_FILE")

    args = ['--data-folder', dataset.as_mount(), '--epochs', epochs, '--batch_size', batch_size, '--model_name', model_name, '--conv_layers',
            conv_layers, '--maxpooling', maxpooling, '--pool_size', pool_size, '--conv_dense', conv_dense, '--val_dense', val_dense, '--comb_dense', comb_dense, '--dropout', dropout, '--model_file', model_file]

    src = ScriptRunConfig(source_directory=script_folder,
                          script='train.py',
                          arguments=args,
                          compute_target=compute_target,  # still missing
                          environment=env)

    return src


def main():
    cli_auth = AzureCliAuthentication()

    workspace_name = os.environ.get("WORKSPACE_NAME")
    epochs = os.environ.get("EPOCHS")
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")

    env_name = os.environ.get("AML_ENV_NAME")
    env_conda_file = os.environ.get("AML_ENV_TRAIN_CONDA_DEP_FILE")

    dataset_name = os.environ.get("DATASET_NAME")

    script_folder = os.path.join(os.environ.get('ROOT_DIR'), 'scripts')
    config_state_folder = os.path.join(
        os.environ.get('ROOT_DIR'), 'config_states')

    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    # Prepare!
    dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)

    compute_target = prepare_machines(ws)
    env = prepare_env(ws, env_name, env_conda_file)
    src = prepare_training(dataset, script_folder, compute_target, env)
    # Start training
    exp = Experiment(workspace=ws, name=experiment_name)
    run = exp.submit(config=src)

    run.wait_for_completion()

    run_details = {k: v for k, v in run.get_details().items() if k not in [
        'inputDatasets', 'outputDatasets']}

    with open(config_state_folder + '/training-run.json', 'w') as training_run_json:
        json.dump(run_details, training_run_json)


if __name__ == '__main__':
    main()
