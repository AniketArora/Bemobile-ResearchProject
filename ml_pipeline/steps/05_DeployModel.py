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
from azureml.core import Run, Experiment, Workspace, Model
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import Webservice, AciWebservice
import uuid

from azureml.core.authentication import AzureCliAuthentication

from dotenv import load_dotenv
# For local development, set values in this section
load_dotenv()

def check_if_deployement_is_needed(details_file):
    try:
        with open(details_file) as f:
            config = json.load(f)
        if config['run']['model_can_be_deployed'] == False:
            raise Exception("No model to be deployed!.")
    except Exception as e:
        print(e)
        sys.exit(0)

    return config

def main():
    cli_auth = AzureCliAuthentication()
    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")

    environment = os.environ.get("AML_ENV_NAME")

    config_state_folder = os.path.join(os.environ.get("ROOT_DIR"), 'config_states')
    score_script_path = os.environ.get("SCORE_SCRIPT_PATH")

    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    config = check_if_deployement_is_needed(config_state_folder + "/model_details.json")

    model = Model.deserialize(workspace=ws, model_payload=config['model'])
    
    env = Environment(environment + '-deployment')
    cd = CondaDependencies.create(
        pip_packages=['azureml-defaults','numpy', 'tensorflow']
    )

    env.python.conda_dependencies = cd
    env.register(workspace = ws)
    inference_config = InferenceConfig(entry_script=score_script_path, environment=env)


    aciconfig = AciWebservice.deploy_configuration(
        cpu_cores=1, 
        memory_gb=1, 
        tags={"data": "trafficprediction",  "method" : "keras"}, 
        description='Predict traffic'
    )

    service_name = 'trafficprediction-svc-' + str(uuid.uuid4())[:4]
    service = Model.deploy(workspace=ws, 
                        name=service_name, 
                        models=[model], 
                        inference_config=inference_config, 
                        deployment_config=aciconfig)

    service.wait_for_deployment(show_output=True)

    with open(config_state_folder + "/service_details.json", "w") as service_details:
        json.dump(service.serialize(), service_details)
    

    
    


if __name__ == '__main__':
    main()