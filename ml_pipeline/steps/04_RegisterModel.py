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
from azureml.core.authentication import AzureCliAuthentication

from dotenv import load_dotenv
# For local development, set values in this section
load_dotenv()

def check_if_registration_is_needed(details_file):
    try:
        with open(details_file) as f:
            config = json.load(f)
        if config['model_can_be_deployed'] == False:
            raise Exception("No new model to register as production model performed better in our evaluation.")
    except Exception as e:
        print(e)
        sys.exit(0)

    return config

def register_model(model_name, model_file, description, run):

    model = run.register_model(
        model_name=model_name,
        model_path='outputs/' + model_file,
        tags={"runId": run.id},
        description=description,
    )

    print(
        "Model registered: {} \nModel Description: {} \nModel Version: {}".format(
            model.name, model.description, model.version
        )
    )
    return model



def main():
    cli_auth = AzureCliAuthentication()

    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    model_name = os.environ.get("MODEL_NAME")
    model_file = os.environ.get("MODEL_FILE")
    model_dir = os.environ.get("MODEL_DIR")
    model_description = os.environ.get("MODEL_DESCRIPTION")
    experiment_name = os.environ.get("EXPERIMENT_NAME")

    config_state_folder = os.path.join(os.environ.get("ROOT_DIR"), 'config_states')

    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    config = check_if_registration_is_needed(config_state_folder + "/training-run.json")
    exp = Experiment(workspace=ws, name=experiment_name)
    run = Run(experiment=exp, run_id=config['runId'])
    model = register_model(model_name, model_file, model_description, run)


    model_json = {}
    model_json["model"] = model.serialize()
    model_json["run"] = config

    print(model_json)

    with open(config_state_folder + "/model_details.json", "w") as model_details:
        json.dump(model_json, model_details)
    

    
    


if __name__ == '__main__':
    main()