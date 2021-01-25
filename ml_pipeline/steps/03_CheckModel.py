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


def get_production_model(ws, exp):
        # Get most recently registered model, we assume that is the model in production. Download this model and compare it with the recently trained model by checking their MSE scores
    model_list = Model.list(ws)
    production_model = next(
        filter(
            lambda x: x.created_time == max(model.created_time for model in model_list),
            model_list,
        )
    )
    production_model_run_id = production_model.tags.get("runId")

    # Get the run history for both production model and newly trained model and compare the logged mse
    production_model_run = Run(exp, run_id=production_model_run_id)

    return production_model_run

def get_new_model(config_file, ws, exp):
    with open(config_file) as training_run_json:
        run_config = json.load(training_run_json)

    new_model_run_id = run_config["runId"]
    new_model_run = Run(exp, run_id=new_model_run_id)

    return new_model_run


def main():
    cli_auth = AzureCliAuthentication()

    workspace_name = os.environ.get("WORKSPACE_NAME")
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")

    config_state_folder = os.path.join(os.environ.get("ROOT_DIR"), 'config_states')

    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )
    exp = Experiment(workspace=ws, name=experiment_name)

    
    promote_new_model = False
    new_run = get_new_model(config_state_folder + '/training-run.json', ws, exp)
    
    try: # Ask forgiveness not permission: https://stackoverflow.com/questions/12265451/ask-forgiveness-not-permission-explain
        production_run = get_production_model(ws, exp)
        production_model_mse = production_run.get_metrics().get("mse")
        new_model_mse = new_run.get_metrics().get("mse")
        print(
            "Current Production model mse: {}, New trained model mse: {}".format(
                production_model_mse, new_model_mse
            )
        )
        if new_model_mse < production_model_mse:
            promote_new_model = True
            print("New trained model performs better, thus it will be registered")
        
    except:
        promote_new_model = True
        print("This is the first model to be trained, thus nothing to evaluate for now")
    
    run_details = {k:v for k,v in new_run.get_details().items() if k in ['runId', 'target', 'properties']}
    run_details["model_can_be_deployed"] = promote_new_model

    with open(config_state_folder + "/training-run.json", "w") as training_run_json:
        json.dump(run_details, training_run_json)



if __name__ == '__main__':
    main()