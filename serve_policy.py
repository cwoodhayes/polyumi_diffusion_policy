"""
Inference server for the PolyUMI visuomotor diffusion policy (SCAFFOLD).

Serves a trained checkpoint over the same HTTP contract the ROS-side ``policy_client_node``
already speaks to ``inference_server/dummy_server.py``:

    POST /predict_cartesian/
      {n_obs_steps, n_action_steps,
       observations: {image: {dtype, shape, data(b64)}, agent_pos: [[8]]}}
    -> {actions: [[...]], n_action_steps}

Running this in the training container (docker/serve.sh) is the whole point of using one image
for both roles: the checkpoint is dill-pickled and must unpickle against the exact dep tree it
was trained with.

STATUS: scaffold. The transport, models, and lifespan are real and runnable (so ROS-side
wiring can be tested end to end), but ``/predict_cartesian/`` returns HTTP 501 until the model
path is filled in. That work is deferred to the post-training inference step and needs three
pieces, none of which is a Docker concern:

  1. Load the checkpoint and take ``workspace.ema_model`` (NOT ``workspace.model`` — eval uses
     the EMA weights).
  2. Translate the wire observation into UMI's obs dict. The wire format is a flat 8-vector
     ``agent_pos`` [x,y,z,qx,qy,qz,qw,gripper] plus one stacked image blob; ``UmiDataset``'s
     model expects ``camera0_rgb``, ``robot0_eef_pos``, ``robot0_eef_rot_axis_angle`` (rotvec),
     ``robot0_gripper_width``. This is the API-contract change flagged in
     docs/franka-inference-bringup.md.
  3. Run the policy and convert its relative-pose action chunk back to absolute EEF targets
     (``convert_pose_mat_rep(..., backward=True)``), returning the chunk the client expects.
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REQUIRED_OBS_KEYS = {'image', 'agent_pos'}


class PredictRequest(BaseModel):
    """Request body for /predict_cartesian/ — mirrors the dummy server's contract."""

    n_obs_steps: Annotated[int, Field(ge=1)] = 2
    n_action_steps: Annotated[int, Field(ge=1)] = 1
    observations: dict


class PredictResponse(BaseModel):
    """Response body for /predict_cartesian/."""

    actions: list[list[float]]
    n_action_steps: int


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # CKPT_PATH is where the model WILL be loaded from once step (1) above is implemented.
    # Only recorded here (not validated) so the scaffold still boots without a checkpoint —
    # that's deliberate: it lets ROS-side wiring be tested against the live transport before any
    # model exists. The present-and-exists check belongs with the model-loading code in step (1),
    # which is where a missing/bad path should fail loudly.
    app.state.ckpt_path = os.environ.get('CKPT_PATH')
    yield


app = FastAPI(title='PolyUMI Inference Server (scaffold)', lifespan=_lifespan)


@app.get('/health')
def health() -> dict:
    """Liveness check; also reports whether a checkpoint path is configured."""
    return {'status': 'ok', 'ckpt_path': app.state.ckpt_path, 'model_loaded': False}


@app.post('/predict_cartesian/', response_model=PredictResponse)
def predict_cartesian(req: PredictRequest) -> PredictResponse:
    """Not yet implemented — see module docstring for the three deferred pieces."""
    missing = REQUIRED_OBS_KEYS - req.observations.keys()
    if missing:
        # sorted() so the message is deterministic (set ordering is not) — keeps client-side
        # assertions/tests stable.
        detail = f'Missing observation keys: {sorted(missing)}'
        raise HTTPException(status_code=422, detail=detail)
    raise HTTPException(
        status_code=501,
        detail=(
            'serve_policy is a scaffold: model loading + obs translation + rel→abs action '
            'conversion are not implemented yet (see module docstring). Use dummy_server for '
            'ROS-side bringup in the meantime.'
        ),
    )
