"""
Inference server for the PolyUMI visuomotor diffusion policy.

Serves a trained checkpoint over the same HTTP contract the ROS-side ``policy_client_node``
already speaks to ``inference_server/dummy_server.py``:

    POST /predict_cartesian/
      {n_obs_steps, n_action_steps,
       observations: {image: {dtype, shape, data(b64)}, agent_pos: [[8]]}}
    -> {actions: [[8]], n_action_steps}

    POST /reset  {agent_pos: [8]}          # cache the episode-start EEF pose (see below)
    GET  /health

Run it inside the training container (``docker/serve.sh``) — that is the whole point of using one
image for both roles: the checkpoint is dill-pickled and must unpickle against the exact dependency
tree it was trained with, and the ``umi`` conda env has both ``diffusion_policy``/torch and
fastapi/uvicorn, so this process **direct-imports** the policy (no subprocess).

The wire contract is unchanged from the dummy server (absolute EEF poses, quaternion), so nothing
changes on the ROS side but the URL. Two frame conversions happen here (see ``serve_obs.py``):
  - obs: absolute wire poses -> UMI's relative, rot6d, name-matched obs dict.
  - action: the policy's relative chunk -> absolute EEF targets (``convert_pose_mat_rep`` backward).

Episode-start pose: the policy consumes ``robot0_eef_rot_axis_angle_wrt_start`` — orientation
relative to where the episode began. The wire ``agent_pos`` only carries the *current* pose, so the
client must ``POST /reset`` with the start pose once per rollout; it is cached here. Absent a reset,
``/predict_cartesian/`` falls back to the current pose (``wrt_start`` -> identity) and warns.
"""

import base64
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serve_obs import (
    actions_rel_to_abs,
    agent_pos_to_pose6,
    agent_pos_to_pose_mat,
    wire_to_obs_dict,
)

logger = logging.getLogger('serve_policy')

AGENT_POS_DIM = 8
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


class ResetRequest(BaseModel):
    """Body for /reset — one wire pose captured at the start of the rollout."""

    agent_pos: list[float]  # a single [8] pose [x,y,z,qx,qy,qz,qw,gripper]


def _load_policy(ckpt_path: str):
    """
    Load the dill-pickled, self-describing checkpoint.

    Returns ``(policy, device)``. Mirrors ``base_workspace.load_payload`` + ``train.py`` — the
    config travels inside the checkpoint, so only a path is needed.
    """
    import dill
    import hydra
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload)
    policy = workspace.ema_model  # EMA weights — NOT workspace.model (eval uses EMA)
    policy.to(device)
    policy.eval()
    return policy, device


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Fail loudly at startup if the checkpoint mount is misconfigured — a server that "looks
    # healthy" but can't serve is worse than one that never starts.
    ckpt_path = os.environ.get('CKPT_PATH')
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise RuntimeError(
            f'CKPT_PATH must point to a checkpoint file; got {ckpt_path!r}. '
            'Set -e CKPT_PATH=/data/checkpoints/<name>.ckpt and mount the checkpoint dir.'
        )
    app.state.ckpt_path = ckpt_path
    app.state.policy, app.state.device = _load_policy(ckpt_path)
    # Episode-start pose (6-vec pos+rotvec), set via POST /reset. None -> current-pose fallback.
    app.state.demo_start_pose6 = None
    logger.info('loaded policy from %s on %s', ckpt_path, app.state.device)
    yield


app = FastAPI(title='PolyUMI Inference Server', lifespan=_lifespan)


@app.get('/health')
def health() -> dict:
    """Liveness/readiness check; reports the checkpoint, device, and whether /reset has run."""
    ready = getattr(app.state, 'policy', None) is not None
    return {
        'status': 'ready' if ready else 'loading',
        'checkpoint': getattr(app.state, 'ckpt_path', None),
        'device': getattr(app.state, 'device', None),
        'episode_start_set': getattr(app.state, 'demo_start_pose6', None) is not None,
    }


@app.post('/reset')
def reset(req: ResetRequest) -> dict:
    """Cache the episode-start EEF pose. Call once at the start of each rollout."""
    if len(req.agent_pos) != AGENT_POS_DIM:
        raise HTTPException(
            status_code=422, detail=f'agent_pos must have length {AGENT_POS_DIM}'
        )
    app.state.demo_start_pose6 = agent_pos_to_pose6(np.asarray(req.agent_pos))
    return {'status': 'ok', 'episode_start_set': True}


def _decode_obs(req: PredictRequest) -> tuple[np.ndarray, np.ndarray]:
    """Validate + decode the wire observation into (image [To,H,W,3], agent_pos [To,8])."""
    missing = REQUIRED_OBS_KEYS - req.observations.keys()
    if missing:
        raise HTTPException(status_code=422, detail=f'Missing observation keys: {sorted(missing)}')

    image = req.observations['image']
    if not isinstance(image, dict) or not {'dtype', 'shape', 'data'} <= image.keys():
        raise HTTPException(
            status_code=422, detail="image must be a dict with 'dtype', 'shape', 'data'"
        )
    try:
        image_arr = np.frombuffer(
            base64.b64decode(image['data']), dtype=np.dtype(image['dtype'])
        ).reshape(image['shape'])
    except Exception as e:  # noqa: BLE001 - surface any decode failure as a 422
        raise HTTPException(status_code=422, detail=f'Failed to decode image: {e}') from e
    if image_arr.shape[0] != req.n_obs_steps:
        raise HTTPException(
            status_code=422,
            detail=f'image leading dim must be n_obs_steps={req.n_obs_steps}, got {image_arr.shape[0]}',
        )

    agent_pos = req.observations['agent_pos']
    if (
        not isinstance(agent_pos, list)
        or len(agent_pos) != req.n_obs_steps
        or not all(isinstance(row, list) and len(row) == AGENT_POS_DIM for row in agent_pos)
    ):
        raise HTTPException(
            status_code=422,
            detail=f'agent_pos must have shape [{req.n_obs_steps}, {AGENT_POS_DIM}]',
        )
    return image_arr, np.asarray(agent_pos, dtype=np.float64)


@app.post('/predict_cartesian/', response_model=PredictResponse)
def predict_cartesian(req: PredictRequest) -> PredictResponse:
    """Run the policy on one observation window and return an absolute EEF action chunk."""
    import torch

    image_arr, agent_pos = _decode_obs(req)

    start6 = app.state.demo_start_pose6
    if start6 is None:
        logger.warning(
            'no episode start set (POST /reset) — approximating '
            'robot0_eef_rot_axis_angle_wrt_start with the current pose'
        )

    obs_np = wire_to_obs_dict(image_arr, agent_pos, demo_start_pose6=start6)
    obs_dict = {k: torch.from_numpy(v).to(app.state.device) for k, v in obs_np.items()}

    with torch.no_grad():
        action_pred = app.state.policy.predict_action(obs_dict)['action_pred']
    action_pred = action_pred[0].detach().cpu().numpy()  # [Ta, 10] relative to current pose

    # The current EEF pose (agent_pos[-1]) is the base the policy's chunk is relative to.
    base_pose_mat = agent_pos_to_pose_mat(agent_pos)[-1]
    actions_abs = actions_rel_to_abs(action_pred, base_pose_mat)  # [Ta, 8]

    # Return at most the requested count; further truncation is the client's job (UMI's policy
    # emits the full horizon with no offset).
    n_return = min(req.n_action_steps, actions_abs.shape[0])
    return PredictResponse(actions=actions_abs[:n_return].tolist(), n_action_steps=n_return)
