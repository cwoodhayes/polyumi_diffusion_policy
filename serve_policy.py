"""
Inference server for the PolyUMI visuomotor diffusion policy.

Serves a trained checkpoint over the same HTTP contract the ROS-side ``policy_client_node``
already speaks to ``inference_server/dummy_server.py``:

    POST /predict_cartesian/   Content-Type: application/octet-stream
      one binary frame: [4B header length][JSON header][channel blobs]   -- see obs_wire
      channels: camera0_rgb [To,H,W,3] uint8, agent_pos [To,8] float64
    -> {actions: [[8]], n_action_steps, server_total_ms, model_ms}

    ``camera0_rgb`` is uint8 -- what the dataset stores and what the client sends; a float array
    already normalized to [0, 1] is accepted too (see ``serve_obs``). The frame format can carry
    any subset of channels, for modalities that update slower than the control loop, but a request
    omitting a required one is REFUSED rather than filled in; ``obs_wire.require_channels``
    explains why.

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

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel

from obs_wire import WireFormatError, require_channels, unpack_observation
from serve_obs import (
    actions_rel_to_abs,
    agent_pos_to_pose6,
    agent_pos_to_pose_mat,
    wire_to_obs_dict,
)

# A child of uvicorn's own logger, NOT a bare getLogger('serve_policy'). Uvicorn configures
# handlers for the 'uvicorn*' loggers only and leaves the root logger bare, so a top-level logger
# here propagates to a root with no handler and every INFO line is silently discarded — which is
# why 'loaded policy from ...' has never appeared in the container output. Hanging off
# 'uvicorn.error' inherits uvicorn's handler and format, so these lines interleave with the
# access log instead of needing a --log-config of their own.
logger = logging.getLogger('uvicorn.error').getChild('serve_policy')

AGENT_POS_DIM = 8
#: Channels the policy cannot run without. Named for the dataset's own fields, so wiring a new
#: modality is adding a name here and in shape_meta rather than reshaping the request.
REQUIRED_CHANNELS = ('camera0_rgb', 'agent_pos')


class PredictResponse(BaseModel):
    """Response body for /predict_cartesian/."""

    actions: list[list[float]]
    n_action_steps: int
    #: Wall time this process spent on the request, in ms — the same number the access log
    #: prints. The client subtracts it from its own round trip to get network + serialization,
    #: which is the only way to tell a slow link from a busy box without instrumenting both.
    #: Nullable so a client can distinguish "the server did not say" from "it was zero".
    server_total_ms: float | None = None
    #: The forward pass alone, in ms, measured through the .cpu() sync point.
    model_ms: float | None = None


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
    with open(ckpt_path, 'rb') as f:
        payload = torch.load(f, pickle_module=dill, map_location='cpu')
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


@app.middleware('http')
async def _log_request_time(request: Request, call_next):
    """
    Log how long each request took to serve, split into total and model time.

    The client already measures its own round trip (``inference=NNNms`` in policy_client_node's
    action-chunk line), but that number bundles network, serialization and compute together, so a
    slow tick is indistinguishable from a slow link. Uvicorn's access log can't close the gap —
    its message format is fixed and carries no duration — so time it here.

    Two numbers because they fail for different reasons and have different fixes. ``total`` minus
    ``model`` is base64 decode + JSON + FastAPI overhead, which scales with the observation
    payload; ``model`` is GPU work, which on a shared box scales with whoever else is running.
    Read against the client's round trip, the pair separates all three: if total ~ the client's
    number, the link is fine and the box is the problem.
    """
    t0 = time.perf_counter()
    # Also handed to the endpoint, which puts it in the response body: the log is for a human
    # reading this box, the wire field is for the client plotting the split live.
    request.state.t_request_start = t0
    response = await call_next(request)
    total_ms = (time.perf_counter() - t0) * 1e3
    model_ms = getattr(request.state, 'model_ms', None)
    model = f', model {model_ms:.0f}ms' if model_ms is not None else ''
    logger.info(
        '%s %s -> %d in %.0f ms%s', request.method, request.url.path,
        response.status_code, total_ms, model,
    )
    return response


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
    # A malformed pose (e.g. a zero-norm quaternion) makes Rotation.from_quat raise; that's a
    # bad request, not a server fault, so surface it as 422 rather than letting it 500.
    try:
        pose6 = agent_pos_to_pose6(np.asarray(req.agent_pos, dtype=np.float64))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f'Invalid agent_pos: {e}') from e
    app.state.demo_start_pose6 = pose6
    return {'status': 'ok', 'episode_start_set': True}


def _decode_obs(body: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Decode one request frame into (camera0_rgb [To,H,W,3], agent_pos [To,8], n_action_steps).

    Every rejection is a 422: a frame this server cannot read is a bad request, not a fault here.
    """
    try:
        channels, header = unpack_observation(body)
        require_channels(channels, REQUIRED_CHANNELS)
    except WireFormatError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    image_arr = channels['camera0_rgb']
    agent_pos = channels['agent_pos']
    n_obs_steps = header.get('n_obs_steps')

    # The header's window length and the arrays' leading dim are two independent claims about the
    # same thing; a mismatch means the client packed something other than what it says it packed.
    for name, arr in (('camera0_rgb', image_arr), ('agent_pos', agent_pos)):
        if arr.shape[0] != n_obs_steps:
            raise HTTPException(
                status_code=422,
                detail=f'{name} leading dim must be n_obs_steps={n_obs_steps}, got {arr.shape[0]}',
            )
    if image_arr.ndim != 4 or image_arr.shape[-1] != 3:
        raise HTTPException(
            status_code=422, detail=f'camera0_rgb must be [To,H,W,3], got {list(image_arr.shape)}'
        )
    if agent_pos.ndim != 2 or agent_pos.shape[1] != AGENT_POS_DIM:
        raise HTTPException(
            status_code=422,
            detail=f'agent_pos must be [To,{AGENT_POS_DIM}], got {list(agent_pos.shape)}',
        )

    n_action_steps = header.get('n_action_steps')
    if not isinstance(n_action_steps, int) or n_action_steps < 1:
        raise HTTPException(
            status_code=422, detail=f'n_action_steps must be a positive int, got {n_action_steps!r}'
        )
    # float64 because agent_pos_to_pose_mat builds rotations from it; the wire dtype is the
    # client's business, the precision the pose maths needs is ours.
    return image_arr, np.asarray(agent_pos, dtype=np.float64), n_action_steps


@app.post('/predict_cartesian/', response_model=PredictResponse)
def predict_cartesian(
    request: Request,
    body: Annotated[bytes, Body(media_type='application/octet-stream')],
) -> PredictResponse:
    """Run the policy on one observation window and return an absolute EEF action chunk."""
    import torch

    image_arr, agent_pos, n_action_steps = _decode_obs(body)

    start6 = app.state.demo_start_pose6
    if start6 is None:
        logger.warning(
            'no episode start set (POST /reset) — approximating '
            'robot0_eef_rot_axis_angle_wrt_start with the current pose'
        )

    obs_np = wire_to_obs_dict(image_arr, agent_pos, demo_start_pose6=start6)
    obs_dict = {k: torch.from_numpy(v).to(app.state.device) for k, v in obs_np.items()}

    # Timed through the .cpu() call, not just predict_action: CUDA kernels launch asynchronously,
    # so stopping the clock at the end of the `with` block would measure queueing, not diffusion.
    # The copy back to host is the synchronization point, and thus the honest end of the work.
    t_model = time.perf_counter()
    with torch.no_grad():
        action_pred = app.state.policy.predict_action(obs_dict)['action_pred']
    action_pred = action_pred[0].detach().cpu().numpy()  # [Ta, 10] relative to current pose
    request.state.model_ms = (time.perf_counter() - t_model) * 1e3

    # The current EEF pose (agent_pos[-1]) is the base the policy's chunk is relative to.
    base_pose_mat = agent_pos_to_pose_mat(agent_pos)[-1]
    actions_abs = actions_rel_to_abs(action_pred, base_pose_mat)  # [Ta, 8]

    # Return at most the requested count; further truncation is the client's job (UMI's policy
    # emits the full horizon with no offset).
    n_return = min(n_action_steps, actions_abs.shape[0])
    t_start = getattr(request.state, 't_request_start', None)
    return PredictResponse(
        actions=actions_abs[:n_return].tolist(),
        n_action_steps=n_return,
        server_total_ms=None if t_start is None else (time.perf_counter() - t_start) * 1e3,
        model_ms=request.state.model_ms,
    )
