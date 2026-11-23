import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch

from environment.custom_env import DESTINATIONS, ENTER, COLLECT, OBS_DIM, FarumasiRiderEnv
from service.agent import RiderAgent, state_to_observation
from service.server import make_handler
from training.common import make_policy_net


def state_of(env: FarumasiRiderEnv) -> dict:
    return {
        "phase": int(env.phase),
        "node": int(env.node),
        "destination": DESTINATIONS.index(env.dest),
        "medicine": int(env.med),
        "minutes": float(env.minutes),
        "steps": int(env.steps),
        "raining": bool(env.raining),
        "rain_soon": bool(env.rain_soon),
        "box_covered": bool(env.covered),
        "box_condition": float(env.condition),
        "rx_required": bool(env.rx_required),
        "rx_checked": bool(env.rx_checked),
        "rx_valid": bool(env.rx_valid),
        "returning": bool(env.returning),
        "handed_over": bool(env.handed),
        "entrance_cleared": bool(env.cleared),
        "traffic": [int(v) for v in env.traffic],
    }


def test_json_state_rebuilds_the_training_observation():
    env = FarumasiRiderEnv()
    observation, _ = env.reset(seed=11)
    for action in (ENTER, COLLECT, 1, 1):
        observation, *_ = env.step(action)
    rebuilt = state_to_observation(state_of(env))
    assert np.allclose(rebuilt, observation)


@pytest.fixture()
def agent(tmp_path: Path) -> RiderAgent:
    net = make_policy_net(32)
    return RiderAgent(net, {"algo": "test", "experiment": 0})


def test_agent_never_recommends_an_illegal_action(agent: RiderAgent):
    env = FarumasiRiderEnv()
    env.reset(seed=3)
    answer = agent.recommend(state=state_of(env))
    assert answer["action_name"] in answer["legal_actions"]
    assert set(answer["legal_actions"]) == {"enter_pharmacy"}
    assert abs(sum(answer["scores"].values()) - 1.0) < 1e-3


def test_http_api_round_trip(agent: RiderAgent):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(agent))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/health") as response:
            assert json.load(response)["status"] == "ok"
        env = FarumasiRiderEnv()
        env.reset(seed=5)
        request = urllib.request.Request(
            base + "/act",
            data=json.dumps({"state": state_of(env)}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            body = json.load(response)
        assert body["action_name"] == "enter_pharmacy"
        bad = urllib.request.Request(base + "/act", data=json.dumps({"observation": [0, 1]}).encode())
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_exported_json_policy_matches_the_network(tmp_path: Path):
    from service.export import export

    net = make_policy_net(16)
    checkpoint = tmp_path / "agent.pt"
    torch.save({"algo": "test", "hidden": 16, "state_dict": net.state_dict(), "config": {"id": 0}}, checkpoint)
    _script, json_path = export(str(checkpoint))
    spec = json.loads(json_path.read_text())
    x = np.random.default_rng(0).random(OBS_DIM).astype(np.float32)
    for layer in spec["layers"]:
        x = np.asarray(layer["weight"], dtype=np.float32) @ x + np.asarray(layer["bias"], dtype=np.float32)
        if layer["activation"] == "relu":
            x = np.maximum(x, 0.0)
    expected = net(torch.as_tensor(np.random.default_rng(0).random(OBS_DIM).astype(np.float32)).unsqueeze(0)).detach().numpy()[0]
    assert np.allclose(x, expected, atol=1e-4)
