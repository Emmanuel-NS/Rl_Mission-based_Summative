# Video checklist (5 minutes)

The assignment asks for the whole screen shared with the camera on, the file execution inside 3 of the 5 minutes, the GUI and the terminal output both visible, and the directory navigation shown. Say everything in your own words; the points below are what the rubric looks for.

## Before recording

1. Close everything except the terminal, the file explorer and the code editor. Arrange the terminal on the left and leave room on the right for the game window (1280 x 760).
2. In a fresh PowerShell: `cd` to the folder, run `uv sync` once, then clear the screen.
3. Make sure `$env:SDL_VIDEODRIVER` is not set (`Remove-Item Env:SDL_VIDEODRIVER`), otherwise the window will not appear.
4. Do one dry run of `uv run python play.py --episodes 2 --seed 1000` and check that the window shows and the terminal prints.

## Timeline

| Time | What to show and say |
| --- | --- |
| 0:00-0:40 | Camera on. Who you are, the FARUMASI mission, the problem: a rider must deliver a medicine order in Kigali inside 30 minutes, keep the box dry in rain, avoid jammed roads, and hand over prescription medicine only after checking the prescription. A wrong prescription goes back to the pharmacy. |
| 0:40-1:10 | Agent behaviour and objective: the agent chooses one of 16 actions (ride in four directions, cover the box, wait, enter, collect, park, knock, carry, verify, hand over, scan, walk back, return the medicine). Its objective is to finish the delivery quickly, with a dry box, and never hand over without a valid prescription. |
| 1:10-1:50 | Reward structure, read from `docs/ENVIRONMENT.md`: -0.03 per minute; +0.2 collect, +0.3 verify, +0.5 hand over; about +5 to +7 for a scan, more when early and dry; +2.5 for returning a wrong prescription; -6 for handing over without a valid check; -4 for a ruined box; -3 for cancellation. |
| 1:50-2:10 | Navigate the folders: show `environment/`, `training/`, `models/`, `logs/`, then say that the same environment was used for DQN, REINFORCE, PPO and A2C with TorchRL. |
| 2:10-5:00 | Run `uv run python play.py --episodes 6 --seed 1000` (about 3 minutes). Keep the game window and the terminal both in view. Talk while it runs: point at the minimap and the traffic colours, the rain and umbrella, the porch when the agent waits, the prescription check, the QR scan. Read one or two terminal lines (action, event, reward, total, minutes, box condition). When it ends, read the SUMMARY line. |

If the time allows, a final 20 seconds: show `logs/figures/test_returns.png` and say the best agent's test return against the random and the rule-following rider.

## Things to say about performance

- Say honestly what the agent does well and where it still fails. Use the numbers printed in the SUMMARY line and the test table in `docs/RESULTS.md`.
- Name the best algorithm and the setting (see `models/best_agent.json`).
- Mention the weakness the results show (see docs/RESULTS.md, for example the weakest algorithm or the invalid-prescription case) and what you would change.
