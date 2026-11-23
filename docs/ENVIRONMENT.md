# The FARUMASI rider environment

One episode is one medicine delivery in Kigali. FARUMASI sends an order from Kigali City Pharmacy; a rider has to bring it to the patient inside a 30-minute promise with the box dry, and may hand over a prescription medicine only after checking the prescription.

`environment/custom_env.py` holds the rules (a Gymnasium environment). `environment/rendering.py` draws it with pygame. `training/torchrl_env.py` wraps it for TorchRL.

## Why this needs reinforcement learning

A single prediction or a shortest-path query cannot solve it, for four reasons.

1. **Actions change the state, and the state persists.** Where the rider is, how wet the box is, whether the prescription has been checked and whether the box has been handed over all carry into the next decision.
2. **The trade-offs pay off late.** Covering the box costs a minute now and saves 0.12 of box condition per ride step later. Checking the prescription costs a minute and avoids a -6 failure. A side street adds distance and avoids a jammed main road that is worth +3 or +6 minutes. None of these shows up in the next reward.
3. **The world is stochastic and changes while the rider decides.** Rush hour jams KN 3 Rd, an incident blocks a random road at step 5-9, traffic on three roads drifts every 6 steps, and rain starts at step 3-10 in 65% of orders and lasts 6-14 steps.
4. **The rider does not see everything.** Whether a prescription is valid is hidden until it is checked (30% of prescriptions are invalid), and rain is only hinted at by dark clouds two steps ahead.

A rule-following rider (`environment/baselines.py`) shows what the task needs: it routes by live traffic with Dijkstra, covers the box when the sky darkens, checks every prescription, and takes a wrong prescription back. It scores about +5.6 per delivery. A rider that picks random legal actions scores about -4.3. The agents have to find the same behaviour from reward alone.

## Start state

`reset(seed)` draws a delivery. The rider is outside the pharmacy at junction 12 (row 2, column 0), box unloaded, condition 1.0, clock at 0 minutes.

| Random element | Distribution |
| --- | --- |
| Destination | uniform over 6 addresses: 2 homes, 2 offices, 2 hotels |
| Medicine | uniform over Metformin, Amoxicillin (prescription needed), ORS, Paracetamol |
| Prescription valid | 70% when one is required |
| Rush hour | 60% of orders (clock starts 17:05, KN 3 Rd is heavy or blocked); otherwise 11:20 |
| Road traffic | each of the 38 roads: clear 55%, busy 33%, heavy 12% |
| Rain | starts at step 3-10 in 65% of orders |
| Incident | blocks one random road at step 5-9 |

## Actions (16, discrete, with a legal-action mask)

Illegal actions are masked out of the policy and the Q-values, and an illegal choice that gets through costs -0.2.

| # | Action | Legal when | Real-world step |
| --- | --- | --- | --- |
| 0-3 | ride north, east, south, west | riding and a road exists in that direction | drive to the next junction; takes 1 minute plus 1 busy, 3 heavy, 6 blocked, plus 1 in rain |
| 4 | cover the box | riding or at the entrance, box uncovered | fit the rain cover |
| 5 | wait (take shelter) | riding, only while it rains | pull over under a porch with the bike until the rain stops; the box stays dry, the clock keeps running |
| 6 | enter the pharmacy | outside the pharmacy | walk in |
| 7 | collect the order | inside the pharmacy | pharmacist seals the box |
| 8 | park and walk in | riding, at the destination junction | park the bike |
| 9 | knock or check in | at the entrance | knock at a home gate or check in at hotel or office reception (2 min) |
| 10 | carry to the patient | inside the building | walk to the patient (2 min) |
| 11 | verify the prescription | with the patient, prescription required and unchecked | check name, medicine and date |
| 12 | hand over | with the patient, not yet handed over | give the box |
| 13 | scan the QR code | after the handover | patient confirms delivery in the FARUMASI app |
| 14 | walk back to the bike | wrong prescription found | start the return trip (2 min) |
| 15 | return the medicine | back at the pharmacy | give the medicine to the pharmacist |

## Observation (94 numbers, all scaled to 0-1)

Every value is produced by the simulator and recomputed after each decision. The last column says what the same value would be in the live FARUMASI app. Every value reaches the agent in one vector, in this order, through the same MLP input layer.

| Index | Name | Encoding and range | Simulation rule | Real-world source |
| --- | --- | --- | --- | --- |
| 0-5 | phase | one-hot of 6 delivery phases | set by the last action (enter, collect, park, knock, carry) | order status in the rider app |
| 6-7 | x, y | column / 5 and row / 3 of the junction | changes with each ride action | GPS snapped to the junction grid |
| 8-9 | to patient dx, dy | (patient - rider) offset rescaled to 0-1 | destination drawn at reset, rider position updates | geocoded delivery address |
| 10-11 | to pharmacy dx, dy | (pharmacy - rider) offset rescaled to 0-1 | pharmacy is fixed at junction 12 | pharmacy location |
| 12 | clock | minutes used / 55 | each action adds its minutes | order accepted timestamp |
| 13 | raining | 0 or 1 | `rain_at <= steps < rain_end` | weather service |
| 14 | rain soon | 0 or 1 | rain starts within 2 steps | weather nowcast |
| 15 | box covered | 0 or 1 | set by the cover action, cleared at handover | rider app toggle |
| 16 | box condition | 0 to 1 | drops 0.12 per ride step, 0.07 per other step, when uncovered in rain and not sheltering | humidity sensor in the box |
| 17 | prescription required | 0 or 1 | drawn from the medicine | order system |
| 18 | prescription checked | 0 or 1 | set by verify | rider app |
| 19-20 | prescription known ok / known wrong | 0 or 1 each | visible only after verify | rider's check result |
| 21-23 | address kind | one-hot: home, office, hotel | drawn at reset | order |
| 24 | returning | 0 or 1 | set by walk back to the bike | rider app |
| 25 | handed over | 0 or 1 | set by hand over | rider app |
| 26 | entrance cleared | 0 or 1 | set by knock or check in | rider app |
| 27 | steps | decisions / 80 | +1 per decision | app counter |
| 28-31 | medicine | one-hot of 4 medicines | drawn at reset | order |
| 32-35 | traffic around the junction | level / 3 on the road leading north, east, south, west | traffic table | live traffic service |
| 36-39 | compass bearing | 1 if riding north, east, south, west gets closer to the destination (patient, or the pharmacy when returning) | grid distance to the target | phone compass and delivery address |
| 40-77 | traffic on every road | level / 3 on all 38 roads | random start, drift every 6 steps, one incident, jam on KN 3 Rd at rush hour | live traffic service, 4 bands |
| 78-93 | legal-action mask | 0 or 1 for each of 16 actions | derived from phase and flags | app business rules |

`uv run python main.py --describe` prints the same list from the code.

## Rewards

| Event | Reward |
| --- | --- |
| every minute used | -0.03 |
| riding closer to the target by traffic-aware time | +0.12 per minute saved (shaping, negative when riding away) |
| order collected | +0.2 |
| prescription checked | +0.3 |
| handed over | +0.5 |
| QR scan, delivered | (5 + up to 2 for being early) x (0.6 + 0.4 x box condition) - 0.12 per minute past 30 |
| wrong prescription returned to the pharmacy | +2.5 |
| rain on an uncovered box | box condition falls 0.12 (riding) or 0.07 (other actions, except sheltering) and the same amount is subtracted |
| hand over a prescription medicine without a valid check | -6 and the episode ends |
| box condition at or below 0.3 | -4 and the episode ends |
| illegal action | -0.2 and a minute passes |
| order cancelled at 55 minutes | -3 and the episode ends |
| 80 steps reached | -2 and the episode ends |

## Terminal conditions

| Outcome | How it ends | Counts as success |
| --- | --- | --- |
| delivered | QR scan | yes |
| returned | wrong prescription handed back to the pharmacy | yes |
| violation | handed over without a valid prescription check | no |
| ruined | box condition at or below 0.3 | no |
| cancelled | 55 minutes used | no |
| abandoned | 80 decisions without finishing | no |

## Visualisation

`environment/rendering.py` (pygame, 30 fps) shows a top-down Kigali with lanes, traffic lights, a bike lane, simulated cars and a live traffic overlay, and a minimap. The pharmacy, gate, lobby, porch and handover are side-on scenes. Rain, an umbrella, the rain cover on the box and the QR scan are drawn. Screenshots are in `assets/`.

## Using the trained agent from another system

`uv run python serve.py` serves the best checkpoint as a JSON API. The rider app posts the delivery state (the fields in the `/schema` response) to `/act` and receives the recommended action with the legal actions and the policy's scores. `uv run python -m service.export` writes a TorchScript file and a plain JSON copy of the weights for an on-device implementation. See `service/` and `tests/test_service.py`.
