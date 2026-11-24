"""Text that discusses the measured results (environment rider-delivery-4).

Every number below comes from docs/RESULTS.md and logs/test_evaluation.json. If the experiments are rerun,
check each number against docs/RESULTS.md before using the report.
"""

from narrative import TEXT

TEXT["hp_dqn"] = (
    "DQN was the least sensitive: all ten settings finished with a positive test return (+4.14 to +6.02, mean +5.29) and seven of them reached +5 or more. "
    "The best validation setting was a myopic gamma of 0.9 (exp 4, validation +6.01, test +6.02, 98% success). A far-sighted gamma of 0.999 (exp 5, +4.62) "
    "was worse, which fits an episode of about 15 decisions with a dense progress reward: there is little to gain from looking far ahead. "
    "The weakest settings were the plain DQN with a wide network, no double-Q and an L2 loss (exp 10, +4.14, 79%) and the fastest learning rate (exp 3, +4.33). "
    "Epsilon schedules, replay size and the target-update interval changed the result by less than one return point."
)

TEXT["hp_reinforce"] = (
    "REINFORCE was the weakest method. Eight of ten settings ended between -2.15 and -1.32 on the test deliveries (21-27% success), whatever was changed: "
    "gamma, entropy bonus, learning rate, network size and even the removal of the baseline (exp 2, -1.54, against -2.15 with the baseline) gave the same plateau, "
    "so with one seed per setting the baseline shows no measurable benefit. Only two settings escaped: shorter rollouts of 128 frames (exp 9, +1.70, 55%), "
    "which give more updates per frame, and a large learning rate of 0.005 (exp 4, +0.19). Monte Carlo returns over a whole delivery are noisy, and "
    "30,000 frames is only about 1,000 to 2,000 deliveries."
)

TEXT["hp_ppo"] = (
    "PPO was robust: six of ten settings reached +5 or more and the clip range and entropy bonus made little difference "
    "(clip 0.1, 0.2 and 0.3 gave +5.75, +4.79 and +5.62). Two controls mattered. A single optimisation epoch per batch (exp 8) "
    "failed completely (-1.52, 27%), while ten epochs with small minibatches (exp 9) gave +5.94: PPO's advantage is that it reuses each batch, "
    "so it gets many more gradient steps per frame. A long GAE horizon of 0.99 (exp 7) was clearly worse (+2.85) than 0.8 (exp 6, +6.05), "
    "probably because a high lambda adds variance to the advantage on this short task. The best validation score came from exp 10 "
    "(fast learning rate, short rollout; validation +6.00, test +5.77)."
)

TEXT["hp_a2c"] = (
    "A2C had the widest spread. Two settings learned the task: a short rollout of 64 frames with GAE lambda 0.9 (exp 9, +5.95, 97%) and "
    "a large learning rate of 0.003 (exp 3, +4.93, 81%). A loose gradient clip (exp 7, +1.62) and removing the entropy bonus (exp 4, +0.08) "
    "gave partial policies, and the other six stayed near -1.2 (about 27% success). As with REINFORCE and PPO, the common factor "
    "of the successful settings is more or faster parameter updates per frame. A first version of exp 6 and 7 varied the value-loss coefficient, "
    "which had no measurable effect in this implementation (the critic is a separate network, so its loss weight only rescales its own gradient); those two "
    "experiments were replaced by the network size and the gradient clip above."
)

TEXT["results_intro"] = (
    "The rule-following rider scores +6.54 on the test deliveries and random legal actions -4.14, so the useful range is about 10.7 return points. "
    "Figure 2 shows the best run of each algorithm in the sweep and Figure 3 all ten runs of each. After 30,000 frames the best run of "
    "DQN, PPO and A2C is within 0.6 of the rule-following rider, while the best REINFORCE run (+1.70) is far behind. Every figure below is generated from the saved logs by main.py."
)

TEXT["convergence"] = (
    "In Figure 2 DQN and PPO reach a return above +5 within 10,000 to 15,000 frames and then stay there; A2C needs 25,000 frames; REINFORCE only starts to improve after 20,000. "
    "Figure 3 shows what the ten settings do on average: PPO's mean training return climbs smoothly to about +4, DQN's mean reaches about +4.3, A2C's mean about +1 and REINFORCE's stays near 0, "
    "with wide bands because the setting matters a lot for the on-policy methods. All curves are still rising for the mean setting when the sweep stops, so the long runs in section 5.1 are needed."
)

TEXT["loss_entropy"] = (
    "The DQN loss (Figure 4, the best DQN setting) stays between about 0.02 and 0.17 with no downward trend and a few spikes, for instance near 25,000 frames. "
    "This is expected for Q-learning: the targets move as the policy improves and the replay buffer keeps changing, so the TD error is not driven to zero, "
    "and the return curve is the better guide to progress. Policy entropy (Figure 5, best run of each policy-gradient method) starts near 1.15, the entropy of "
    "an almost uniform choice among the legal actions, and falls at different speeds. PPO falls fastest, to about 0.1 by 20,000 frames, and this is a healthy collapse: "
    "the policy became nearly deterministic after it had found a good behaviour (its validation return was +6). A2C falls to about 0.3 and is noisy because it updates on "
    "64-frame batches. REINFORCE stays highest, between 0.5 and 0.8, and is also the method that learned least: the entropy is still high because its updates are noisy and "
    "its return was still low."
)

TEXT["final_intro"] = (
    "The sweep ranks settings after only 30,000 frames. The setting with the best validation return for each algorithm was trained for "
    "120,000 frames on two seeds, and every final checkpoint was scored on 100 unseen deliveries (seeds 5000-5099). The settings were DQN with gamma 0.9, PPO with a fast learning rate and "
    "short rollout, A2C with a short rollout and REINFORCE with a short rollout. Table 5.1 lists the result. The two baselines show the range: "
    "random legal actions score -4.14 and the rule-following rider +6.54."
)

TEXT["final_discussion"] = (
    "DQN, PPO and A2C all end close to the rule-following rider: +6.24 (both DQN seeds), +6.21 (PPO mean) and +6.00 (A2C mean), which is 97%, 97% and 95% of the way "
    "from random to the rule-following rider. The differences among these three are smaller than the standard error of a single run (0.1 to 0.25) and than the gap between seeds "
    "of A2C (+5.69 and +6.32), so the honest ranking is a tie between DQN and PPO, with A2C close behind. They differ in speed: PPO first reaches a validation return of +5 after "
    "10,000 to 15,000 frames and DQN after 20,000, while A2C needed 25,000 frames on one seed and 105,000 on the other (Figure 6, where A2C's mean sits at about +2 for most of the run). "
    "REINFORCE did not converge to the same level: +0.64 and +4.93, 42% and 81% success. Its second seed is a good policy, so the method can learn the task, but it is slow and "
    "depends strongly on the seed. None of the agents beats the rule-following rider; they rediscover its behaviour from reward alone."
)

TEXT["behaviour"] = (
    "Figure 9 shows the share of decisions on each action. DQN, PPO and A2C look like the rule-following rider: mostly riding east towards the destination, then the "
    "same chain (enter, collect, park, knock, carry, hand over, scan). They check the prescription in 3.3-3.5% of decisions, the same as the rule-following rider, and they cover the box "
    "(rule-following rider 5.3%, PPO and A2C 7.6%, DQN 3.5-4.3%). None of them uses WAIT (0-1.7% of decisions); sheltering under a porch is legal while it rains, "
    "but riding on with a covered box is cheaper than waiting for the rain to stop, so no agent learned to wait. "
    "Figure 8 splits the test deliveries by condition. DQN seed 0 succeeds in every condition; the others are at 89-100% in the dry, in rain and at rush hour, and the "
    "hardest case is an invalid prescription, which needs the rider to verify, refuse to hand over, walk back and return the medicine (A2C seed 0 79%, PPO seed 0 89%, "
    "the other long runs 95-100%). REINFORCE shows the failure clearly: seed 1 succeeds on every order without a prescription or with a valid one (100%) and on none of the 19 invalid prescriptions (0%): "
    "it never learned the return path, and those orders end by cancellation."
)

TEXT["outcomes"] = (
    "Figure 10 shows how the test deliveries ended. The rule-following rider delivers 81 and returns 19. DQN seed 0 does the same with no failure (81 delivered, 19 returned), "
    "and the other DQN, PPO and A2C runs have 0 to 7 cancelled orders. REINFORCE seed 0 has 58 cancelled orders and seed 1 has 19. No long run ended in a prescription violation or a "
    "ruined box, and none of the 40 sweep checkpoints handed over without a valid prescription on the test deliveries; the only failures are cancellations (and 9 ruined boxes in the sweep, 6 of them from REINFORCE settings). "
    "Figure 11 compares each sweep run's validation return with its test return. The points lie on or above the diagonal: the weaker runs score about 0.5 to 1.5 points higher on the test set, so the "
    "20 validation deliveries were slightly harder than the 100 test deliveries and there is no sign that choosing the best validation score over-fitted. For the four winners the validation and test returns "
    "are +6.01 and +6.02 (DQN), +6.00 and +5.77 (PPO), +5.67 and +5.95 (A2C) and +0.09 and +1.70 (REINFORCE)."
)

TEXT["limitations"] = [
    "The environment was changed after the first sweep. In the first version (rider-delivery-3) the policy-gradient methods reached only 26-42% success, because WAIT was legal everywhere and never useful, the progress reward was weak, and the riders could not read a direction to the patient. "
    "I made WAIT legal only while it rains, made rain stop after 6-14 steps, raised the progress reward from 0.04 to 0.12 per minute and added a compass bearing to the observation. The numbers in this report come only from the changed environment (v4); the first sweep is not comparable, and the compass bearing makes the routing part easier.",
    "One seed per sweep setting and 30,000 frames. Differences of less than about 0.5 return points between settings are inside the seed noise, and the sweep says more about speed of learning than about the final quality of a setting.",
    "Two seeds in the long runs and 100 test deliveries per agent. DQN, PPO and A2C are within one standard error of each other, so no ranking among them is supported; only the gap to REINFORCE is clear.",
    "The agents match the rule-following rider but do not beat it. The designed rules (cover in rain, check the prescription, ride the fastest road) are close to optimal in this simulator, so there is little room for a learned policy to improve, and nobody uses the shelter action.",
    "All agents are evaluated with the greedy policy. The environment is a simulation with invented parameters inspired by Kigali; the returns are not measured on real FARUMASI deliveries, and the observation values would have to come from real GPS, weather and order data before deployment.",
]

TEXT["conclusion"] = (
    "A delivery with traffic, rain and a prescription check is a sequential decision problem, and a hand-written rule-following rider (+6.54) gives the ceiling to compare against. "
    "Under identical conditions DQN, PPO and A2C learned almost all of it (+6.24, +6.21 and +6.00, with 95-100% success), and DQN and PPO needed about 20,000 frames or fewer to get there. "
    "REINFORCE was much slower and seed-dependent (+0.64 and +4.93) and never learned to take back a wrong prescription. The sweeps show that for the on-policy methods the number of parameter updates per frame "
    "(rollout length, PPO epochs, learning rate) mattered more than the clip range, the entropy bonus or the baseline, while DQN was robust to almost every setting. "
    "The most useful next steps are several seeds per setting, a harder task where the rules are not already near-optimal (for example real traffic and weather data through the serving API), and a shelter decision that is "
    "worth taking. The report, code, logs and trained agents are in the repository."
)
