# King of the Hill — SHCBO agent

A browser implementation of **SHCBO** (Score-Horizon Coupled Belief Optimizer), a decision engine for
simultaneous-move, finite-horizon grid games under positional uncertainty.

Two players contest a 5×5 grid with a scoring hill at (2,2) over a 10-turn horizon. Both sides commit
simultaneously each turn from `{Move, Shoot, Hold}`, which form a non-transitive cycle:

```
Move ≻ Hold        Shoot ≻ Move        Hold ≻ Shoot
```

You score +1 for each turn you end alone on the hill. Simultaneous occupation scores nothing. Highest
total after 10 turns wins.

No build step, no dependencies, no backend — open `index.html` directly, or serve the folder:

```bash
python -m http.server 8000
```

## How the agent works

The agent **never reads the opponent's coordinates**. Everything it knows is reconstructed from three
public signals: the opponent's action *type*, who took the hill, and whether its own shot hit, was
shielded, or missed. The inspector panel is a live trace of that inference.

### 1. Bayesian belief grid

The agent maintains `B_t(r,c)`, a distribution over opponent position, updated each turn by a
predict/correct cycle.

**Predict.** Only `Move` displaces a player, so the transition kernel is applied only when a move was
observed. The kernel is hill-attracted:

```
P(cell) ∝ exp(−λ · ‖cell − hill‖₁),    λ = 0.5
```

**Correct.** The likelihood `P(o_t | (r,c))` combines signals that each genuinely constrain position:

- **Hill outcome** partitions the grid on the hill cell — if the hill went vacant, the opponent was
  not standing on it.
- **A shielded shot** is *positive* evidence: they were there to block it.
- **A clean miss** is negative evidence: they were not on the targeted cell.
- **A landed shot** forces a respawn, revealing the cell exactly.

Because both sides commit simultaneously, the posterior describes where the opponent was at the *end*
of the previous turn. Firing at that cell is systematically stale, so the agent acts on a one-step
predictive projection, mixing "stayed put" against the transition kernel in the proportion the
detected regime expects them to move.

### 2. Score-horizon pressure

A scalar `α ∈ [0,1]` couples the score differential to the remaining horizon:

```
α = clamp( ( (S_A − S_B) + (T − t + 1)⁻¹ · 1[S_A ≥ S_B] ) / (T − t + 1.5),  0,  1 )
```

α rises as the agent falls behind late, and it sharpens the softmax below toward decisive play.

### 3. A* commitment cost

A* over the grid yields the optimal path to the hill and its cost. Each candidate action carries the
opportunity cost of not advancing:

```
C(Move)  = 0
C(Hold)  = 0.15 · (1 − α)
C(Shoot) = 0.40 · (1 − B_t(target))
```

### 4. Regime detection with hysteresis

Opponent action frequencies over a rolling window of `W = 4` turns classify play into **Aggressive**
(move-heavy), **Predictive** (shoot-heavy) or **Defensive** (hold-heavy). Switching requires a
challenger above `τ_high = 0.65` *and* the incumbent decaying below `τ_low = 0.35`, which prevents the
exploit/counter-exploit oscillation that intransitive games otherwise induce.

### Action selection

Expected utility is taken over the belief grid and the regime-conditioned opponent prior:

```
U(a_B) = Σ_(r,c) B_t(r,c) · Σ_(a_A) P(a_A | regime) · Payoff(a_B, a_A | r,c)
```

Actions are then sampled from a pressure-scaled softmax over `Ũ(a) = U(a) − C(a)`:

```
π(a) = softmax( Ũ(a) / (τ · (1 − 0.5α)) ),    τ = 0.25
```

Falling temperature under rising α makes the agent progressively more decisive as the horizon closes.
Finally a confidence gate suppresses `Shoot` when `max B_t < θ = 0.30`, falling back to the A* move —
without it the agent wastes shots into a flat belief.

## Measured performance

150 matches against each scripted opponent:

| Opponent | Agent win | Draw | Agent loss | Avg score (human–agent) |
|---|---|---|---|---|
| Hill-rusher | 0% | 55% | 45% | 0.79 – 0.00 |
| Sniper | 59% | 23% | 19% | 0.60 – 1.57 |
| Random | 81% | 11% | 8% | 0.40 – 3.05 |

Against a pure hill-turtle a 0–0 draw is the best available outcome — `Hold ≻ Shoot` means you cannot
shoot an opponent off the hill, so contesting it is the correct play. Draw rate is the meaningful
metric there, not win rate.

Belief quality: the agent places a mean **0.41** probability mass on the opponent's true cell, against
**0.04** for uniform guessing.

## Tuning notes

The payoff matrix is denominated in expected hill points: a step toward the hill is worth more when
the horizon is barely long enough to reach it, a landed shot is worth the progress it destroys, and
contesting an occupied hill is valued by α, since denial matters more when trailing.

## Controls

| Key | Action |
|---|---|
| `1` `2` `3` | Move / Shoot / Hold |
| `↑` `↓` `←` `→` | Aim |
| `Enter` | Confirm turn |
| `Esc` | Clear target |
| `B` | Project the agent's belief onto the board |
| `R` | Restart |
| `?` | Rules |

## Files

```
index.html    markup and inspector layout
app.js        SHCBO engine (lines 1–441) and UI (442+)
styles.css    instrument theme, dark and light
```
