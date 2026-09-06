# King of the Hill — SHCBO agent

A browser implementation of **SHCBO** (Score-Horizon Coupled Belief Optimizer), a decision engine for
simultaneous-move, finite-horizon grid games under positional uncertainty.

Two players contest a 5×5 grid with a scoring hill at **C3** over a **25-round** horizon. Both sides
commit simultaneously each round from `{Move, Shoot, Hold}`, which form a non-transitive cycle:

```
Move ≻ Hold        Shoot ≻ Move        Hold ≻ Shoot
```

You score **+1** for each round you end alone on the hill; simultaneous occupation scores nothing. A
point does not end the match — play always runs the full 25 rounds, and the higher score wins.

Two rules keep the board moving:

- **Respawn on score.** Whoever scores is moved off the hill into one of the four blocks around it
  (B3, D3, C2, C4), chosen uniformly. A point cannot turn into permanent camping.
- **No consecutive repeat shots.** Neither side may target the same block on two shots in a row. The
  restriction lapses as soon as that side takes a different action, so no block is ever locked out of
  play. It binds the agent exactly as it binds you.

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

150 matches of 25 rounds against each scripted opponent, current rules:

| Opponent | Agent win | Draw | Agent loss | Avg score (human – agent) |
|---|---|---|---|---|
| Hill-rusher | 0% | 43% | 57% | 0.89 – 0.00 |
| Sniper | 53% | 9% | 38% | 2.65 – 2.91 |
| Random | 99% | 1% | 0% | 0.46 – 15.61 |

Belief quality separates into two numbers. On an undisrupted observation stream the filter places a
mean **0.745** probability mass on the opponent's true block, against **0.04** for uniform guessing.
In live play that falls to roughly **0.09**, because the respawn rule deliberately resets belief to a
uniform over four cells — capping it at 0.25 — after every point, and the one-step projection spreads
it further. Both figures are produced by `tests.js`.

### Known limitation: the hill-camper standoff

Against an opponent that walks to the hill and then holds forever, the match grinds to roughly 1–0.
This is a property of the rule set combined with the agent's action space, not a defect in the
filter:

- `Hold ≻ Shoot`, so a player standing on the hill and holding cannot be shot off it. Over 1500
  measured rounds the agent fired 597 shots at such an opponent and landed **zero**.
- Simultaneous occupation scores nothing, so the agent contests the hill and both sides grind at 0.
- The agent's `Move` is defined as the A\* step toward the hill. Once it is *on* the hill that step
  is a no-op, so it has no way to express "retreat, concede a point, and retake the vacated hill" —
  which is the play that would break the deadlock.

Sweeping the denial weight from `0.25 + 0.50α` down to `0` changes the outcome not at all, which
confirms the cause is the action space rather than payoff tuning. Widening `Move` to evaluate all
four neighbours instead of only the A\* step would address it, at the cost of changing the agent's
action space — deliberately left alone here.

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
engine.js     SHCBO engine and game rules — pure logic, no DOM, runs under node
app.js        presentation layer: rendering, animation, panels
index.html    dashboard markup
styles.css    research-dashboard theme
tests.js      dependency-free test suite
```

## Tests

```bash
node tests.js
```

44 assertions covering the round flow, winner determination, hill scoring, the respawn rule, the
consecutive-shot restriction, belief normalisation and collapse, regime hysteresis, policy
normalisation, the confidence gate, A* optimality, and the invariant that the agent's decision is
unchanged when the opponent's true position is altered behind its back.
