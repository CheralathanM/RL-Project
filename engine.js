/* ==========================================================================
   SHCBO — Score-Horizon Coupled Belief Optimizer
   Pure decision engine and game rules. No DOM access lives in this file, so
   the same code runs in the browser and under node for the test suite.

     1  Bayesian belief grid ......... predictStep / likelihood / correctStep
     2  Score-horizon pressure ....... computePressure
     3  A* commitment cost ........... aStar / commitmentCost
     4  Regime detection ............. detectRegime
        Expected utility ............. bimatrixUtility
        Pressure-scaled softmax ...... softmax
        Confidence gate .............. computeBotAction

   Invariant: the engine never reads the opponent's true position when making a
   decision. Everything it "knows" lives in beliefGrid, which is built only from
   public observations (action type, hill outcome, shot feedback, respawns).
   ========================================================================== */

(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    else root.SHCBO = api;
}(typeof self !== 'undefined' ? self : this, function () {
'use strict';

const GRID = 5;
const HILL = { r: 2, c: 2 };

const CONFIG = {
    GRID,
    HILL,
    T: 25,                          // finite horizon — exactly 25 rounds
    LAMBDA: 0.5,                    // hill-attraction decay
    TAU: 0.25,                      // base softmax temperature
    THETA_SHOOT: 0.30,              // confidence gate
    W: 4,                           // rolling window for regime detection
    TAU_HIGH: 0.65,                 // hysteresis upper
    TAU_LOW: 0.35,                  // hysteresis lower
    SPAWN_HUMAN: { r: 0, c: 4 },
    SPAWN_BOT: { r: 4, c: 0 },
};

// The four blocks diagonally surrounding the hill at C3 — B2, B4, D2, D4.
// A player that scores respawns into exactly one of these.
// Labels map as file = 'A' + c, rank = r + 1, so C3 is { r: 2, c: 2 }.
const RESPAWN_CELLS = [
    { r: 1, c: 1 },   // B2
    { r: 3, c: 1 },   // B4
    { r: 1, c: 3 },   // D2
    { r: 3, c: 3 },   // D4
];

const ACTIONS = ['MOVE', 'SHOOT', 'HOLD'];

// Opponent action priors per detected regime.
const REGIME_PRIOR = {
    AGGRESSIVE: { MOVE: 0.60, SHOOT: 0.20, HOLD: 0.20 },
    PREDICTIVE: { MOVE: 0.20, SHOOT: 0.60, HOLD: 0.20 },
    DEFENSIVE:  { MOVE: 0.20, SHOOT: 0.20, HOLD: 0.60 },
};

const REGIME_META = {
    AGGRESSIVE: { label: 'Aggressive', sub: 'move-heavy', trigger: 'MOVE' },
    PREDICTIVE: { label: 'Predictive', sub: 'shoot-heavy', trigger: 'SHOOT' },
    DEFENSIVE:  { label: 'Defensive',  sub: 'hold-heavy',  trigger: 'HOLD' },
};

/* ── Pure helpers ──────────────────────────────────────────────────────── */

const same = (a, b) => !!a && !!b && a.r === b.r && a.c === b.c;
const manhattan = (a, b) => Math.abs(a.r - b.r) + Math.abs(a.c - b.c);
const inBounds = (r, c) => r >= 0 && r < GRID && c >= 0 && c < GRID;
const clone = c => ({ r: c.r, c: c.c });

// Board coordinates as file/rank, e.g. { r:3, c:1 } -> "B4".
const cellLabel = cell => `${String.fromCharCode(65 + cell.c)}${cell.r + 1}`;

function neighbors(r, c) {
    return [[-1, 0], [1, 0], [0, -1], [0, 1]]
        .map(([dr, dc]) => ({ r: r + dr, c: c + dc }))
        .filter(p => inBounds(p.r, p.c));
}

function zeroGrid() {
    return Array.from({ length: GRID }, () => Array(GRID).fill(0));
}

function forEachCell(fn) {
    for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) fn(r, c);
}

function isRespawnCell(cell) {
    return RESPAWN_CELLS.some(p => same(p, cell));
}

/* ── 3 — A* search ─────────────────────────────────────────────────────── */

function aStar(start, goal) {
    const key = p => p.r * GRID + p.c;
    const open = [{ r: start.r, c: start.c, f: manhattan(start, goal) }];
    const gScore = new Map([[key(start), 0]]);
    const cameFrom = new Map();
    const closed = new Set();
    let expansions = 0;

    while (open.length) {
        open.sort((a, b) => a.f - b.f);
        const cur = open.shift();
        expansions++;

        if (same(cur, goal)) {
            const path = [{ r: cur.r, c: cur.c }];
            let k = key(cur);
            while (cameFrom.has(k)) {
                const prev = cameFrom.get(k);
                path.unshift({ r: prev.r, c: prev.c });
                k = key(prev);
            }
            return { path, cost: path.length - 1, expansions };
        }

        closed.add(key(cur));

        for (const nb of neighbors(cur.r, cur.c)) {
            if (closed.has(key(nb))) continue;
            const tentative = gScore.get(key(cur)) + 1;               // g(n)
            if (gScore.has(key(nb)) && tentative >= gScore.get(key(nb))) continue;
            gScore.set(key(nb), tentative);
            cameFrom.set(key(nb), { r: cur.r, c: cur.c });
            const f = tentative + manhattan(nb, goal);                 // f = g + h
            const existing = open.find(o => key(o) === key(nb));
            if (existing) existing.f = f;
            else open.push({ r: nb.r, c: nb.c, f });
        }
    }
    return { path: [{ r: start.r, c: start.c }], cost: 0, expansions };
}

/* ── Distribution helpers ──────────────────────────────────────────────── */

// Softmax at temperature tau * (1 - 0.5 * alpha).
function softmax(values, temperature) {
    const t = Math.max(1e-6, temperature);
    const keys = Object.keys(values);
    const max = Math.max(...keys.map(k => values[k]));
    const exp = {};
    let z = 0;
    for (const k of keys) { exp[k] = Math.exp((values[k] - max) / t); z += exp[k]; }
    const out = {};
    for (const k of keys) out[k] = exp[k] / z;
    return out;
}

function gridArgmax(grid, exclude) {
    let best = null, bestP = -1;
    forEachCell((r, c) => {
        if (exclude && same(exclude, { r, c })) return;   // consecutive-shot ban
        if (grid[r][c] > bestP) { bestP = grid[r][c]; best = { r, c }; }
    });
    if (!best) best = clone(HILL);
    return { cell: best, prob: bestP < 0 ? 0 : bestP };
}

function gridEntropy(grid) {
    let h = 0;
    forEachCell((r, c) => {
        const p = grid[r][c];
        if (p > 1e-9) h -= p * Math.log2(p);
    });
    return h;
}

/* ==========================================================================
   Engine instance
   ========================================================================== */

function createEngine(options) {
    const opts = options || {};
    const rng = opts.rng || Math.random;

    const S = {
        turn: 1,
        scoreHuman: 0,
        scoreBot: 0,
        humanPos: clone(CONFIG.SPAWN_HUMAN),
        botPos: clone(CONFIG.SPAWN_BOT),
        gameOver: false,
        winner: null,

        beliefGrid: zeroGrid(),        // filtered posterior
        decisionBelief: zeroGrid(),    // one-step predictive projection
        alpha: 0,
        regime: 'DEFENSIVE',
        opponentHistory: [],
        pendingObs: null,
        telemetry: null,

        // Consecutive-shot restriction: the cell each side targeted on its most
        // recent shot, held only while that side keeps shooting. Taking any
        // other action clears it, so the ban blocks back-to-back shots at one
        // block without permanently locking a target out of play.
        lastShot: { human: null, bot: null },

        history: [],                   // per-round record for charts and debrief
    };

    /* ── 1 — Bayesian spatial belief grid ──────────────────────────────── */

    function initBelief() {
        S.beliefGrid = zeroGrid();
        S.beliefGrid[CONFIG.SPAWN_HUMAN.r][CONFIG.SPAWN_HUMAN.c] = 1.0;
        S.decisionBelief = S.beliefGrid.map(row => row.slice());
    }

    // Hill-attraction transition kernel, exp(-lambda * L1 distance to hill).
    function transitionKernel(from) {
        const candidates = [...neighbors(from.r, from.c), clone(from)];
        const weights = candidates.map(p => Math.exp(-CONFIG.LAMBDA * manhattan(p, HILL)));
        const z = weights.reduce((a, b) => a + b, 0);
        return candidates.map((p, i) => ({ cell: p, prob: weights[i] / z }));
    }

    // Prediction. Only MOVE displaces a player, so the kernel is applied only
    // when a MOVE was observed; SHOOT and HOLD leave position unchanged.
    function predictStep(observedAction) {
        if (observedAction !== 'MOVE') return;
        const next = zeroGrid();
        forEachCell((r, c) => {
            const prior = S.beliefGrid[r][c];
            if (prior <= 1e-6) return;
            for (const { cell, prob } of transitionKernel({ r, c })) {
                next[cell.r][cell.c] += prior * prob;
            }
        });
        S.beliefGrid = next;
    }

    // Likelihood P(o_t | (r,c)) assembled from the turn's public outcomes.
    function likelihood(r, c, obs) {
        const cell = { r, c };

        // A landed shot forces a knockback, which reveals the exact cell.
        if (obs.humanKnockedTo) return same(cell, obs.humanKnockedTo) ? 1 : 0;

        // Scoring respawns the scorer into one of four cells around the hill.
        // The event is public but the draw is not, so the posterior collapses
        // to a uniform over exactly those four candidates.
        if (obs.humanRespawned) return isRespawnCell(cell) ? 1 : 0;

        let L = 1;

        // The hill outcome is public: it partitions the grid on the hill cell.
        if (obs.hill === 'HUMAN' || obs.hill === 'CONTESTED') L *= same(cell, HILL) ? 1 : 0;
        else L *= same(cell, HILL) ? 0 : 1;

        // Shot feedback. A shielded shot is positive evidence (the opponent was
        // standing there to block it); a clean miss is negative evidence.
        if (obs.shotTarget) {
            if (obs.shotOutcome === 'SHIELDED') L *= same(cell, obs.shotTarget) ? 1 : 0;
            else if (obs.shotOutcome === 'MISS') L *= same(cell, obs.shotTarget) ? 0 : 1;
        }

        return L;
    }

    // Measurement correction with renormalisation.
    function correctStep(obs) {
        const posterior = zeroGrid();
        let z = 0;
        forEachCell((r, c) => {
            posterior[r][c] = likelihood(r, c, obs) * S.beliefGrid[r][c];
            z += posterior[r][c];
        });
        if (z <= 1e-9) {
            // Evidence inconsistent with the prior (can happen after a respawn
            // draw the filter had ruled out). Fall back to the likelihood alone
            // rather than keeping a posterior we know to be wrong.
            let z2 = 0;
            forEachCell((r, c) => { posterior[r][c] = likelihood(r, c, obs); z2 += posterior[r][c]; });
            if (z2 <= 1e-9) return;
            forEachCell((r, c) => { S.beliefGrid[r][c] = posterior[r][c] / z2; });
            return;
        }
        forEachCell((r, c) => { S.beliefGrid[r][c] = posterior[r][c] / z; });
    }

    function bayesianUpdate() {
        if (!S.pendingObs) return null;
        const before = gridEntropy(S.beliefGrid);
        predictStep(S.pendingObs.humanAction);
        correctStep(S.pendingObs);
        const after = gridEntropy(S.beliefGrid);
        return { entropyBefore: before, entropyAfter: after, delta: after - before };
    }

    // beliefGrid estimates the opponent's position at the END of the previous
    // round. Both sides commit simultaneously, so a shot fired this round
    // resolves against wherever they move next. The engine therefore acts on a
    // one-step predictive projection, mixing "stayed put" against the
    // transition kernel in the proportion the detected regime expects them to
    // move.
    function refreshDecisionBelief() {
        const pMove = REGIME_PRIOR[S.regime].MOVE;
        const next = zeroGrid();
        forEachCell((r, c) => {
            const prior = S.beliefGrid[r][c];
            if (prior <= 1e-6) return;
            next[r][c] += prior * (1 - pMove);
            for (const { cell, prob } of transitionKernel({ r, c })) {
                next[cell.r][cell.c] += prior * pMove * prob;
            }
        });
        S.decisionBelief = next;
    }

    /* ── 2 — Score-horizon pressure ────────────────────────────────────── */

    function computePressure() {
        const sA = S.scoreHuman, sB = S.scoreBot;
        const t = Math.min(S.turn, CONFIG.T);
        const horizon = CONFIG.T - t + 1;
        const indicator = sA >= sB ? 1 : 0;
        const numerator = (sA - sB) + (1 / horizon) * indicator;
        const denominator = CONFIG.T - t + 1.5;
        S.alpha = Math.min(1, Math.max(0, numerator / denominator));
        return { numerator, denominator, horizon, indicator, alpha: S.alpha };
    }

    /* ── 4 — Regime detection with hysteresis ──────────────────────────── */

    function actionFrequencies() {
        const w = S.opponentHistory.slice(-CONFIG.W);
        const f = { MOVE: 0, SHOOT: 0, HOLD: 0 };
        if (!w.length) return { freq: f, n: 0 };
        w.forEach(a => { f[a]++; });
        ACTIONS.forEach(a => { f[a] = f[a] / w.length; });
        return { freq: f, n: w.length };
    }

    function detectRegime() {
        const { freq, n } = actionFrequencies();
        if (n < 2) return { regime: S.regime, freq, n, switched: false, reason: 'window not yet filled' };

        const trigger = REGIME_META[S.regime].trigger;

        // Hysteresis: hold the incumbent until its own signal decays past
        // tau_low, and only then adopt a challenger that has cleared tau_high.
        if (freq[trigger] >= CONFIG.TAU_LOW) {
            return {
                regime: S.regime, freq, n, switched: false,
                reason: `incumbent ${trigger} at ${freq[trigger].toFixed(2)} held above τ_low ${CONFIG.TAU_LOW}`,
            };
        }

        const challenger = ACTIONS.find(a => freq[a] > CONFIG.TAU_HIGH);
        if (!challenger) {
            return { regime: S.regime, freq, n, switched: false, reason: `no challenger cleared τ_high ${CONFIG.TAU_HIGH}` };
        }

        const next = challenger === 'MOVE' ? 'AGGRESSIVE' : challenger === 'SHOOT' ? 'PREDICTIVE' : 'DEFENSIVE';
        const switched = next !== S.regime;
        const previous = S.regime;
        S.regime = next;
        return {
            regime: next, previous, freq, n, switched,
            reason: `${challenger} at ${freq[challenger].toFixed(2)} cleared τ_high ${CONFIG.TAU_HIGH}`,
        };
    }

    /* ── Expected utility ──────────────────────────────────────────────── */

    // Opportunity cost of not advancing toward the hill.
    function commitmentCost(action, ctx) {
        if (action === 'MOVE') return 0.0;
        if (action === 'HOLD') return 0.15 * (1 - S.alpha);
        return 0.40 * (1 - S.decisionBelief[ctx.shootTarget.r][ctx.shootTarget.c]);
    }

    // Payoff to the bot for (aB, aA) given the opponent occupying hCell.
    // Encodes the cyclic dominance MOVE > HOLD, SHOOT > MOVE, HOLD > SHOOT.
    //
    // Denominated in expected hill points:
    //   - a step toward the hill is worth more when the remaining horizon is
    //     barely long enough to still get there;
    //   - a landed shot is worth the progress it destroys, since the target is
    //     knocked back and must walk in again;
    //   - contesting an occupied hill scores nothing but denies the opponent,
    //     which matters more as the bot falls behind (hence the alpha weight).
    function payoffBot(aB, aA, hCell, ctx) {
        const humanOnHill = same(hCell, HILL);
        let v = 0;

        const denial = 0.25 + 0.50 * S.alpha;
        const hitValue = Math.min(ctx.horizon, 0.8 + 0.35 * manhattan(hCell, CONFIG.SPAWN_HUMAN));

        if (aB === 'MOVE') {
            v += (ctx.dNow - ctx.dNext) * ctx.stepValue;
            if (same(ctx.nextCell, HILL)) v += humanOnHill ? denial : 1.0;
            if (aA === 'SHOOT') v -= 0.84;                            // SHOOT > MOVE
            if (aA === 'HOLD') v += 0.15;                             // MOVE > HOLD
        } else if (aB === 'SHOOT') {
            const onTarget = same(ctx.shootTarget, hCell);
            if (onTarget && aA !== 'HOLD') v += hitValue;
            // on target against HOLD yields nothing: HOLD > SHOOT
            if (ctx.dNow > 0) v -= 0.25;                              // forgone advance
            if (same(ctx.botPos, HILL)) v += humanOnHill ? denial : 1.0;
        } else { // HOLD
            if (same(ctx.botPos, HILL)) v += humanOnHill ? denial : 1.0;
            if (aA === 'SHOOT') v += 0.9;                             // HOLD > SHOOT
            if (aA === 'MOVE') v -= 0.15;                             // MOVE > HOLD
        }
        return v;
    }

    // Expectation over the belief grid and the regime-conditioned opponent prior.
    function bimatrixUtility(aB, ctx) {
        const prior = REGIME_PRIOR[S.regime];
        let u = 0;
        forEachCell((r, c) => {
            const b = S.decisionBelief[r][c];
            if (b <= 1e-6) return;
            let inner = 0;
            for (const aA of ACTIONS) inner += prior[aA] * payoffBot(aB, aA, { r, c }, ctx);
            u += b * inner;
        });
        return u;
    }

    function sampleFrom(dist) {
        const keys = Object.keys(dist);
        let x = rng();
        for (const k of keys) {
            if (x < dist[k]) return k;
            x -= dist[k];
        }
        return keys[keys.length - 1];
    }

    /* ── SHCBO core turn execution ─────────────────────────────────────── */

    function computeBotAction() {
        const observed = S.pendingObs;
        const beliefTrace = bayesianUpdate();
        const pressureTrace = computePressure();
        const regimeTrace = detectRegime();

        refreshDecisionBelief();

        // The consecutive-shot restriction applies to the agent too: its own
        // previous target is excluded before the argmax is taken, so the
        // confidence gate is evaluated against a target it may legally fire at.
        const banned = S.lastShot.bot;
        const argmax = gridArgmax(S.decisionBelief, banned);
        const unconstrained = gridArgmax(S.decisionBelief, null);

        const search = aStar(S.botPos, HILL);
        const ctx = {
            botPos: S.botPos,
            nextCell: search.path.length > 1 ? search.path[1] : clone(S.botPos),
            dNow: manhattan(S.botPos, HILL),
            dNext: 0,
            shootTarget: argmax.cell,
            horizon: pressureTrace.horizon,
            stepValue: 0,
        };
        ctx.dNext = manhattan(ctx.nextCell, HILL);
        ctx.stepValue = 0.35 + 0.65 * Math.min(1, ctx.dNow / Math.max(1, ctx.horizon));

        const U = {}, C = {}, net = {};
        for (const a of ACTIONS) {
            C[a] = commitmentCost(a, ctx);
            U[a] = bimatrixUtility(a, ctx);
            net[a] = U[a] - C[a];
        }

        const temperature = CONFIG.TAU * (1 - 0.5 * S.alpha);
        const policy = softmax(net, temperature);
        const sampled = sampleFrom(policy);

        let action = sampled;
        let target;
        let gated = false;

        if (action === 'SHOOT') {
            target = clone(argmax.cell);
            if (argmax.prob < CONFIG.THETA_SHOOT) {          // confidence gate
                gated = true;
                action = 'MOVE';                             // A* fallback
                target = clone(ctx.nextCell);
            }
        } else if (action === 'MOVE') {
            target = clone(ctx.nextCell);
        } else {
            target = clone(S.botPos);
        }

        S.telemetry = {
            U, C, net, policy, temperature, argmax, unconstrained, search, gated,
            pressure: pressureTrace, regime: regimeTrace, belief: beliefTrace,
            entropy: gridEntropy(S.decisionBelief),
            observation: observed,
            shotBanned: banned ? clone(banned) : null,
            sampled, action, target: clone(target),
            opponentPrior: REGIME_PRIOR[S.regime],
        };
        S.pendingObs = null;

        return { action, target: clone(target) };
    }

    /* ── Game rules ────────────────────────────────────────────────────── */

    // A player may not target the same block on two consecutive shots.
    function validateShot(side, cell) {
        const last = S.lastShot[side];
        if (last && same(last, cell)) {
            return {
                ok: false,
                reason: `You cannot target ${cellLabel(cell)} twice in a row — pick another block.`,
            };
        }
        return { ok: true };
    }

    function validateAction(action, target) {
        if (S.gameOver) return { ok: false, reason: 'The match is over.' };
        if (action === 'MOVE') {
            if (!target || !inBounds(target.r, target.c)) return { ok: false, reason: 'Target is off the board.' };
            if (manhattan(target, S.humanPos) > 1) return { ok: false, reason: 'Move is one orthogonal step.' };
            return { ok: true };
        }
        if (action === 'SHOOT') {
            if (!target || !inBounds(target.r, target.c)) return { ok: false, reason: 'Target is off the board.' };
            return validateShot('human', target);
        }
        return { ok: true };
    }

    function pickRespawn() {
        return clone(RESPAWN_CELLS[Math.floor(rng() * RESPAWN_CELLS.length) % RESPAWN_CELLS.length]);
    }

    /**
     * Resolve one full round. Returns a plan describing everything that
     * happened, in order, so the presentation layer can animate it without
     * re-deriving any rules.
     */
    function resolveTurn(humanAct, humanTargetRaw) {
        if (S.gameOver) return null;

        const check = validateAction(humanAct, humanTargetRaw);
        if (!check.ok) return { rejected: true, reason: check.reason };

        const roundNo = S.turn;
        const decision = computeBotAction();
        const events = [];
        const push = (text, kind) => events.push({ text, kind: kind || 'system' });

        const humanTarget = humanAct === 'HOLD' ? clone(S.humanPos) : clone(humanTargetRaw);
        const botAct = decision.action;
        const botTarget = clone(decision.target);

        push(`Round ${roundNo} started`, 'round');
        push(`You chose ${humanAct}${humanAct === 'HOLD' ? '' : ` → ${cellLabel(humanTarget)}`}`, 'human');
        push('Belief state updated', 'ai');
        push(`Prediction → ${cellLabel(S.telemetry.argmax.cell)} at ${(S.telemetry.argmax.prob * 100).toFixed(0)}%`, 'ai');
        push(`SHCBO chose ${botAct}${botAct === 'HOLD' ? '' : ` → ${cellLabel(botTarget)}`}`, 'bot');
        if (S.telemetry.gated) push('Shoot suppressed by confidence gate', 'ai');

        const from = { human: clone(S.humanPos), bot: clone(S.botPos) };

        // Phase 1 — movement.
        if (humanAct === 'MOVE') S.humanPos = clone(humanTarget);
        if (botAct === 'MOVE') S.botPos = clone(botTarget);
        const afterMove = { human: clone(S.humanPos), bot: clone(S.botPos) };

        // Phase 2 — shots, resolved against post-move positions.
        const shots = [];
        let humanKnockedTo = null;
        let botKnockedTo = null;

        if (humanAct === 'SHOOT') {
            S.lastShot.human = clone(humanTarget);
            const hit = same(humanTarget, S.botPos);
            const outcome = !hit ? 'MISS' : (botAct === 'HOLD' ? 'SHIELDED' : 'HIT');
            shots.push({ side: 'human', from: clone(afterMove.human), target: clone(humanTarget), outcome });
            if (outcome === 'HIT') {
                botKnockedTo = clone(CONFIG.SPAWN_BOT);
                S.botPos = clone(botKnockedTo);
                push(`Direct hit on ${cellLabel(humanTarget)} — SHCBO knocked back`, 'score');
            } else if (outcome === 'SHIELDED') {
                push(`SHCBO shielded your shot at ${cellLabel(humanTarget)}`, 'bot');
            } else {
                push(`Your shot at ${cellLabel(humanTarget)} missed`, 'system');
            }
        }

        let shotOutcome = null;
        if (botAct === 'SHOOT') {
            S.lastShot.bot = clone(botTarget);
            const hit = same(botTarget, S.humanPos);
            shotOutcome = !hit ? 'MISS' : (humanAct === 'HOLD' ? 'SHIELDED' : 'HIT');
            shots.push({ side: 'bot', from: clone(afterMove.bot), target: clone(botTarget), outcome: shotOutcome });
            if (shotOutcome === 'HIT') {
                humanKnockedTo = clone(CONFIG.SPAWN_HUMAN);
                S.humanPos = clone(humanKnockedTo);
                push(`SHCBO hit you at ${cellLabel(botTarget)} — knocked back`, 'bot');
            } else if (shotOutcome === 'SHIELDED') {
                push(`You shielded the shot at ${cellLabel(botTarget)}`, 'human');
            } else {
                push(`SHCBO missed at ${cellLabel(botTarget)}`, 'system');
            }
        }

        // The restriction covers back-to-back shots only. A side that did
        // something else this round starts clean next round, so no block can be
        // locked out of play indefinitely.
        if (humanAct !== 'SHOOT') S.lastShot.human = null;
        if (botAct !== 'SHOOT') S.lastShot.bot = null;

        const afterShots = { human: clone(S.humanPos), bot: clone(S.botPos) };

        // Phase 3 — hill control and scoring.
        const humanOnHill = same(S.humanPos, HILL);
        const botOnHill = same(S.botPos, HILL);
        let hill, scored = null;

        if (humanOnHill && botOnHill) {
            hill = 'CONTESTED';
            push('Hill contested — no point awarded', 'system');
        } else if (humanOnHill) {
            hill = 'HUMAN'; scored = 'HUMAN'; S.scoreHuman++;
            push('You captured the hill  +1 point', 'score');
        } else if (botOnHill) {
            hill = 'BOT'; scored = 'BOT'; S.scoreBot++;
            push('SHCBO captured the hill  +1 point', 'bot');
        } else {
            hill = 'VACANT';
            push('Hill uncontrolled', 'system');
        }

        // Phase 4 — the scorer respawns into one of the four cells around the
        // hill, so a point never turns into permanent hill camping.
        let respawn = null;
        if (scored) {
            const to = pickRespawn();
            respawn = { side: scored, to, candidates: RESPAWN_CELLS.map(clone) };
            if (scored === 'HUMAN') S.humanPos = clone(to);
            else S.botPos = clone(to);
            push(`${scored === 'HUMAN' ? 'You respawn' : 'SHCBO respawns'} at ${cellLabel(to)}`, 'system');
        }

        // Observations the engine may condition on next round. All of these are
        // publicly visible events, never the opponent's coordinates.
        S.pendingObs = {
            humanAction: humanAct,
            hill,
            shotTarget: botAct === 'SHOOT' ? clone(botTarget) : null,
            shotOutcome,
            humanKnockedTo,
            humanRespawned: scored === 'HUMAN' ? clone(respawn.to) : null,
        };
        S.opponentHistory.push(humanAct);

        S.history.push({
            round: roundNo,
            alpha: S.alpha,
            hill,
            botAction: botAct,
            humanAction: humanAct,
            scoreHuman: S.scoreHuman,
            scoreBot: S.scoreBot,
            beliefOnTruth: S.decisionBelief[S.humanPos.r][S.humanPos.c],
        });

        S.turn++;
        computePressure();

        if (S.turn > CONFIG.T) {
            S.gameOver = true;
            S.winner = S.scoreHuman > S.scoreBot ? 'HUMAN'
                     : S.scoreBot > S.scoreHuman ? 'BOT' : 'DRAW';
            push(`Match complete — ${S.winner === 'DRAW' ? 'draw' : S.winner === 'HUMAN' ? 'you win' : 'SHCBO wins'}`, 'round');
        }

        return {
            rejected: false,
            round: roundNo,
            humanAct, humanTarget, botAct, botTarget,
            from, afterMove, afterShots,
            final: { human: clone(S.humanPos), bot: clone(S.botPos) },
            shots, humanKnockedTo, botKnockedTo,
            hill, scored, respawn,
            scores: { human: S.scoreHuman, bot: S.scoreBot },
            gameOver: S.gameOver,
            winner: S.winner,
            telemetry: S.telemetry,
            events,
        };
    }

    function reset() {
        S.turn = 1;
        S.scoreHuman = 0;
        S.scoreBot = 0;
        S.humanPos = clone(CONFIG.SPAWN_HUMAN);
        S.botPos = clone(CONFIG.SPAWN_BOT);
        S.gameOver = false;
        S.winner = null;
        S.alpha = 0;
        S.regime = 'DEFENSIVE';
        S.opponentHistory = [];
        S.pendingObs = null;
        S.telemetry = null;
        S.lastShot = { human: null, bot: null };
        S.history = [];
        initBelief();
        computePressure();
    }

    reset();

    return {
        state: S,
        reset,
        resolveTurn,
        computeBotAction,
        validateAction,
        validateShot,
        // exposed for the presentation layer and the test suite
        refreshDecisionBelief,
        computePressure,
        detectRegime,
        actionFrequencies,
        aStar: (a, b) => aStar(a, b),
        beliefArgmax: exclude => gridArgmax(S.decisionBelief, exclude),
        beliefEntropy: () => gridEntropy(S.decisionBelief),
    };
}

return {
    CONFIG, ACTIONS, HILL, GRID, RESPAWN_CELLS, REGIME_PRIOR, REGIME_META,
    createEngine,
    // pure utilities shared with the UI and tests
    same, manhattan, inBounds, neighbors, zeroGrid, forEachCell,
    cellLabel, isRespawnCell, aStar, softmax, gridArgmax, gridEntropy,
};
}));
