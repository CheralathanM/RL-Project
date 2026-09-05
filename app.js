/* ==========================================================================
   SHCBO — Score-Horizon Coupled Belief Optimizer

   A decision engine for simultaneous-move, finite-horizon grid games under
   positional uncertainty. Four components feed one action-selection step:

     1  Bayesian belief grid ......... predictStep(), likelihood(), correctStep()
     2  Score-horizon pressure ....... computePressure()
     3  A* commitment cost ........... aStar(), commitmentCost()
     4  Regime detection ............. detectRegime()
        Expected utility ............. bimatrixUtility()
        Pressure-scaled softmax ...... softmax()
        Confidence gate .............. computeBotAction()

   Invariant: the engine never reads humanPos. Everything it "knows" about the
   opponent lives in beliefGrid, which is built only from public observations
   (action type, hill outcome, shot feedback).
   ========================================================================== */

const CONFIG = {
    GRID: 5,
    HILL: { r: 2, c: 2 },
    T: 10,                          // finite horizon
    LAMBDA: 0.5,                    // hill-attraction decay
    TAU: 0.25,                      // base softmax temperature
    THETA_SHOOT: 0.30,              // confidence gate
    W: 4,                           // rolling window
    TAU_HIGH: 0.65,                 // hysteresis upper
    TAU_LOW: 0.35,                  // hysteresis lower
    SPAWN_HUMAN: { r: 0, c: 4 },
    SPAWN_BOT: { r: 4, c: 0 },
};

const ACTIONS = ['MOVE', 'SHOOT', 'HOLD'];

// Opponent action priors per detected regime.
const REGIME_PRIOR = {
    AGGRESSIVE: { MOVE: 0.60, SHOOT: 0.20, HOLD: 0.20 },
    PREDICTIVE: { MOVE: 0.20, SHOOT: 0.60, HOLD: 0.20 },
    DEFENSIVE:  { MOVE: 0.20, SHOOT: 0.20, HOLD: 0.60 },
};

const REGIME_META = {
    AGGRESSIVE: { label: 'AGGRESSIVE', sub: 'Move-heavy', trigger: 'MOVE' },
    PREDICTIVE: { label: 'PREDICTIVE', sub: 'Shoot-heavy', trigger: 'SHOOT' },
    DEFENSIVE:  { label: 'DEFENSIVE',  sub: 'Hold-heavy',  trigger: 'HOLD' },
};

/* ── Game state ────────────────────────────────────────────────────────── */
let turn, scoreHuman, scoreBot, humanPos, botPos, gameOver;
let selectedAction, selectedTarget;

/* ── Engine state ──────────────────────────────────────────────────────── */
let beliefGrid;          // B_t, filtered posterior
let decisionBelief;      // one-step predictive projection the engine acts on
let alpha;               // score-horizon pressure
let regime;              // current FSM state
let opponentHistory;     // observed human action types
let pendingObs;          // observations from turn t, consumed at start of t+1
let telemetry;           // last decision trace, for the inspector

/* ── Geometry helpers ──────────────────────────────────────────────────── */
const same = (a, b) => !!a && !!b && a.r === b.r && a.c === b.c;
const manhattan = (a, b) => Math.abs(a.r - b.r) + Math.abs(a.c - b.c);
const inBounds = (r, c) => r >= 0 && r < CONFIG.GRID && c >= 0 && c < CONFIG.GRID;

function neighbors(r, c) {
    return [[-1, 0], [1, 0], [0, -1], [0, 1]]
        .map(([dr, dc]) => ({ r: r + dr, c: c + dc }))
        .filter(p => inBounds(p.r, p.c));
}

function zeroGrid() {
    return Array.from({ length: CONFIG.GRID }, () => Array(CONFIG.GRID).fill(0));
}

function forEachCell(fn) {
    for (let r = 0; r < CONFIG.GRID; r++)
        for (let c = 0; c < CONFIG.GRID; c++) fn(r, c);
}

/* ==========================================================================
   3 — A* search and commitment cost
   ========================================================================== */

function aStar(start, goal) {
    const key = p => p.r * CONFIG.GRID + p.c;
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

// Opportunity cost of not advancing toward the hill.
function commitmentCost(action, ctx) {
    if (action === 'MOVE') return 0.0;
    if (action === 'HOLD') return 0.15 * (1 - alpha);
    return 0.40 * (1 - decisionBelief[ctx.shootTarget.r][ctx.shootTarget.c]);
}

/* ==========================================================================
   1 — Bayesian spatial belief grid
   ========================================================================== */

function initBelief() {
    beliefGrid = zeroGrid();
    beliefGrid[CONFIG.SPAWN_HUMAN.r][CONFIG.SPAWN_HUMAN.c] = 1.0;
}

// Hill-attraction transition kernel, exp(-lambda * L1 distance to hill).
function transitionKernel(from) {
    const candidates = [...neighbors(from.r, from.c), { r: from.r, c: from.c }];
    const weights = candidates.map(p => Math.exp(-CONFIG.LAMBDA * manhattan(p, CONFIG.HILL)));
    const z = weights.reduce((a, b) => a + b, 0);
    return candidates.map((p, i) => ({ cell: p, prob: weights[i] / z }));
}

// Prediction. Only MOVE displaces a player, so the kernel is applied
// only when a MOVE was observed; SHOOT and HOLD leave position unchanged.
function predictStep(observedAction) {
    if (observedAction !== 'MOVE') return;
    const next = zeroGrid();
    forEachCell((r, c) => {
        const prior = beliefGrid[r][c];
        if (prior <= 1e-6) return;
        for (const { cell, prob } of transitionKernel({ r, c })) {
            next[cell.r][cell.c] += prior * prob;
        }
    });
    beliefGrid = next;
}

// Likelihood P(o_t | (r,c)) assembled from the turn's public outcomes.
function likelihood(r, c, obs) {
    const cell = { r, c };

    // A landed shot forces a respawn, which reveals the exact cell.
    if (obs.humanKnockedTo) return same(cell, obs.humanKnockedTo) ? 1 : 0;

    let L = 1;

    // The hill outcome is public: it partitions the grid on the hill cell.
    if (obs.hill === 'HUMAN' || obs.hill === 'CONTESTED') L *= same(cell, CONFIG.HILL) ? 1 : 0;
    else L *= same(cell, CONFIG.HILL) ? 0 : 1;

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
        posterior[r][c] = likelihood(r, c, obs) * beliefGrid[r][c];
        z += posterior[r][c];
    });
    if (z <= 1e-9) return;                 // inconsistent evidence: keep the prior
    forEachCell((r, c) => { beliefGrid[r][c] = posterior[r][c] / z; });
}

function bayesianUpdate() {
    if (!pendingObs) return;
    predictStep(pendingObs.humanAction);
    correctStep(pendingObs);
}

// beliefGrid estimates the opponent's position at the END of the previous turn.
// Both sides commit simultaneously, so a shot fired this turn resolves against
// wherever they move next. The engine therefore acts on a one-step predictive
// projection, mixing "stayed put" against the transition kernel in the
// proportion the detected regime expects them to move.
function refreshDecisionBelief() {
    const pMove = REGIME_PRIOR[regime].MOVE;
    const next = zeroGrid();
    forEachCell((r, c) => {
        const prior = beliefGrid[r][c];
        if (prior <= 1e-6) return;
        next[r][c] += prior * (1 - pMove);
        for (const { cell, prob } of transitionKernel({ r, c })) {
            next[cell.r][cell.c] += prior * pMove * prob;
        }
    });
    decisionBelief = next;
}

function gridArgmax(grid) {
    let best = { r: CONFIG.HILL.r, c: CONFIG.HILL.c }, bestP = -1;
    forEachCell((r, c) => {
        if (grid[r][c] > bestP) { bestP = grid[r][c]; best = { r, c }; }
    });
    return { cell: best, prob: bestP };
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
   2 — Score-horizon pressure
   ========================================================================== */

function computePressure() {
    const sA = scoreHuman, sB = scoreBot;
    const horizon = CONFIG.T - Math.min(turn, CONFIG.T) + 1;   // T - t + 1
    const indicator = sA >= sB ? 1 : 0;
    const numerator = (sA - sB) + (1 / horizon) * indicator;
    const denominator = CONFIG.T - Math.min(turn, CONFIG.T) + 1.5;
    alpha = Math.min(1, Math.max(0, numerator / denominator));
    return { numerator, denominator, horizon, indicator };
}

/* ==========================================================================
   4 — Intransitive regime detection with hysteresis
   ========================================================================== */

function actionFrequencies() {
    const w = opponentHistory.slice(-CONFIG.W);
    const f = { MOVE: 0, SHOOT: 0, HOLD: 0 };
    if (!w.length) return { freq: f, n: 0 };
    w.forEach(a => { f[a]++; });
    ACTIONS.forEach(a => { f[a] = f[a] / w.length; });
    return { freq: f, n: w.length };
}

function detectRegime() {
    const { freq, n } = actionFrequencies();
    if (n < 2) return { regime, freq, n, switched: false, reason: 'window not yet filled' };

    const trigger = REGIME_META[regime].trigger;

    // Hysteresis: hold the incumbent until its own signal decays past tau_low,
    // and only then adopt a challenger that has cleared tau_high.
    if (freq[trigger] >= CONFIG.TAU_LOW) {
        return {
            regime, freq, n, switched: false,
            reason: `incumbent ${trigger} at ${freq[trigger].toFixed(2)} ≥ τ_low ${CONFIG.TAU_LOW}`,
        };
    }

    const challenger = ACTIONS.find(a => freq[a] > CONFIG.TAU_HIGH);
    if (!challenger) {
        return { regime, freq, n, switched: false, reason: `no challenger above τ_high ${CONFIG.TAU_HIGH}` };
    }

    const next = challenger === 'MOVE' ? 'AGGRESSIVE' : challenger === 'SHOOT' ? 'PREDICTIVE' : 'DEFENSIVE';
    const switched = next !== regime;
    regime = next;
    return { regime, freq, n, switched, reason: `${challenger} at ${freq[challenger].toFixed(2)} > τ_high ${CONFIG.TAU_HIGH}` };
}

/* ==========================================================================
   Expected utility and pressure-scaled softmax
   ========================================================================== */

// Payoff to the bot for (aB, aA) given the opponent occupying hCell.
// Encodes the cyclic dominance MOVE ≻ HOLD, SHOOT ≻ MOVE, HOLD ≻ SHOOT.
//
// The payoff matrix is denominated in expected hill points:
//   - a step toward the hill is worth more when the remaining horizon is barely
//     long enough to still get there;
//   - a landed shot is worth the progress it destroys, since the target
//     respawns and must walk back;
//   - contesting an occupied hill scores nothing but denies the opponent, which
//     matters more as the bot falls behind (hence the alpha weighting).
function payoffBot(aB, aA, hCell, ctx) {
    const humanOnHill = same(hCell, CONFIG.HILL);
    let v = 0;

    const denial = 0.25 + 0.50 * alpha;
    const hitValue = Math.min(
        ctx.horizon,
        0.8 + 0.35 * manhattan(hCell, CONFIG.SPAWN_HUMAN)
    );

    if (aB === 'MOVE') {
        v += (ctx.dNow - ctx.dNext) * ctx.stepValue;              // closing on the hill
        if (same(ctx.nextCell, CONFIG.HILL)) v += humanOnHill ? denial : 1.0;
        if (aA === 'SHOOT') v -= 0.84;                            // SHOOT ≻ MOVE
        if (aA === 'HOLD') v += 0.15;                             // MOVE ≻ HOLD
    } else if (aB === 'SHOOT') {
        const onTarget = same(ctx.shootTarget, hCell);
        if (onTarget && aA !== 'HOLD') v += hitValue;
        // on target against HOLD yields nothing: HOLD ≻ SHOOT
        if (ctx.dNow > 0) v -= 0.25;                              // forgone advance
        if (same(ctx.botPos, CONFIG.HILL)) v += humanOnHill ? denial : 1.0;
    } else { // HOLD
        if (same(ctx.botPos, CONFIG.HILL)) v += humanOnHill ? denial : 1.0;
        if (aA === 'SHOOT') v += 0.9;                             // HOLD ≻ SHOOT
        if (aA === 'MOVE') v -= 0.15;                             // MOVE ≻ HOLD
    }
    return v;
}

// Expectation over the belief grid and the regime-conditioned opponent prior.
function bimatrixUtility(aB, ctx) {
    const prior = REGIME_PRIOR[regime];
    let u = 0;
    forEachCell((r, c) => {
        const b = decisionBelief[r][c];
        if (b <= 1e-6) return;
        let inner = 0;
        for (const aA of ACTIONS) inner += prior[aA] * payoffBot(aB, aA, { r, c }, ctx);
        u += b * inner;
    });
    return u;
}

// Softmax at temperature tau * (1 - 0.5 * alpha).
function softmax(values, temperature) {
    const t = Math.max(1e-6, temperature);
    const keys = Object.keys(values);
    const max = Math.max(...keys.map(k => values[k]));
    const exp = {};
    let z = 0;
    for (const k of keys) {
        exp[k] = Math.exp((values[k] - max) / t);
        z += exp[k];
    }
    const out = {};
    for (const k of keys) out[k] = exp[k] / z;
    return out;
}

function sampleFrom(dist) {
    const keys = Object.keys(dist);
    let x = Math.random();
    for (const k of keys) {
        if (x < dist[k]) return k;
        x -= dist[k];
    }
    return keys[keys.length - 1];
}

/* ==========================================================================
   SHCBO core turn execution
   ========================================================================== */

function computeBotAction() {
    bayesianUpdate();
    const pressureTrace = computePressure();
    const regimeTrace = detectRegime();

    refreshDecisionBelief();
    const argmax = gridArgmax(decisionBelief);
    const search = aStar(botPos, CONFIG.HILL);
    const ctx = {
        botPos,
        nextCell: search.path.length > 1 ? search.path[1] : { r: botPos.r, c: botPos.c },
        dNow: manhattan(botPos, CONFIG.HILL),
        dNext: 0,
        shootTarget: argmax.cell,
        horizon: pressureTrace.horizon,
        stepValue: 0,
    };
    ctx.dNext = manhattan(ctx.nextCell, CONFIG.HILL);
    // A step is worth more when the horizon is only just long enough to reach
    // the hill; with time to spare, advancing is cheap and can wait a turn.
    ctx.stepValue = 0.35 + 0.65 * Math.min(1, ctx.dNow / Math.max(1, ctx.horizon));

    const U = {}, C = {}, net = {};
    for (const a of ACTIONS) {
        C[a] = commitmentCost(a, ctx);
        U[a] = bimatrixUtility(a, ctx);
        net[a] = U[a] - C[a];
    }

    const temperature = CONFIG.TAU * (1 - 0.5 * alpha);
    const policy = softmax(net, temperature);
    let action = sampleFrom(policy);
    let target;
    let gated = false;

    if (action === 'SHOOT') {
        target = argmax.cell;
        if (argmax.prob < CONFIG.THETA_SHOOT) {           // confidence gate
            gated = true;
            action = 'MOVE';                              // A* fallback
            target = ctx.nextCell;
        }
    } else if (action === 'MOVE') {
        target = ctx.nextCell;
    } else {
        target = { r: botPos.r, c: botPos.c };
    }

    telemetry = {
        U, C, net, policy, temperature, argmax, search, gated,
        pressure: pressureTrace, regime: regimeTrace,
        entropy: gridEntropy(decisionBelief),
        observation: pendingObs,
        sampled: action,
    };
    pendingObs = null;

    return { action, target };
}

/* ==========================================================================
   Match history (telemetry for the charts and the debrief)
   ========================================================================== */

let alphaHistory;        // [{ turn, alpha }]
let hillHistory;         // 'HUMAN' | 'BOT' | 'CONTESTED' | 'VACANT' per turn
let botActionCounts;     // { MOVE, SHOOT, HOLD }
let beliefSamples;       // predictive mass placed on the opponent's true cell
let busy = false;        // input lock while a turn animates
let overlayOn = false;

const sleep = ms => new Promise(r => setTimeout(r, ms));
const SVG_NS = 'http://www.w3.org/2000/svg';

/* ==========================================================================
   Game loop
   ========================================================================== */

function initGame() {
    turn = 1;
    scoreHuman = 0;
    scoreBot = 0;
    humanPos = { r: CONFIG.SPAWN_HUMAN.r, c: CONFIG.SPAWN_HUMAN.c };
    botPos = { r: CONFIG.SPAWN_BOT.r, c: CONFIG.SPAWN_BOT.c };
    gameOver = false;
    busy = false;

    selectedAction = 'MOVE';
    selectedTarget = null;

    initBelief();
    alpha = 0;
    regime = 'DEFENSIVE';
    opponentHistory = [];
    pendingObs = null;
    telemetry = null;
    decisionBelief = beliefGrid.map(row => row.slice());

    alphaHistory = [];
    hillHistory = [];
    botActionCounts = { MOVE: 0, SHOOT: 0, HOLD: 0 };
    beliefSamples = [];

    computePressure();
    document.getElementById('match-log').innerHTML = '';
    document.getElementById('tracers').innerHTML = '';
    hideGameOver();
    hideReveal();
    logMessage(`Match initialised. A at (${humanPos.r},${humanPos.c}), B at (${botPos.r},${botPos.c}). Hill (2,2), ${CONFIG.T} turns.`, 'system');
    setAction('MOVE');
    renderAll();
    positionTokens(false);
}

async function executeTurn() {
    if (gameOver || busy || !selectedTarget) return;
    busy = true;
    document.getElementById('btn-confirm').disabled = true;

    const decision = computeBotAction();

    const humanAct = selectedAction;
    const humanTarget = { r: selectedTarget.r, c: selectedTarget.c };
    const botAct = decision.action;
    const botTarget = { r: decision.target.r, c: decision.target.c };
    botActionCounts[botAct]++;

    logMessage(`Turn ${turn}`, 'turn');
    logMessage(`A  ${humanAct} → (${humanTarget.r},${humanTarget.c})`, 'human');
    logMessage(`B  ${botAct} → (${botTarget.r},${botTarget.c})${telemetry.gated ? '   [shoot gated]' : ''}`, 'bot');

    // The inspector reflects the decision the engine just made.
    renderDecision();
    renderBelief();
    renderPath();
    renderRegime();

    // ── Phase 1: simultaneous commit reveal ───────────────────────────────
    showReveal(
        `${humanAct} → (${humanTarget.r},${humanTarget.c})`,
        `${botAct} → (${botTarget.r},${botTarget.c})`
    );
    await sleep(560);

    // ── Phase 2: movement ─────────────────────────────────────────────────
    let nextHuman = humanAct === 'MOVE' ? humanTarget : { r: humanPos.r, c: humanPos.c };
    let nextBot = botAct === 'MOVE' ? botTarget : { r: botPos.r, c: botPos.c };
    const moved = humanAct === 'MOVE' || botAct === 'MOVE';
    humanPos = nextHuman;
    botPos = nextBot;
    positionTokens(true);
    if (moved) await sleep(430);

    // ── Phase 3: shots, resolved against post-move positions ──────────────
    const shots = [];
    if (humanAct === 'SHOOT') {
        const hitBot = same(humanTarget, botPos);
        shots.push({ from: humanTarget, to: humanTarget, side: 'a',
                     outcome: !hitBot ? 'MISS' : (botAct === 'HOLD' ? 'SHIELDED' : 'HIT') });
    }
    let shotOutcome = null, humanKnockedTo = null;
    if (botAct === 'SHOOT') {
        const hitHuman = same(botTarget, humanPos);
        shotOutcome = !hitHuman ? 'MISS' : (humanAct === 'HOLD' ? 'SHIELDED' : 'HIT');
        shots.push({ from: botTarget, to: botTarget, side: 'b', outcome: shotOutcome });
    }

    if (shots.length) {
        for (const s of shots) fireTracer(s, s.side === 'a' ? humanPos : botPos);
        await sleep(340);

        // Apply and narrate outcomes.
        if (humanAct === 'SHOOT') {
            const s = shots.find(x => x.side === 'a');
            if (s.outcome === 'SHIELDED') { flashToken('tok-b', 'shield'); logMessage(`B shielded the shot at (${humanTarget.r},${humanTarget.c}).`, 'system'); }
            else if (s.outcome === 'HIT') {
                flashToken('tok-b', 'hit');
                logMessage(`A hit B at (${humanTarget.r},${humanTarget.c}). B respawns.`, 'score');
                botPos = { r: CONFIG.SPAWN_BOT.r, c: CONFIG.SPAWN_BOT.c };
            } else logMessage(`A missed at (${humanTarget.r},${humanTarget.c}).`, 'system');
        }
        if (botAct === 'SHOOT') {
            if (shotOutcome === 'SHIELDED') { flashToken('tok-a', 'shield'); logMessage(`A shielded the shot at (${botTarget.r},${botTarget.c}).`, 'system'); }
            else if (shotOutcome === 'HIT') {
                flashToken('tok-a', 'hit');
                humanKnockedTo = { r: CONFIG.SPAWN_HUMAN.r, c: CONFIG.SPAWN_HUMAN.c };
                logMessage(`B hit A at (${botTarget.r},${botTarget.c}). A respawns.`, 'bot');
                humanPos = { r: humanKnockedTo.r, c: humanKnockedTo.c };
            } else logMessage(`B missed at (${botTarget.r},${botTarget.c}).`, 'system');
        }
        await sleep(260);
        positionTokens(true);
        if (humanKnockedTo || shots.some(s => s.outcome === 'HIT')) await sleep(380);
    }

    // ── Phase 4: hill resolution ──────────────────────────────────────────
    const humanOnHill = same(humanPos, CONFIG.HILL);
    const botOnHill = same(botPos, CONFIG.HILL);
    let hill;
    if (humanOnHill && botOnHill) { hill = 'CONTESTED'; logMessage('Hill contested — no points.', 'system'); }
    else if (humanOnHill) { hill = 'HUMAN'; scoreHuman++; logMessage(`A claims the hill. +1 (${scoreHuman})`, 'score'); }
    else if (botOnHill) { hill = 'BOT'; scoreBot++; logMessage(`B claims the hill. +1 (${scoreBot})`, 'bot'); }
    else { hill = 'VACANT'; logMessage('Hill vacant.', 'system'); }

    if (hill === 'HUMAN' || hill === 'BOT') {
        const side = hill === 'HUMAN' ? 'a' : 'b';
        flashHill(side);
        floatPoint(CONFIG.HILL, side);
        bumpScore(side);
        await sleep(480);
    } else if (hill === 'CONTESTED') {
        flashHill('c');
        await sleep(320);
    }

    // Public observations the engine may condition on next turn.
    pendingObs = {
        humanAction: humanAct,
        hill,
        shotTarget: botAct === 'SHOOT' ? botTarget : null,
        shotOutcome,
        humanKnockedTo,
    };
    opponentHistory.push(humanAct);

    // Evaluation only — never read by the engine.
    beliefSamples.push(decisionBelief[humanPos.r][humanPos.c]);
    alphaHistory.push({ turn, alpha });
    hillHistory.push(hill);

    turn++;
    computePressure();
    selectedTarget = null;
    hideReveal();

    if (turn > CONFIG.T) endMatch();
    else setAction('MOVE');

    renderAll();
    busy = false;
    renderPrompt();
}

function endMatch() {
    gameOver = true;
    const winner = scoreHuman > scoreBot ? 'HUMAN' : scoreBot > scoreHuman ? 'BOT' : 'DRAW';
    const text = winner === 'HUMAN' ? 'You win' : winner === 'BOT' ? 'SHCBO wins' : 'Draw';
    logMessage(`Match complete — ${text} (${scoreHuman}–${scoreBot})`, 'score');
    setTimeout(() => showGameOver(text, winner), 420);
}

/* ==========================================================================
   Input
   ========================================================================== */

function setAction(type) {
    if (gameOver || busy) return;
    selectedAction = type;
    selectedTarget = type === 'HOLD' ? { r: humanPos.r, c: humanPos.c } : null;

    document.querySelectorAll('.action-btn').forEach(b => {
        const on = b.dataset.action === type;
        b.classList.toggle('active', on);
        b.setAttribute('aria-pressed', String(on));
    });

    renderBoard();
    renderPrompt();
}

function cellClicked(r, c) {
    if (gameOver || busy || selectedAction === 'HOLD') return;
    if (selectedAction === 'MOVE' && manhattan({ r, c }, humanPos) > 1) {
        toast('MOVE is one orthogonal step — pick a highlighted cell.');
        return;
    }
    selectedTarget = { r, c };
    renderBoard();
    renderPrompt();
}

function nudge(dr, dc) {
    if (gameOver || busy || selectedAction === 'HOLD') return;
    const base = selectedAction === 'MOVE' ? humanPos : (selectedTarget || humanPos);
    const t = { r: base.r + dr, c: base.c + dc };
    if (!inBounds(t.r, t.c)) return;
    selectedTarget = t;
    renderBoard();
    renderPrompt();
}

function setupKeys() {
    window.addEventListener('keydown', e => {
        if (e.metaKey || e.ctrlKey || e.altKey) return;

        if (document.getElementById('rules-modal').classList.contains('active')) {
            if (e.key === 'Escape' || e.key === 'Enter') { e.preventDefault(); toggleRules(); }
            return;
        }
        if (document.getElementById('gameover').classList.contains('active')) {
            if (e.key === 'Enter' || e.key.toLowerCase() === 'r') { e.preventDefault(); initGame(); }
            return;
        }

        const arrows = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
        if (arrows[e.key]) { e.preventDefault(); nudge(arrows[e.key][0], arrows[e.key][1]); return; }

        const k = e.key.toLowerCase();
        if (e.key === '1') { e.preventDefault(); setAction('MOVE'); }
        else if (e.key === '2') { e.preventDefault(); setAction('SHOOT'); }
        else if (e.key === '3') { e.preventDefault(); setAction('HOLD'); }
        else if (e.key === 'Enter') { e.preventDefault(); executeTurn(); }
        else if (e.key === 'Escape') { e.preventDefault(); selectedTarget = null; renderBoard(); renderPrompt(); }
        else if (k === 'r') { e.preventDefault(); initGame(); }
        else if (k === 'b') { e.preventDefault(); toggleBeliefOverlay(); }
        else if (e.key === '?') { e.preventDefault(); toggleRules(); }
    });

    window.addEventListener('resize', () => positionTokens(false));
}

/* ==========================================================================
   Board geometry, tokens, effects
   ========================================================================== */

function cellEl(r, c) {
    return document.getElementById('board').children[r * CONFIG.GRID + c];
}

function cellCenter(cell) {
    const wrap = document.querySelector('.board-wrap');
    const el = cellEl(cell.r, cell.c);
    if (!el) return { x: 0, y: 0 };
    const w = wrap.getBoundingClientRect(), k = el.getBoundingClientRect();
    return { x: k.left - w.left + k.width / 2, y: k.top - w.top + k.height / 2 };
}

function positionTokens(animate) {
    const stacked = same(humanPos, botPos);
    place(document.getElementById('tok-a'), humanPos, animate, stacked ? -11 : 0);
    place(document.getElementById('tok-b'), botPos, animate, stacked ? 11 : 0);
}

function place(el, cell, animate, dx) {
    if (!el) return;
    const p = cellCenter(cell);
    if (!animate) {
        el.style.transition = 'none';
        el.style.left = `${p.x + dx}px`;
        el.style.top = `${p.y}px`;
        void el.offsetWidth;               // flush, so the next move animates
        el.style.transition = '';
    } else {
        el.style.left = `${p.x + dx}px`;
        el.style.top = `${p.y}px`;
    }
}

function fireTracer(shot, shooterPos) {
    const svg = document.getElementById('tracers');
    const wrap = document.querySelector('.board-wrap');
    const w = wrap.clientWidth, h = wrap.clientHeight;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

    const p1 = cellCenter(shooterPos);
    const p2 = cellCenter(shot.from);
    const colour = shot.side === 'a' ? 'var(--a)' : 'var(--b)';

    const line = document.createElementNS(SVG_NS, 'line');
    line.setAttribute('x1', p1.x); line.setAttribute('y1', p1.y);
    line.setAttribute('x2', p2.x); line.setAttribute('y2', p2.y);
    line.setAttribute('stroke', colour);
    line.setAttribute('stroke-width', '3');
    line.setAttribute('stroke-linecap', 'round');
    const len = Math.max(1, Math.hypot(p2.x - p1.x, p2.y - p1.y));
    line.style.strokeDasharray = len;
    line.style.strokeDashoffset = len;
    line.style.filter = 'drop-shadow(0 0 5px currentColor)';
    line.style.color = colour;
    svg.appendChild(line);

    const ring = document.createElementNS(SVG_NS, 'circle');
    ring.setAttribute('cx', p2.x); ring.setAttribute('cy', p2.y);
    ring.setAttribute('r', '6');
    ring.setAttribute('fill', 'none');
    ring.setAttribute('stroke', shot.outcome === 'MISS' ? 'var(--ink-3)' : colour);
    ring.setAttribute('stroke-width', '2');
    ring.style.opacity = '0';
    svg.appendChild(ring);

    requestAnimationFrame(() => {
        line.style.transition = 'stroke-dashoffset .2s linear, opacity .34s .2s';
        line.style.strokeDashoffset = '0';
        line.style.opacity = '0';
        ring.style.transition = 'r .38s ease-out .18s, opacity .38s ease-out .18s';
        ring.style.opacity = '1';
        setTimeout(() => { ring.setAttribute('r', '22'); ring.style.opacity = '0'; }, 190);
    });

    setTimeout(() => { line.remove(); ring.remove(); }, 900);
}

function flashToken(id, kind) {
    const el = document.getElementById(id);
    if (!el) return;
    let fx = el.querySelector('.tokfx');
    if (!fx) { fx = document.createElement('span'); fx.className = 'tokfx'; el.appendChild(fx); }
    el.classList.remove('hit', 'shield');
    void el.offsetWidth;
    el.classList.add(kind);
    setTimeout(() => el.classList.remove(kind), 750);
}

function flashHill(side) {
    const el = cellEl(CONFIG.HILL.r, CONFIG.HILL.c);
    if (!el) return;
    const cls = side === 'a' ? 'flash-a' : side === 'b' ? 'flash-b' : 'flash-a';
    el.classList.remove('flash-a', 'flash-b');
    void el.offsetWidth;
    el.classList.add(cls);
    setTimeout(() => el.classList.remove(cls), 800);
}

function floatPoint(cell, side) {
    const layer = document.getElementById('fx-layer');
    const p = cellCenter(cell);
    const el = document.createElement('div');
    el.className = 'floatpt';
    el.textContent = '+1';
    el.style.left = `${p.x}px`;
    el.style.top = `${p.y}px`;
    el.style.color = side === 'a' ? 'var(--a)' : 'var(--b)';
    layer.appendChild(el);
    setTimeout(() => el.remove(), 1200);
}

function bumpScore(side) {
    const el = document.getElementById(side === 'a' ? 'score-a' : 'score-b');
    el.classList.remove('bump');
    void el.offsetWidth;
    el.classList.add('bump');
}

function showReveal(aText, bText) {
    document.getElementById('reveal-a').textContent = aText;
    document.getElementById('reveal-b').textContent = bText;
    document.getElementById('reveal').classList.add('show');
}

function hideReveal() {
    document.getElementById('reveal').classList.remove('show');
}

/* ==========================================================================
   Rendering
   ========================================================================== */

function renderAll() {
    renderBoard();
    renderPrompt();
    renderStatus();
    renderBelief();
    renderPressure();
    renderPath();
    renderRegime();
    renderDecision();
    renderTimeline(document.getElementById('timeline'), true);
    renderAlphaChart(document.getElementById('alpha-chart'));
    positionTokens(true);
}

function renderBoard() {
    const board = document.getElementById('board');
    board.innerHTML = '';
    const path = telemetry ? telemetry.search.path : [];
    const argmax = gridArgmax(decisionBelief);

    forEachCell((r, c) => {
        const cell = document.createElement('button');
        cell.type = 'button';
        cell.className = 'cell';
        cell.tabIndex = -1;

        if (same({ r, c }, CONFIG.HILL)) cell.classList.add('is-hill');
        if (path.some(p => same(p, { r, c }))) cell.classList.add('on-path');
        if (!gameOver && !busy && selectedAction === 'MOVE' && manhattan({ r, c }, humanPos) <= 1) cell.classList.add('legal');
        if (!gameOver && !busy && selectedAction === 'SHOOT') cell.classList.add('targetable');
        if (selectedTarget && same(selectedTarget, { r, c })) {
            cell.classList.add('selected', selectedAction === 'SHOOT' ? 'sel-shoot' : 'sel-move');
        }

        // Belief projected onto the arena (toggled by the overlay button).
        const p = decisionBelief[r][c];
        const fill = document.createElement('span');
        fill.className = 'bfill';
        cell.style.setProperty('--bt', Math.pow(Math.min(1, p / 0.6), 0.65).toFixed(3));
        cell.appendChild(fill);

        const num = document.createElement('span');
        num.className = 'bnum';
        num.textContent = p >= 0.005 ? `${Math.round(p * 100)}%` : '';
        cell.appendChild(num);
        if (same({ r, c }, argmax.cell) && argmax.prob > 0.02) cell.classList.add('b-max');

        const coord = document.createElement('span');
        coord.className = 'coord';
        coord.textContent = `${r},${c}`;
        cell.appendChild(coord);

        let label = `Cell ${r},${c}`;
        if (same({ r, c }, CONFIG.HILL)) label += ', hill';
        if (same(humanPos, { r, c })) label += ', you';
        if (same(botPos, { r, c })) label += ', bot';
        cell.setAttribute('aria-label', label);

        cell.onclick = () => cellClicked(r, c);
        board.appendChild(cell);
    });
}

function renderPrompt() {
    const el = document.getElementById('prompt');
    const btn = document.getElementById('btn-confirm');
    if (gameOver) {
        el.innerHTML = 'Match complete. Press <kbd>R</kbd> to play again.';
        btn.disabled = true;
        return;
    }
    if (busy) {
        el.innerHTML = 'Resolving…';
        btn.disabled = true;
        return;
    }
    if (selectedAction === 'HOLD') {
        el.innerHTML = `Holding <b>(${humanPos.r},${humanPos.c})</b> — blocks an incoming shot. <kbd>Enter</kbd> to confirm.`;
    } else if (!selectedTarget) {
        el.innerHTML = selectedAction === 'MOVE'
            ? 'Pick an adjacent cell — click, or <span class="keygroup"><kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd></span>'
            : 'Pick any cell to fire at — click, or <span class="keygroup"><kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd></span>';
    } else {
        el.innerHTML = `<b>${selectedAction} → (${selectedTarget.r},${selectedTarget.c})</b> · <kbd>Enter</kbd> confirm · <kbd>Esc</kbd> clear`;
    }
    btn.disabled = !selectedTarget;
}

function renderStatus() {
    document.getElementById('turn-val').textContent = `${Math.min(turn, CONFIG.T)}`;
    document.getElementById('score-a').textContent = scoreHuman;
    document.getElementById('score-b').textContent = scoreBot;

    const bar = document.getElementById('horizon');
    bar.innerHTML = '';
    for (let i = 1; i <= CONFIG.T; i++) {
        const pip = document.createElement('span');
        pip.className = 'pip' + (i < turn ? ' done' : i === turn ? ' now' : '');
        bar.appendChild(pip);
    }
}

function renderBelief() {
    const grid = document.getElementById('belief');
    grid.innerHTML = '';
    const argmax = gridArgmax(decisionBelief);

    forEachCell((r, c) => {
        const p = decisionBelief[r][c];
        const d = document.createElement('div');
        d.className = 'bcell';
        const t = Math.pow(Math.min(1, p / 0.6), 0.65);
        d.style.setProperty('--t', t.toFixed(3));
        if (t > 0.55) d.classList.add('on-dark');
        d.textContent = p >= 0.005 ? Math.round(p * 100) : '·';
        if (same({ r, c }, argmax.cell) && argmax.prob > 0) d.classList.add('argmax');
        if (same({ r, c }, CONFIG.HILL)) d.classList.add('bhill');
        grid.appendChild(d);
    });

    const open = argmax.prob >= CONFIG.THETA_SHOOT;
    const gateEl = document.getElementById('gate');
    gateEl.textContent = open ? 'SHOOT ENABLED' : 'SHOOT GATED';
    gateEl.className = 'gate-tag ' + (open ? 'open' : 'shut');

    document.getElementById('gate-detail').innerHTML =
        `max B<sub>t</sub> <b>${argmax.prob.toFixed(3)}</b> vs θ ${CONFIG.THETA_SHOOT.toFixed(2)} · entropy <b>${gridEntropy(decisionBelief).toFixed(2)}</b> bits`;

    const obs = telemetry && telemetry.observation;
    document.getElementById('last-obs').textContent = obs
        ? `Conditioned on: ${obs.humanAction}, hill ${obs.hill}${obs.shotOutcome ? `, shot ${obs.shotOutcome}` : ''}`
        : 'No observations yet — this is the spawn prior projected one step forward.';
}

function renderPressure() {
    document.getElementById('alpha-fill').style.width = `${(alpha * 100).toFixed(1)}%`;
    document.getElementById('alpha-val').textContent = alpha.toFixed(3);

    const tag = document.getElementById('alpha-tag');
    if (alpha > 0.6) { tag.textContent = 'ESCALATED'; tag.dataset.level = 'hi'; }
    else if (alpha > 0.3) { tag.textContent = 'BALANCED'; tag.dataset.level = 'mid'; }
    else { tag.textContent = 'PATIENT'; tag.dataset.level = 'lo'; }

    const horizon = CONFIG.T - Math.min(turn, CONFIG.T) + 1;
    const ind = scoreHuman >= scoreBot ? 1 : 0;
    document.getElementById('alpha-num').textContent = `(${scoreHuman - scoreBot}) + 1/${horizon}·${ind}`;
    document.getElementById('alpha-den').textContent = (CONFIG.T - Math.min(turn, CONFIG.T) + 1.5).toFixed(1);
}

function renderPath() {
    const line = document.getElementById('path-line');
    const meta = document.getElementById('path-meta');
    const rows = document.getElementById('cost-rows');

    if (!telemetry) {
        line.textContent = 'Awaiting first turn';
        meta.textContent = '';
        rows.innerHTML = '';
        return;
    }
    const { search, C } = telemetry;
    line.textContent = search.path.map(p => `(${p.r},${p.c})`).join('  →  ');
    meta.textContent = `f = g + h · cost ${search.cost} · ${search.expansions} expansions`;

    const formula = { MOVE: '0', HOLD: '0.15(1−α)', SHOOT: '0.40(1−B<sub>t</sub>)' };
    rows.innerHTML = ACTIONS.map(a => `
        <div class="cost-row">
            <span class="ca">${a}</span>
            <span class="cf">${formula[a]}</span>
            <span class="cv">${C[a].toFixed(3)}</span>
        </div>`).join('');
}

function renderRegime() {
    const trace = telemetry
        ? telemetry.regime
        : { freq: { MOVE: 0, SHOOT: 0, HOLD: 0 }, n: 0, reason: 'awaiting first turn' };

    document.querySelectorAll('.fsm-state').forEach(el => {
        el.classList.toggle('active', el.dataset.regime === regime);
    });
    document.getElementById('regime-reason').textContent = trace.reason;

    document.getElementById('freq-rows').innerHTML = ACTIONS.map(a => {
        const f = trace.freq[a] || 0;
        return `<div class="freq-row">
            <span class="fname">${a}</span>
            <span class="freq-track"><span class="freq-fill" style="width:${(f * 100).toFixed(0)}%"></span></span>
            <span class="freq-val">${f.toFixed(2)}</span>
        </div>`;
    }).join('');

    const strip = opponentHistory.slice(-CONFIG.W);
    document.getElementById('window-strip').innerHTML = strip.length
        ? strip.map(a => `<span class="wchip w-${a.toLowerCase()}">${a[0]}</span>`).join('')
        : '<span class="wchip empty">window empty</span>';
}

function renderDecision() {
    const body = document.getElementById('decision-rows');
    const temp = document.getElementById('temp-val');

    if (!telemetry) {
        body.innerHTML = '<div class="empty-note">Confirm a turn to see the engine trace.</div>';
        temp.textContent = '—';
        return;
    }
    const { U, C, net, policy, temperature, sampled, gated } = telemetry;
    temp.innerHTML = `τ(1−0.5α) = <b>${temperature.toFixed(3)}</b>`;

    body.innerHTML = ACTIONS.map(a => `
        <div class="drow${a === sampled ? ' picked' : ''}">
            <span class="dname">${a}</span>
            <span class="dnum">${U[a].toFixed(3)}</span>
            <span class="dnum">${C[a].toFixed(3)}</span>
            <span class="dnum strong">${net[a].toFixed(3)}</span>
            <span class="dbar"><span class="dbar-fill" style="width:${(policy[a] * 100).toFixed(1)}%"></span></span>
            <span class="dpct">${(policy[a] * 100).toFixed(1)}%</span>
        </div>`).join('')
        + (gated
            ? '<div class="gate-note">SHOOT was sampled, then suppressed by the confidence gate — fell back to the A* move.</div>'
            : '');
}

function renderTimeline(el, markNow) {
    if (!el) return;
    const cls = { HUMAN: 't-a', BOT: 't-b', CONTESTED: 't-c', VACANT: '' };
    let html = '';
    for (let i = 0; i < CONFIG.T; i++) {
        const h = hillHistory[i];
        const now = markNow && (i + 1) === turn && !gameOver ? ' t-now' : '';
        html += `<div class="tcell ${h ? cls[h] : ''}${now}">${i + 1}</div>`;
    }
    el.innerHTML = html;
}

function renderAlphaChart(svg) {
    if (!svg) return;
    const W = 260, H = 84, L = 20, R = 6, T = 8, B = 16;
    const x = t => L + ((t - 1) / (CONFIG.T - 1)) * (W - L - R);
    const y = v => T + (1 - v) * (H - T - B);

    let html = '';
    for (const v of [0, 0.5, 1]) {
        html += `<line class="gridline" x1="${L}" y1="${y(v).toFixed(1)}" x2="${W - R}" y2="${y(v).toFixed(1)}"/>`;
        html += `<text class="clabel" x="2" y="${(y(v) + 2.5).toFixed(1)}">${v.toFixed(1)}</text>`;
    }
    html += `<line class="axis" x1="${L}" y1="${y(0)}" x2="${W - R}" y2="${y(0)}"/>`;
    html += `<text class="clabel" x="${L}" y="${H - 3}">t1</text>`;
    html += `<text class="clabel" x="${W - R - 12}" y="${H - 3}">t${CONFIG.T}</text>`;

    const pts = alphaHistory.map(d => [x(d.turn), y(d.alpha)]);
    if (pts.length) {
        const line = pts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
        html += `<polygon class="alpha-area" points="${pts[0][0].toFixed(1)},${y(0).toFixed(1)} ${line} ${pts[pts.length - 1][0].toFixed(1)},${y(0).toFixed(1)}"/>`;
        html += `<polyline class="alpha-line" points="${line}"/>`;
        for (const p of pts) html += `<circle class="dot" cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="2.2"/>`;
    } else {
        html += `<text class="clabel" x="${W / 2 - 30}" y="${H / 2}">no turns played yet</text>`;
    }
    svg.innerHTML = html;
}

/* ==========================================================================
   Chrome
   ========================================================================== */

function logMessage(msg, type = 'system') {
    const log = document.getElementById('match-log');
    const el = document.createElement('div');
    el.className = `log-line ${type}`;
    el.textContent = msg;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
}

let toastTimer;
function toast(msg) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 2400);
}

function toggleRules() {
    document.getElementById('rules-modal').classList.toggle('active');
}

function toggleBeliefOverlay() {
    overlayOn = !overlayOn;
    document.body.classList.toggle('overlay-on', overlayOn);
    document.getElementById('overlay-toggle').setAttribute('aria-pressed', String(overlayOn));
}

function toggleTheme() {
    const root = document.documentElement;
    const next = root.dataset.theme === 'light' ? 'dark' : 'light';
    root.dataset.theme = next;
    try { localStorage.setItem('koth-theme', next); } catch (_) {}
    setTimeout(() => positionTokens(false), 30);
}

function showGameOver(text, winner) {
    document.getElementById('go-title').textContent = text;
    document.getElementById('go-a').textContent = scoreHuman;
    document.getElementById('go-b').textContent = scoreBot;
    document.getElementById('go-sub').textContent =
        winner === 'DRAW' ? `Level after ${CONFIG.T} turns.` : `Final score after ${CONFIG.T} turns.`;

    renderAlphaChart(document.getElementById('go-chart'));
    renderTimeline(document.getElementById('go-timeline'), false);

    const total = ACTIONS.reduce((s, a) => s + botActionCounts[a], 0) || 1;
    document.getElementById('go-mix').innerHTML = ACTIONS.map(a => {
        const pct = 100 * botActionCounts[a] / total;
        return `<div class="mixrow">
            <span>${a}</span>
            <span class="mixtrack"><span class="mixfill" style="width:${pct.toFixed(0)}%"></span></span>
            <span>${pct.toFixed(0)}%</span>
        </div>`;
    }).join('');

    const mean = beliefSamples.length
        ? beliefSamples.reduce((x, y) => x + y, 0) / beliefSamples.length
        : 0;
    document.getElementById('go-belief').textContent = mean.toFixed(3);

    document.getElementById('gameover').classList.add('active');
}

function hideGameOver() {
    document.getElementById('gameover').classList.remove('active');
}

function resetGame() { initGame(); }

document.addEventListener('DOMContentLoaded', () => {
    try {
        const saved = localStorage.getItem('koth-theme');
        if (saved) document.documentElement.dataset.theme = saved;
    } catch (_) {}
    setupKeys();
    initGame();
    setTimeout(() => positionTokens(false), 60);
});
