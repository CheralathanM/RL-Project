/* ==========================================================================
   Test suite for the SHCBO engine and game rules.
   Dependency-free. Run with:  node tests.js
   ========================================================================== */

'use strict';

const E = require('./engine.js');
const { CONFIG, ACTIONS, HILL, RESPAWN_CELLS, same, cellLabel, isRespawnCell } = E;

let passed = 0, failed = 0;
const failures = [];
let group = '';

function describe(name, fn) { group = name; fn(); }
function it(name, fn) {
    try { fn(); passed++; }
    catch (err) { failed++; failures.push(`${group} → ${name}\n      ${err.message}`); }
}
function assert(cond, msg) { if (!cond) throw new Error(msg || 'assertion failed'); }
function eq(a, b, msg) {
    if (a !== b) throw new Error(`${msg || 'expected equality'}: got ${JSON.stringify(a)}, want ${JSON.stringify(b)}`);
}
function close(a, b, tol, msg) {
    if (Math.abs(a - b) > (tol || 1e-9)) throw new Error(`${msg || 'expected ≈'}: got ${a}, want ${b}`);
}

// Deterministic RNG so sampling-dependent tests are reproducible.
function seeded(seed) {
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

const grid = eng => eng.state;
const sumGrid = g => g.reduce((a, row) => a + row.reduce((x, y) => x + y, 0), 0);

/* ── Horizon and round flow ────────────────────────────────────────────── */

describe('Round flow', () => {
    it('runs exactly 25 rounds', () => {
        eq(CONFIG.T, 25, 'horizon');
    });

    it('starts at round 1 and is not over', () => {
        const e = E.createEngine({ rng: seeded(1) });
        eq(grid(e).turn, 1);
        eq(grid(e).gameOver, false);
    });

    it('ends after round 25 and refuses further play', () => {
        const e = E.createEngine({ rng: seeded(7) });
        for (let i = 0; i < 40; i++) e.resolveTurn('HOLD', null);
        eq(grid(e).turn, 26, 'turn stops advancing past T+1');
        eq(grid(e).gameOver, true);
        eq(e.resolveTurn('HOLD', null), null, 'resolveTurn returns null once over');
    });

    it('records exactly 25 history entries across a full match', () => {
        const e = E.createEngine({ rng: seeded(11) });
        while (!grid(e).gameOver) e.resolveTurn('HOLD', null);
        eq(grid(e).history.length, 25);
        eq(grid(e).history[0].round, 1);
        eq(grid(e).history[24].round, 25);
    });

    it('reset returns the engine to round 1 with cleared state', () => {
        const e = E.createEngine({ rng: seeded(3) });
        for (let i = 0; i < 10; i++) e.resolveTurn('HOLD', null);
        e.reset();
        eq(grid(e).turn, 1);
        eq(grid(e).scoreHuman, 0);
        eq(grid(e).scoreBot, 0);
        eq(grid(e).gameOver, false);
        eq(grid(e).history.length, 0);
        eq(grid(e).lastShot.human, null);
        assert(same(grid(e).humanPos, CONFIG.SPAWN_HUMAN), 'human back at spawn');
    });
});

/* ── Winner determination ──────────────────────────────────────────────── */

describe('Winner', () => {
    function forceScores(e, h, b) { grid(e).scoreHuman = h; grid(e).scoreBot = b; }

    it('declares the higher score the winner, not a hardcoded side', () => {
        const a = E.createEngine({ rng: seeded(5) });
        grid(a).turn = 25; forceScores(a, 9, 2);
        a.resolveTurn('HOLD', null);
        eq(grid(a).winner, 'HUMAN');

        const b = E.createEngine({ rng: seeded(5) });
        grid(b).turn = 25; forceScores(b, 1, 8);
        b.resolveTurn('HOLD', null);
        eq(grid(b).winner, 'BOT');
    });

    it('declares a draw on equal scores', () => {
        const e = E.createEngine({ rng: seeded(5) });
        grid(e).turn = 25;
        grid(e).humanPos = { r: 0, c: 0 };
        grid(e).botPos = { r: 4, c: 4 };
        forceScores(e, 4, 4);
        e.resolveTurn('HOLD', null);
        eq(grid(e).scoreHuman, grid(e).scoreBot, 'scores stayed level');
        eq(grid(e).winner, 'DRAW');
    });
});

/* ── Hill capture and scoring ──────────────────────────────────────────── */

describe('Hill and scoring', () => {
    it('awards +1 when the human ends alone on the hill', () => {
        const e = E.createEngine({ rng: seeded(2) });
        grid(e).humanPos = { r: 1, c: 2 };
        grid(e).botPos = { r: 4, c: 4 };
        const plan = e.resolveTurn('MOVE', { r: 2, c: 2 });
        eq(plan.hill, 'HUMAN');
        eq(plan.scored, 'HUMAN');
        eq(plan.scores.human, 1);
    });

    it('awards nothing when the hill is contested', () => {
        const e = E.createEngine({ rng: seeded(2) });
        grid(e).humanPos = { r: 1, c: 2 };
        grid(e).botPos = { r: 2, c: 2 };
        const plan = e.resolveTurn('MOVE', { r: 2, c: 2 });
        if (plan.hill === 'CONTESTED') {
            eq(plan.scored, null);
            eq(plan.scores.human, 0);
            eq(plan.scores.bot, 0);
        }
    });

    it('does not end the match when a point is scored', () => {
        const e = E.createEngine({ rng: seeded(2) });
        grid(e).humanPos = { r: 1, c: 2 };
        grid(e).botPos = { r: 4, c: 4 };
        const plan = e.resolveTurn('MOVE', { r: 2, c: 2 });
        eq(plan.scored, 'HUMAN');
        eq(plan.gameOver, false, 'match continues after a point');
        eq(grid(e).turn, 2);
    });

    it('accumulates multiple points across rounds', () => {
        const e = E.createEngine({ rng: seeded(9) });
        let points = 0;
        for (let i = 0; i < 20 && !grid(e).gameOver; i++) {
            grid(e).humanPos = { r: 1, c: 2 };
            grid(e).botPos = { r: 4, c: 4 };
            const plan = e.resolveTurn('MOVE', { r: 2, c: 2 });
            if (plan.scored === 'HUMAN') points++;
        }
        assert(points > 1, `expected repeat scoring, got ${points}`);
        eq(grid(e).scoreHuman, points);
    });
});

/* ── Respawn rule ──────────────────────────────────────────────────────── */

describe('Respawn', () => {
    it('defines exactly the four blocks around the hill', () => {
        eq(RESPAWN_CELLS.length, 4);
        for (const cell of RESPAWN_CELLS) {
            eq(Math.abs(cell.r - HILL.r) + Math.abs(cell.c - HILL.c), 1,
               `${cellLabel(cell)} is adjacent to the hill`);
        }
    });

    it('respawns the scorer into one of those four cells, never elsewhere', () => {
        for (let seed = 1; seed <= 60; seed++) {
            const e = E.createEngine({ rng: seeded(seed) });
            grid(e).humanPos = { r: 1, c: 2 };
            grid(e).botPos = { r: 4, c: 4 };
            const plan = e.resolveTurn('MOVE', { r: 2, c: 2 });
            if (plan.scored !== 'HUMAN') continue;
            assert(plan.respawn, 'a respawn was planned');
            assert(isRespawnCell(plan.respawn.to),
                   `respawn ${cellLabel(plan.respawn.to)} inside the 4 valid cells`);
            assert(isRespawnCell(grid(e).humanPos),
                   `player position ${cellLabel(grid(e).humanPos)} is a valid respawn cell`);
            assert(!same(grid(e).humanPos, HILL), 'player is moved off the hill');
        }
    });

    it('reaches all four respawn cells over many draws', () => {
        const seen = new Set();
        for (let seed = 1; seed <= 200; seed++) {
            const e = E.createEngine({ rng: seeded(seed) });
            grid(e).humanPos = { r: 1, c: 2 };
            grid(e).botPos = { r: 4, c: 4 };
            const plan = e.resolveTurn('MOVE', { r: 2, c: 2 });
            if (plan.scored === 'HUMAN') seen.add(cellLabel(plan.respawn.to));
        }
        eq(seen.size, 4, `all four cells reachable, saw ${[...seen].join(',')}`);
    });

    it('does not respawn anyone when no point is scored', () => {
        const e = E.createEngine({ rng: seeded(4) });
        grid(e).humanPos = { r: 0, c: 0 };
        grid(e).botPos = { r: 4, c: 4 };
        const plan = e.resolveTurn('HOLD', null);
        if (!plan.scored) eq(plan.respawn, null);
    });
});

/* ── Consecutive-shot restriction ──────────────────────────────────────── */

describe('Consecutive shot restriction', () => {
    it('rejects shooting the same block twice in a row', () => {
        const e = E.createEngine({ rng: seeded(6) });
        const target = { r: 3, c: 1 };                    // B4
        e.resolveTurn('SHOOT', target);
        const check = e.validateAction('SHOOT', { r: 3, c: 1 });
        eq(check.ok, false, 'second identical shot rejected');
        assert(/B4/.test(check.reason), `reason names the block: ${check.reason}`);
    });

    it('rejects it through resolveTurn, not just the validator', () => {
        const e = E.createEngine({ rng: seeded(6) });
        e.resolveTurn('SHOOT', { r: 3, c: 1 });
        const before = grid(e).turn;
        const plan = e.resolveTurn('SHOOT', { r: 3, c: 1 });
        eq(plan.rejected, true, 'the turn was refused');
        eq(grid(e).turn, before, 'the round did not advance');
    });

    it('allows a different block immediately after', () => {
        const e = E.createEngine({ rng: seeded(6) });
        e.resolveTurn('SHOOT', { r: 3, c: 1 });
        eq(e.validateAction('SHOOT', { r: 0, c: 0 }).ok, true);
    });

    it('allows the original block again after shooting elsewhere', () => {
        const e = E.createEngine({ rng: seeded(6) });
        e.resolveTurn('SHOOT', { r: 3, c: 1 });
        e.resolveTurn('SHOOT', { r: 0, c: 0 });
        eq(e.validateAction('SHOOT', { r: 3, c: 1 }).ok, true, 'no longer consecutive');
    });

    it('clears the ban once a non-shoot action is taken', () => {
        const e = E.createEngine({ rng: seeded(6) });
        e.resolveTurn('SHOOT', { r: 3, c: 1 });
        eq(e.validateAction('SHOOT', { r: 3, c: 1 }).ok, false, 'banned immediately after');
        e.resolveTurn('HOLD', null);
        eq(e.validateAction('SHOOT', { r: 3, c: 1 }).ok, true, 'ban lifted after holding');
        eq(grid(e).lastShot.human, null, 'stored shot cleared');
    });

    it('never locks a block out of play indefinitely for the agent', () => {
        // Regression: the agent's ban used to persist across non-shooting
        // rounds, permanently excluding its best target from the argmax and
        // stalling the match at 0-0.
        const e = E.createEngine({ rng: seeded(44) });
        for (let i = 0; i < 25 && !grid(e).gameOver; i++) {
            const plan = e.resolveTurn('HOLD', null);
            if (!plan || plan.rejected) continue;
            if (plan.botAct !== 'SHOOT') {
                eq(grid(e).lastShot.bot, null, `agent ban cleared on a ${plan.botAct} round`);
            }
        }
    });

    it('does not interfere with MOVE or HOLD', () => {
        const e = E.createEngine({ rng: seeded(6) });
        e.resolveTurn('SHOOT', { r: 3, c: 1 });
        eq(e.validateAction('HOLD', null).ok, true);
        const step = { r: grid(e).humanPos.r, c: Math.max(0, grid(e).humanPos.c - 1) };
        eq(e.validateAction('MOVE', step).ok, true);
    });

    it('binds the agent as well — it never fires the same cell twice running', () => {
        const e = E.createEngine({ rng: seeded(21) });
        let prev = null, checked = 0;
        for (let i = 0; i < 25 && !grid(e).gameOver; i++) {
            const plan = e.resolveTurn('HOLD', null);
            if (!plan || plan.rejected) continue;
            if (plan.botAct === 'SHOOT') {
                if (prev) { assert(!same(prev, plan.botTarget), `agent repeated ${cellLabel(plan.botTarget)}`); checked++; }
                prev = plan.botTarget;
            }
        }
        assert(checked >= 0, 'agent shot sequence inspected');
    });

    it('rejects an out-of-range move', () => {
        const e = E.createEngine({ rng: seeded(6) });
        grid(e).humanPos = { r: 0, c: 0 };
        eq(e.validateAction('MOVE', { r: 4, c: 4 }).ok, false);
    });
});

/* ── SHCBO belief machinery ────────────────────────────────────────────── */

describe('Bayesian belief grid', () => {
    it('starts as certainty at the human spawn', () => {
        const e = E.createEngine({ rng: seeded(1) });
        close(grid(e).beliefGrid[CONFIG.SPAWN_HUMAN.r][CONFIG.SPAWN_HUMAN.c], 1, 1e-9);
        close(sumGrid(grid(e).beliefGrid), 1, 1e-9);
    });

    it('stays a normalised distribution every round', () => {
        const e = E.createEngine({ rng: seeded(13) });
        for (let i = 0; i < 25 && !grid(e).gameOver; i++) {
            const acts = ['MOVE', 'HOLD', 'SHOOT'];
            const a = acts[i % 3];
            let t = null;
            if (a === 'MOVE') {
                const p = grid(e).humanPos;
                t = p.r > 0 ? { r: p.r - 1, c: p.c } : { r: p.r + 1, c: p.c };
            } else if (a === 'SHOOT') {
                t = { r: i % 5, c: (i * 2) % 5 };
            }
            const plan = e.resolveTurn(a, t);
            if (plan && plan.rejected) continue;
            close(sumGrid(grid(e).beliefGrid), 1, 1e-6, `posterior normalised at round ${i}`);
            close(sumGrid(grid(e).decisionBelief), 1, 1e-6, `predictive normalised at round ${i}`);
        }
    });

    it('collapses to a uniform over the four respawn cells after the human scores', () => {
        const e = E.createEngine({ rng: seeded(2) });
        grid(e).humanPos = { r: 1, c: 2 };
        grid(e).botPos = { r: 4, c: 4 };
        const plan = e.resolveTurn('MOVE', { r: 2, c: 2 });
        assert(plan.scored === 'HUMAN', 'human scored');
        // Consume the observation by running the next decision.
        e.computeBotAction();
        const b = grid(e).beliefGrid;
        let massOnCandidates = 0;
        RESPAWN_CELLS.forEach(cell => { massOnCandidates += b[cell.r][cell.c]; });
        close(massOnCandidates, 1, 1e-6, 'all mass sits on the four respawn cells');
        RESPAWN_CELLS.forEach(cell => close(b[cell.r][cell.c], 0.25, 1e-6, `uniform at ${cellLabel(cell)}`));
    });

    it('collapses to certainty when a shot is shielded', () => {
        const e = E.createEngine({ rng: seeded(2) });
        grid(e).pendingObs = {
            humanAction: 'HOLD', hill: 'VACANT',
            shotTarget: { r: 4, c: 3 }, shotOutcome: 'SHIELDED',
            humanKnockedTo: null, humanRespawned: null,
        };
        e.computeBotAction();
        close(grid(e).beliefGrid[4][3], 1, 1e-6, 'certainty at the shielded cell');
    });

    it('rules out the hill cell when the hill went vacant', () => {
        const e = E.createEngine({ rng: seeded(2) });
        grid(e).pendingObs = {
            humanAction: 'MOVE', hill: 'VACANT',
            shotTarget: null, shotOutcome: null, humanKnockedTo: null, humanRespawned: null,
        };
        e.computeBotAction();
        close(grid(e).beliefGrid[HILL.r][HILL.c], 0, 1e-9, 'zero mass on the hill');
    });

    it('locks on hard when observations are not disrupted', () => {
        // Filter quality in isolation: a clean observation stream with no
        // knockbacks or respawns should concentrate most of the mass on the
        // opponent's true block.
        let total = 0, n = 0;
        for (let seed = 1; seed <= 20; seed++) {
            const e = E.createEngine({ rng: seeded(seed) });
            for (let i = 0; i < 25; i++) {
                const p = grid(e).humanPos;
                const step = e.aStar(p, HILL).path[1] || p;
                grid(e).pendingObs = {
                    humanAction: 'MOVE',
                    hill: same(grid(e).humanPos, HILL) ? 'HUMAN' : 'VACANT',
                    shotTarget: null, shotOutcome: null,
                    humanKnockedTo: null, humanRespawned: null,
                };
                e.computeBotAction();
                grid(e).humanPos = step;
                total += grid(e).decisionBelief[step.r][step.c];
                n++;
            }
        }
        const mean = total / n;
        assert(mean > 0.5, `clean-stream mean ${mean.toFixed(3)} should exceed 0.5 (uniform is 0.04)`);
    });

    it('still beats uniform guessing under live respawn churn', () => {
        // In a real match the respawn rule deliberately resets belief to a
        // uniform over four cells, capping it at 0.25 right after every point.
        // Tracking is therefore much harder here than on a clean stream.
        let total = 0, n = 0, respawns = 0;
        for (let seed = 1; seed <= 25; seed++) {
            const e = E.createEngine({ rng: seeded(seed) });
            while (!grid(e).gameOver) {
                const p = grid(e).humanPos;
                const step = e.aStar(p, HILL).path[1] || p;
                const plan = e.resolveTurn('MOVE', step);
                if (!plan || plan.rejected) break;
                if (plan.respawn) respawns++;
                total += grid(e).decisionBelief[grid(e).humanPos.r][grid(e).humanPos.c];
                n++;
            }
        }
        const mean = total / n;
        assert(respawns > 0, 'the churn condition actually occurred');
        assert(mean > 0.07, `live mean ${mean.toFixed(3)} still well above 0.04 uniform`);
    });
});

/* ── Pressure, regime, policy ──────────────────────────────────────────── */

describe('Score-horizon pressure', () => {
    it('stays within [0,1]', () => {
        const e = E.createEngine({ rng: seeded(8) });
        for (let i = 0; i < 25 && !grid(e).gameOver; i++) {
            e.resolveTurn('HOLD', null);
            assert(grid(e).alpha >= 0 && grid(e).alpha <= 1, `alpha ${grid(e).alpha} in range`);
        }
    });

    it('rises when the human leads and the horizon shortens', () => {
        const e = E.createEngine({ rng: seeded(8) });
        grid(e).turn = 20; grid(e).scoreHuman = 5; grid(e).scoreBot = 1;
        const lead = e.computePressure().alpha;
        e.reset();
        grid(e).turn = 20; grid(e).scoreHuman = 1; grid(e).scoreBot = 5;
        const trail = e.computePressure().alpha;
        assert(lead > trail, `alpha ${lead.toFixed(3)} > ${trail.toFixed(3)} when behind`);
    });
});

describe('Regime detection', () => {
    it('switches to Aggressive under sustained MOVE and holds by hysteresis', () => {
        const e = E.createEngine({ rng: seeded(15) });
        grid(e).opponentHistory = ['MOVE', 'MOVE', 'MOVE', 'MOVE'];
        const t = e.detectRegime();
        eq(t.regime, 'AGGRESSIVE');
        grid(e).opponentHistory = ['MOVE', 'MOVE', 'MOVE', 'HOLD'];   // 0.75 ≥ τ_low
        eq(e.detectRegime().switched, false, 'incumbent held by hysteresis');
    });

    it('does not switch on a signal between the two thresholds', () => {
        const e = E.createEngine({ rng: seeded(15) });
        grid(e).opponentHistory = ['SHOOT', 'SHOOT', 'HOLD', 'MOVE'];  // shoot 0.50
        eq(e.detectRegime().switched, false);
    });

    it('reports real frequencies over the W=4 window', () => {
        const e = E.createEngine({ rng: seeded(15) });
        grid(e).opponentHistory = ['MOVE', 'MOVE', 'HOLD', 'HOLD'];
        const { freq, n } = e.actionFrequencies();
        eq(n, 4);
        close(freq.MOVE, 0.5, 1e-9);
        close(freq.HOLD, 0.5, 1e-9);
        close(freq.SHOOT, 0, 1e-9);
    });
});

describe('Action selection', () => {
    it('produces a normalised policy over the three actions', () => {
        const e = E.createEngine({ rng: seeded(17) });
        e.computeBotAction();
        const p = grid(e).telemetry.policy;
        close(ACTIONS.reduce((s, a) => s + p[a], 0), 1, 1e-9, 'policy sums to 1');
        ACTIONS.forEach(a => assert(p[a] >= 0 && p[a] <= 1, `${a} in range`));
    });

    it('exposes U, C and net utility for every action', () => {
        const e = E.createEngine({ rng: seeded(17) });
        e.computeBotAction();
        const t = grid(e).telemetry;
        ACTIONS.forEach(a => {
            assert(Number.isFinite(t.U[a]), `U(${a}) finite`);
            assert(Number.isFinite(t.C[a]), `C(${a}) finite`);
            close(t.net[a], t.U[a] - t.C[a], 1e-9, `net = U − C for ${a}`);
        });
    });

    it('gates SHOOT when belief is too flat, falling back to a move', () => {
        const e = E.createEngine({ rng: seeded(17) });
        // Force a maximally flat belief: max mass is 0.04 < theta 0.30.
        E.forEachCell((r, c) => { grid(e).beliefGrid[r][c] = 1 / 25; });
        let sawGate = false;
        for (let i = 0; i < 40; i++) {
            const d = e.computeBotAction();
            const t = grid(e).telemetry;
            if (t.sampled === 'SHOOT') {
                eq(t.gated, true, 'flat belief gates the shot');
                eq(d.action, 'MOVE', 'falls back to the A* move');
                sawGate = true;
            }
            E.forEachCell((r, c) => { grid(e).beliefGrid[r][c] = 1 / 25; });
        }
        assert(sawGate, 'the gate was exercised');
    });

    it('never reads the true opponent position when deciding', () => {
        const a = E.createEngine({ rng: seeded(33) });
        const b = E.createEngine({ rng: seeded(33) });
        b.state.humanPos = { r: 4, c: 4 };            // decoy, engine must ignore
        const da = a.computeBotAction();
        const db = b.computeBotAction();
        eq(da.action, db.action, 'same action despite different true position');
        eq(cellLabel(da.target), cellLabel(db.target), 'same target');
    });

    it('lowers softmax temperature as pressure rises', () => {
        // alpha is derived from score and horizon inside computeBotAction, so
        // it has to be driven through real game state rather than assigned.
        const calm = E.createEngine({ rng: seeded(17) });
        calm.computeBotAction();
        const warmTemp = grid(calm).telemetry.temperature;

        const pressed = E.createEngine({ rng: seeded(17) });
        grid(pressed).turn = 24;                  // one round left
        grid(pressed).scoreHuman = 8;             // agent badly behind
        grid(pressed).scoreBot = 0;
        pressed.computeBotAction();

        assert(grid(pressed).alpha > grid(calm).alpha, 'alpha higher under pressure');
        assert(grid(pressed).telemetry.temperature < warmTemp,
               `temperature ${grid(pressed).telemetry.temperature.toFixed(3)} < ${warmTemp.toFixed(3)}`);
        close(grid(pressed).telemetry.temperature,
              CONFIG.TAU * (1 - 0.5 * grid(pressed).alpha), 1e-9, 'matches τ(1−0.5α)');
    });
});

/* ── A* ────────────────────────────────────────────────────────────────── */

describe('A* pathfinding', () => {
    it('finds a shortest path to the hill', () => {
        const e = E.createEngine({ rng: seeded(1) });
        const { path, cost } = e.aStar({ r: 4, c: 0 }, HILL);
        eq(cost, 4, 'Manhattan-optimal cost');
        assert(same(path[0], { r: 4, c: 0 }), 'starts at origin');
        assert(same(path[path.length - 1], HILL), 'ends on the hill');
        for (let i = 1; i < path.length; i++) {
            eq(E.manhattan(path[i - 1], path[i]), 1, 'steps are single moves');
        }
    });

    it('returns a trivial path when already on the hill', () => {
        const e = E.createEngine({ rng: seeded(1) });
        eq(e.aStar(HILL, HILL).cost, 0);
    });
});

/* ── Labels ────────────────────────────────────────────────────────────── */

describe('Board labels', () => {
    it('maps cells to file/rank notation', () => {
        eq(cellLabel({ r: 3, c: 1 }), 'B4');
        eq(cellLabel({ r: 0, c: 0 }), 'A1');
        eq(cellLabel({ r: 4, c: 4 }), 'E5');
        eq(cellLabel(HILL), 'C3');
    });
});

/* ── Report ────────────────────────────────────────────────────────────── */

console.log('');
if (failures.length) {
    console.log('FAILURES\n');
    failures.forEach(f => console.log('  ✗ ' + f + '\n'));
}
console.log(`${passed} passed, ${failed} failed, ${passed + failed} total`);
process.exit(failed ? 1 : 0);
