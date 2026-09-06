/* ==========================================================================
   Presentation layer.

   All game rules and every number shown here come from engine.js. This file
   only renders state and animates the resolution plan the engine returns — it
   never decides an outcome or invents a value.
   ========================================================================== */

'use strict';

const {
    CONFIG, ACTIONS, HILL, RESPAWN_CELLS,
    same, manhattan, inBounds, forEachCell, cellLabel, isRespawnCell,
    createEngine,
} = SHCBO;

const engine = createEngine();
const S = engine.state;

/* ── UI state ──────────────────────────────────────────────────────────── */
let selectedAction = 'MOVE';
let selectedTarget = null;
let busy = false;           // locks input while a round animates
let started = false;
let overlayOn = false;
let lastShotMarks = { human: null, bot: null };
let beliefSamples = [];
let regimeSwitches = 0;
let botActionCounts = { MOVE: 0, SHOOT: 0, HOLD: 0 };

const sleep = ms => new Promise(r => setTimeout(r, ms));
const SVG_NS = 'http://www.w3.org/2000/svg';
const $ = id => document.getElementById(id);

/* ==========================================================================
   Lifecycle
   ========================================================================== */

function startGame() {
    started = true;
    $('start-screen').classList.remove('active');
    newMatch();
}

function resetGame() {
    $('over-modal').classList.remove('active');
    newMatch();
}

function newMatch() {
    engine.reset();
    selectedAction = 'MOVE';
    selectedTarget = null;
    busy = false;
    lastShotMarks = { human: null, bot: null };
    beliefSamples = [];
    regimeSwitches = 0;
    botActionCounts = { MOVE: 0, SHOOT: 0, HOLD: 0 };

    $('feed').innerHTML = '';
    $('tracers').innerHTML = '';
    hideReveal();
    pushEvent(`Match ready — ${CONFIG.T} rounds`, 'round');

    setAction('MOVE');
    renderAll();
    positionTokens(false);
    setPipelineIdle();
}

/* ==========================================================================
   Turn submission and animated resolution
   ========================================================================== */

async function submitTurn() {
    if (!started || busy || S.gameOver) return;
    if (selectedAction !== 'HOLD' && !selectedTarget) return;

    // Ask the engine first — it owns validation, including the
    // consecutive-shot restriction.
    const check = engine.validateAction(selectedAction, selectedTarget);
    if (!check.ok) {
        rejectFeedback(check.reason);
        return;
    }

    busy = true;
    $('btn-confirm').disabled = true;
    $('phase-tag').textContent = 'Resolving';

    const plan = engine.resolveTurn(selectedAction, selectedTarget);
    if (!plan || plan.rejected) {
        // Defensive: engine refused after all. Surface it rather than hang.
        busy = false;
        rejectFeedback(plan ? plan.reason : 'Move refused.');
        renderAll();
        return;
    }

    botActionCounts[plan.botAct]++;
    if (plan.telemetry.regime.switched) regimeSwitches++;
    plan.events.forEach(e => pushEvent(e.text, e.kind));

    // Inspector reflects the decision the engine just made.
    renderTelemetry(plan);
    await runPipeline(plan);

    // Phase 1 — reveal both commitments.
    showReveal(
        `${plan.humanAct}${plan.humanAct === 'HOLD' ? '' : ' → ' + cellLabel(plan.humanTarget)}`,
        `${plan.botAct}${plan.botAct === 'HOLD' ? '' : ' → ' + cellLabel(plan.botTarget)}`
    );
    await sleep(520);

    // Phase 2 — movement.
    renderBoard(plan.afterMove);
    positionTokens(true, plan.afterMove);
    if (plan.humanAct === 'MOVE' || plan.botAct === 'MOVE') await sleep(430);

    // Phase 3 — shots.
    if (plan.shots.length) {
        for (const shot of plan.shots) fireTracer(shot);
        await sleep(330);
        for (const shot of plan.shots) {
            const victim = shot.side === 'human' ? 'tok-bot' : 'tok-human';
            if (shot.outcome === 'HIT') flashToken(victim, 'hit');
            else if (shot.outcome === 'SHIELDED') flashToken(victim, 'shield');
            if (shot.side === 'human') lastShotMarks.human = shot.target;
            else lastShotMarks.bot = shot.target;
        }
        await sleep(240);
        renderBoard(plan.afterShots);
        positionTokens(true, plan.afterShots);
        if (plan.humanKnockedTo || plan.botKnockedTo) await sleep(400);
    }

    // Phase 4 — hill resolution.
    if (plan.scored) {
        const side = plan.scored === 'HUMAN' ? 'p' : 'a';
        flashHill(side);
        floatText(HILL, 'HILL CAPTURED', side, 0);
        floatText(HILL, '+1 POINT', side, 260);
        bumpScore(plan.scored);
        renderScores();
        await sleep(700);

        // Respawn into one of the four cells around the hill.
        highlightRespawnCells(true);
        $('phase-tag').textContent = 'Respawning';
        await sleep(420);
        renderBoard(plan.final);
        positionTokens(true, plan.final);
        markRespawnLanding(plan.respawn.to);
        await sleep(420);
        highlightRespawnCells(false);
    }

    // Round complete.
    selectedTarget = null;
    hideReveal();
    if (!S.gameOver) setAction('MOVE');

    // Evaluation only — measures the engine, never feeds it.
    beliefSamples.push(S.decisionBelief[S.humanPos.r][S.humanPos.c]);

    renderAll();
    busy = false;
    renderPrompt();

    if (S.gameOver) {
        $('phase-tag').textContent = 'Complete';
        await sleep(500);
        showGameOver();
    } else {
        $('phase-tag').textContent = 'Your move';
    }
}

function rejectFeedback(reason) {
    toast(reason, true);
    pushEvent(reason, 'bot');
    const p = $('prompt');
    p.classList.add('warn');
    p.textContent = reason;
    setTimeout(() => { p.classList.remove('warn'); renderPrompt(); }, 2200);
}

/* ==========================================================================
   SHCBO pipeline — Observe → Update belief → Predict → Act
   Every string below is derived from engine telemetry.
   ========================================================================== */

function setPipelineIdle() {
    ['observe', 'belief', 'predict', 'act'].forEach(k => {
        const el = document.querySelector(`.stage[data-stage="${k}"]`);
        el.classList.remove('lit', 'done');
    });
    $('pipe-observe').textContent = 'Awaiting first round';
    $('pipe-belief').textContent = '—';
    $('pipe-predict').textContent = '—';
    $('pipe-act').textContent = '—';
    $('pipe-tag').textContent = 'idle';
}

function pipelineText(plan) {
    const t = plan.telemetry;
    const obs = t.observation;

    const observe = obs
        ? `Action <em>${obs.humanAction}</em> · hill <em>${obs.hill}</em>` +
          (obs.shotOutcome ? ` · own shot <em>${obs.shotOutcome}</em>` : '') +
          (obs.humanRespawned ? ' · <em>opponent respawned</em>' : '')
        : 'Round 1 — no prior observation, belief is the spawn prior';

    const belief = t.belief
        ? `Entropy <em>${t.belief.entropyBefore.toFixed(2)}</em> → <em>${t.belief.entropyAfter.toFixed(2)}</em> bits ` +
          `(${t.belief.delta <= 0 ? 'sharpened' : 'diffused'} ${Math.abs(t.belief.delta).toFixed(2)})`
        : `Prior entropy <em>${t.entropy.toFixed(2)}</em> bits`;

    const predict =
        `Most likely block <em>${cellLabel(t.argmax.cell)}</em> at <em>${(t.argmax.prob * 100).toFixed(0)}%</em> · ` +
        `regime <em>${SHCBO.REGIME_META[t.regime.regime].label}</em>`;

    const act = t.gated
        ? `Sampled <em>SHOOT</em>, gated (peak ${t.argmax.prob.toFixed(2)} &lt; θ ${CONFIG.THETA_SHOOT}) → ` +
          `<em>${t.action} ${cellLabel(t.target)}</em>`
        : `Chose <em>${t.action}${t.action === 'HOLD' ? '' : ' ' + cellLabel(t.target)}</em> ` +
          `at π = <em>${(t.policy[t.sampled] * 100).toFixed(0)}%</em>`;

    return { observe, belief, predict, act };
}

async function runPipeline(plan) {
    const text = pipelineText(plan);
    const order = [
        ['observe', text.observe],
        ['belief', text.belief],
        ['predict', text.predict],
        ['act', text.act],
    ];
    $('pipe-tag').textContent = 'running';
    document.querySelectorAll('.stage').forEach(s => s.classList.remove('lit', 'done'));

    for (const [key, html] of order) {
        const stage = document.querySelector(`.stage[data-stage="${key}"]`);
        stage.classList.add('lit');
        $(`pipe-${key}`).innerHTML = html;
        await sleep(150);
        stage.classList.remove('lit');
        stage.classList.add('done');
    }
    $('pipe-tag').textContent = `round ${plan.round}`;
}

/* ==========================================================================
   Input
   ========================================================================== */

function setAction(type) {
    if (busy || S.gameOver) return;
    selectedAction = type;
    selectedTarget = type === 'HOLD' ? { r: S.humanPos.r, c: S.humanPos.c } : null;
    document.querySelectorAll('.act').forEach(b =>
        b.setAttribute('aria-pressed', String(b.dataset.action === type)));
    renderBoard();
    renderPrompt();
}

function cellClicked(r, c) {
    if (!started || busy || S.gameOver || selectedAction === 'HOLD') return;
    const cell = { r, c };

    if (selectedAction === 'MOVE' && manhattan(cell, S.humanPos) > 1) {
        toast('Move is one orthogonal step — pick an adjacent block.', true);
        return;
    }
    if (selectedAction === 'SHOOT') {
        const check = engine.validateShot('human', cell);
        if (!check.ok) { rejectFeedback(check.reason); return; }
    }
    selectedTarget = cell;
    renderBoard();
    renderPrompt();
}

function nudge(dr, dc) {
    if (busy || S.gameOver || selectedAction === 'HOLD') return;
    const base = selectedAction === 'MOVE' ? S.humanPos : (selectedTarget || S.humanPos);
    const t = { r: base.r + dr, c: base.c + dc };
    if (!inBounds(t.r, t.c)) return;
    if (selectedAction === 'MOVE' && manhattan(t, S.humanPos) > 1) return;
    selectedTarget = t;
    renderBoard();
    renderPrompt();
}

function setupKeys() {
    window.addEventListener('keydown', e => {
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        const k = e.key.toLowerCase();

        if ($('rules-modal').classList.contains('active')) {
            if (e.key === 'Escape' || e.key === 'Enter') { e.preventDefault(); toggleRules(); }
            return;
        }
        if ($('over-modal').classList.contains('active')) {
            if (e.key === 'Enter' || k === 'r') { e.preventDefault(); resetGame(); }
            return;
        }
        if ($('start-screen').classList.contains('active')) {
            if (e.key === 'Enter') { e.preventDefault(); startGame(); }
            else if (e.key === '?') { e.preventDefault(); toggleRules(); }
            return;
        }

        const arrows = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
        if (arrows[e.key]) { e.preventDefault(); nudge(arrows[e.key][0], arrows[e.key][1]); return; }

        if (e.key === '1') { e.preventDefault(); setAction('MOVE'); }
        else if (e.key === '2') { e.preventDefault(); setAction('SHOOT'); }
        else if (e.key === '3') { e.preventDefault(); setAction('HOLD'); }
        else if (e.key === 'Enter') { e.preventDefault(); submitTurn(); }
        else if (e.key === 'Escape') { e.preventDefault(); selectedTarget = null; renderBoard(); renderPrompt(); }
        else if (k === 'b') { e.preventDefault(); toggleBeliefOverlay(); }
        else if (k === 'r') { e.preventDefault(); resetGame(); }
        else if (e.key === '?') { e.preventDefault(); toggleRules(); }
    });

    window.addEventListener('resize', () => positionTokens(false));
}

/* ==========================================================================
   Board rendering
   ========================================================================== */

function renderCoordinateRails() {
    $('files').innerHTML = Array.from({ length: CONFIG.GRID },
        (_, c) => `<span>${String.fromCharCode(65 + c)}</span>`).join('');
    $('ranks').innerHTML = Array.from({ length: CONFIG.GRID },
        (_, r) => `<span>${r + 1}</span>`).join('');
}

function renderBoard(posOverride) {
    const pos = posOverride || { human: S.humanPos, bot: S.botPos };
    const board = $('board');
    board.innerHTML = '';

    const path = S.telemetry ? S.telemetry.search.path : [];
    const peak = engine.beliefArgmax(null);
    const bannedHuman = S.lastShot.human;
    const hillHeldBy = same(pos.human, HILL) && same(pos.bot, HILL) ? 'both'
                     : same(pos.human, HILL) ? 'p'
                     : same(pos.bot, HILL) ? 'a' : null;

    forEachCell((r, c) => {
        const cell = { r, c };
        const el = document.createElement('button');
        el.type = 'button';
        el.className = 'cell';
        el.dataset.cell = cellLabel(cell);

        if (same(cell, HILL)) {
            el.classList.add('hill');
            if (hillHeldBy === 'p') el.classList.add('held-p');
            if (hillHeldBy === 'a') el.classList.add('held-a');
            const crown = document.createElement('span');
            crown.className = 'hill-crown';
            crown.textContent = '♛';
            el.appendChild(crown);
        }
        if (isRespawnCell(cell)) el.classList.add('respawn');
        if (path.some(p => same(p, cell))) el.classList.add('path');

        const live = started && !busy && !S.gameOver;
        if (live && selectedAction === 'MOVE' && manhattan(cell, S.humanPos) <= 1) el.classList.add('legal');
        if (live && selectedAction === 'SHOOT') {
            if (bannedHuman && same(bannedHuman, cell)) {
                el.classList.add('blocked');
                const nr = document.createElement('span');
                nr.className = 'no-repeat';
                nr.textContent = 'NO';
                el.appendChild(nr);
            } else {
                el.classList.add('aimable');
            }
        }
        if (selectedTarget && same(selectedTarget, cell)) {
            el.classList.add(selectedAction === 'SHOOT' ? 'sel-shoot' : 'sel-move');
        }

        // Belief overlay (real predictive distribution).
        const p = S.decisionBelief[r][c];
        const wash = document.createElement('span');
        wash.className = 'b-wash';
        el.style.setProperty('--bt', Math.pow(Math.min(1, p / 0.6), 0.6).toFixed(3));
        el.appendChild(wash);

        const num = document.createElement('span');
        num.className = 'b-num';
        num.textContent = p >= 0.005 ? `${Math.round(p * 100)}%` : '';
        el.appendChild(num);
        if (same(cell, peak.cell) && peak.prob > 0.02) el.classList.add('b-peak');

        // Previous shot markers.
        if (lastShotMarks.human && same(lastShotMarks.human, cell)) el.appendChild(shotMark('by-p'));
        if (lastShotMarks.bot && same(lastShotMarks.bot, cell)) el.appendChild(shotMark('by-a'));

        const id = document.createElement('span');
        id.className = 'cell-id';
        id.textContent = cellLabel(cell);
        el.appendChild(id);

        let label = `Block ${cellLabel(cell)}`;
        if (same(cell, HILL)) label += ', the hill';
        if (same(pos.human, cell)) label += ', your position';
        if (same(pos.bot, cell)) label += ', SHCBO position';
        el.setAttribute('aria-label', label);

        el.onclick = () => cellClicked(r, c);
        board.appendChild(el);
    });
}

function shotMark(cls) {
    const m = document.createElement('span');
    m.className = `shot-mark ${cls}`;
    m.textContent = '✕';
    return m;
}

function cellEl(cell) {
    return $('board').children[cell.r * CONFIG.GRID + cell.c];
}

function cellCenter(cell) {
    const wrap = document.querySelector('.board-wrap');
    const el = cellEl(cell);
    if (!el || !wrap) return { x: 0, y: 0 };
    const w = wrap.getBoundingClientRect(), k = el.getBoundingClientRect();
    return { x: k.left - w.left + k.width / 2, y: k.top - w.top + k.height / 2 };
}

function positionTokens(animate, posOverride) {
    const pos = posOverride || { human: S.humanPos, bot: S.botPos };
    const stacked = same(pos.human, pos.bot);
    place($('tok-human'), pos.human, animate, stacked ? -12 : 0);
    place($('tok-bot'), pos.bot, animate, stacked ? 12 : 0);
}

function place(el, cell, animate, dx) {
    if (!el) return;
    const p = cellCenter(cell);
    if (!animate) {
        el.style.transition = 'none';
        el.style.left = `${p.x + dx}px`;
        el.style.top = `${p.y}px`;
        void el.offsetWidth;
        el.style.transition = '';
    } else {
        el.style.left = `${p.x + dx}px`;
        el.style.top = `${p.y}px`;
    }
}

/* ── Effects ───────────────────────────────────────────────────────────── */

function fireTracer(shot) {
    const svg = $('tracers');
    const wrap = document.querySelector('.board-wrap');
    const w = wrap.clientWidth, h = wrap.clientHeight;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

    const p1 = cellCenter(shot.from);
    const p2 = cellCenter(shot.target);
    const colour = shot.side === 'human' ? '#56A2FF' : '#FF7482';

    const line = document.createElementNS(SVG_NS, 'line');
    line.setAttribute('x1', p1.x); line.setAttribute('y1', p1.y);
    line.setAttribute('x2', p2.x); line.setAttribute('y2', p2.y);
    line.setAttribute('stroke', colour);
    line.setAttribute('stroke-width', '3');
    line.setAttribute('stroke-linecap', 'round');
    const len = Math.max(1, Math.hypot(p2.x - p1.x, p2.y - p1.y));
    line.style.strokeDasharray = len;
    line.style.strokeDashoffset = len;
    line.style.filter = `drop-shadow(0 0 6px ${colour})`;
    svg.appendChild(line);

    const ring = document.createElementNS(SVG_NS, 'circle');
    ring.setAttribute('cx', p2.x); ring.setAttribute('cy', p2.y);
    ring.setAttribute('r', '6');
    ring.setAttribute('fill', 'none');
    ring.setAttribute('stroke', shot.outcome === 'MISS' ? '#9AA9BF' : colour);
    ring.setAttribute('stroke-width', '2.5');
    ring.style.opacity = '0';
    svg.appendChild(ring);

    requestAnimationFrame(() => {
        line.style.transition = 'stroke-dashoffset .2s linear, opacity .32s .2s';
        line.style.strokeDashoffset = '0';
        line.style.opacity = '0';
        ring.style.transition = 'r .36s ease-out .16s, opacity .36s ease-out .16s';
        ring.style.opacity = '1';
        setTimeout(() => { ring.setAttribute('r', '24'); ring.style.opacity = '0'; }, 180);
    });
    setTimeout(() => { line.remove(); ring.remove(); }, 900);
}

function flashToken(id, kind) {
    const el = $(id);
    if (!el) return;
    let fx = el.querySelector('.fx');
    if (!fx) { fx = document.createElement('span'); fx.className = 'fx'; el.appendChild(fx); }
    el.classList.remove('hit', 'shield');
    void el.offsetWidth;
    el.classList.add(kind);
    setTimeout(() => el.classList.remove(kind), 800);
}

function flashHill(side) {
    const el = cellEl(HILL);
    if (!el) return;
    el.classList.add(side === 'p' ? 'capture-p' : 'capture-a');
    setTimeout(() => el.classList.remove('capture-p', 'capture-a'), 850);
}

function highlightRespawnCells(on) {
    RESPAWN_CELLS.forEach(cell => {
        const el = cellEl(cell);
        if (el) el.classList.toggle('respawn-live', on);
    });
}

function markRespawnLanding(cell) {
    const el = cellEl(cell);
    if (!el) return;
    el.classList.add('respawn-live');
    setTimeout(() => el.classList.remove('respawn-live'), 900);
}

function floatText(cell, text, side, delay) {
    setTimeout(() => {
        const layer = $('fx-layer');
        const p = cellCenter(cell);
        const el = document.createElement('div');
        el.className = 'float';
        el.textContent = text;
        el.style.left = `${p.x}px`;
        el.style.top = `${p.y}px`;
        el.style.fontSize = text.length > 10 ? '.78rem' : '1.05rem';
        el.style.color = side === 'p' ? '#8CC1FF' : '#FFA0AA';
        layer.appendChild(el);
        setTimeout(() => el.remove(), 1300);
    }, delay || 0);
}

function bumpScore(side) {
    const el = $(side === 'HUMAN' ? 'score-human' : 'score-bot');
    el.classList.remove('pop');
    void el.offsetWidth;
    el.classList.add('pop');
}

function showReveal(a, b) {
    $('rev-human').textContent = a;
    $('rev-bot').textContent = b;
    $('reveal').classList.add('show');
}
function hideReveal() { $('reveal').classList.remove('show'); }

/* ==========================================================================
   Panels
   ========================================================================== */

function renderAll() {
    renderBoard();
    renderPrompt();
    renderScores();
    renderStatus();
    renderBelief();
    renderDecision();
    renderOpponentModel();
    renderPressure();
    renderPath();
    positionTokens(true);
}

function renderTelemetry() {
    renderBelief();
    renderDecision();
    renderOpponentModel();
    renderPressure();
    renderPath();
}

function renderScores() {
    $('score-human').textContent = S.scoreHuman;
    $('score-bot').textContent = S.scoreBot;
    $('st-player').textContent = S.scoreHuman;
    $('st-bot').textContent = S.scoreBot;
}

function renderStatus() {
    const round = Math.min(S.turn, CONFIG.T);
    $('round-now').textContent = round;
    $('st-round').textContent = `${round} / ${CONFIG.T}`;
    $('round-fill').style.width = `${((round - (S.gameOver ? 0 : 1)) / CONFIG.T) * 100}%`;
    const track = document.querySelector('.round-track');
    if (track) track.setAttribute('aria-valuenow', String(round));

    const hillEl = $('st-hill');
    if (same(S.humanPos, HILL) && same(S.botPos, HILL)) { hillEl.textContent = 'Contested'; hillEl.className = 'st-v'; }
    else if (same(S.humanPos, HILL)) { hillEl.textContent = 'Player'; hillEl.className = 'st-v accent-p'; }
    else if (same(S.botPos, HILL)) { hillEl.textContent = 'SHCBO'; hillEl.className = 'st-v accent-a'; }
    else { hillEl.textContent = 'Uncontrolled'; hillEl.className = 'st-v'; }

    const last = S.history[S.history.length - 1];
    $('st-last').textContent = last
        ? `You ${last.humanAction} · AI ${last.botAction}`
        : '—';

    if (!started) $('phase-tag').textContent = 'Ready';
    else if (S.gameOver) $('phase-tag').textContent = 'Complete';
    else if (!busy) $('phase-tag').textContent = 'Your move';
}

function renderPrompt() {
    const el = $('prompt');
    const btn = $('btn-confirm');

    if (S.gameOver) {
        el.innerHTML = 'Match complete — press <kbd>R</kbd> to play again.';
        btn.disabled = true; return;
    }
    if (busy) { el.textContent = 'Resolving round…'; btn.disabled = true; return; }

    if (selectedAction === 'HOLD') {
        el.innerHTML = `Holding <b>${cellLabel(S.humanPos)}</b> — blocks one incoming shot. <kbd>Enter</kbd> to confirm.`;
    } else if (!selectedTarget) {
        const banned = S.lastShot.human;
        el.innerHTML = selectedAction === 'MOVE'
            ? 'Pick an adjacent block — click, or <span class="keys"><kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd></span>'
            : `Pick a block to fire at${banned ? ` — <b>${cellLabel(banned)}</b> is blocked this round` : ''}.`;
    } else {
        el.innerHTML = `<b>${selectedAction} → ${cellLabel(selectedTarget)}</b> · <kbd>Enter</kbd> confirm · <kbd>Esc</kbd> clear`;
    }
    btn.disabled = !selectedTarget;
}

function renderBelief() {
    const grid = $('belief');
    grid.innerHTML = '';
    const peak = engine.beliefArgmax(null);

    forEachCell((r, c) => {
        const p = S.decisionBelief[r][c];
        const d = document.createElement('div');
        d.className = 'bcell';
        const t = Math.pow(Math.min(1, p / 0.6), 0.6);
        d.style.setProperty('--t', t.toFixed(3));
        if (t > 0.5) d.classList.add('lit');
        if (same({ r, c }, HILL)) d.classList.add('hillcell');
        if (same({ r, c }, peak.cell) && peak.prob > 0) d.classList.add('peak');
        d.textContent = p >= 0.005 ? Math.round(p * 100) : '·';
        d.title = `${cellLabel({ r, c })}: ${(p * 100).toFixed(1)}%`;
        grid.appendChild(d);
    });

    $('belief-peak').textContent = `${cellLabel(peak.cell)} · ${(peak.prob * 100).toFixed(0)}%`;
    $('belief-entropy').textContent = `${engine.beliefEntropy().toFixed(2)} bits`;

    const open = peak.prob >= CONFIG.THETA_SHOOT;
    const tag = $('gate-tag');
    tag.textContent = open ? 'shoot armed' : 'shoot gated';
    tag.className = 'tag' + (open ? ' tag-live' : '');
}

function renderDecision() {
    const body = $('decision-rows');
    const t = S.telemetry;
    if (!t) {
        body.innerHTML = '<div class="empty">Confirm a turn to populate</div>';
        $('temp-tag').textContent = '—';
        return;
    }
    $('temp-tag').textContent = `τ ${t.temperature.toFixed(3)}`;

    body.innerHTML = ACTIONS.map(a => `
        <div class="drow${a === t.sampled ? ' pick' : ''}">
            <span class="da">${a}</span>
            <span class="dn">${t.U[a].toFixed(2)}</span>
            <span class="dn">${t.C[a].toFixed(2)}</span>
            <span class="dnet">${t.net[a].toFixed(2)}</span>
            <span class="dpi">${(t.policy[a] * 100).toFixed(0)}%</span>
        </div>`).join('')
        + (t.gated
            ? `<div class="gate-note">SHOOT sampled, then suppressed: peak belief ${t.argmax.prob.toFixed(2)} is below θ ${CONFIG.THETA_SHOOT}. Fell back to the A* move.</div>`
            : '')
        + (t.shotBanned
            ? `<div class="gate-note">Agent may not re-target ${cellLabel(t.shotBanned)} this round; its aim was taken over the remaining blocks.</div>`
            : '');
}

function bars(dist, containerId) {
    $(containerId).innerHTML = ACTIONS.map(a => {
        const v = dist[a] || 0;
        return `<div class="bar-row">
            <span>${a}</span>
            <span class="bar-track"><span class="bar-fill f-${a.toLowerCase()}" style="width:${(v * 100).toFixed(0)}%"></span></span>
            <b>${(v * 100).toFixed(0)}%</b>
        </div>`;
    }).join('');
}

function renderOpponentModel() {
    const { freq } = engine.actionFrequencies();
    const meta = SHCBO.REGIME_META[S.regime];
    const tag = $('regime-tag');
    tag.textContent = meta.label;
    tag.className = 'tag' + (S.regime === 'AGGRESSIVE' ? ' hot' : S.regime === 'PREDICTIVE' ? ' warm' : '');

    bars(freq, 'freq-rows');
    bars(SHCBO.REGIME_PRIOR[S.regime], 'prior-rows');

    $('regime-reason').textContent = S.telemetry
        ? S.telemetry.regime.reason
        : 'Window fills from your first two actions.';
}

function renderPressure() {
    $('alpha-fill').style.width = `${(S.alpha * 100).toFixed(1)}%`;
    $('alpha-val').textContent = S.alpha.toFixed(3);
    const tag = $('alpha-tag');
    if (S.alpha > 0.6) { tag.textContent = 'escalated'; tag.className = 'tag hot'; }
    else if (S.alpha > 0.3) { tag.textContent = 'balanced'; tag.className = 'tag warm'; }
    else { tag.textContent = 'patient'; tag.className = 'tag'; }
    renderAlphaChart($('alpha-chart'), 280, 70);
}

function renderAlphaChart(svg, W, H) {
    if (!svg) return;
    const L = 20, R = 6, T = 7, B = 14;
    const x = t => L + ((t - 1) / (CONFIG.T - 1)) * (W - L - R);
    const y = v => T + (1 - v) * (H - T - B);

    let h = '';
    for (const v of [0, 0.5, 1]) {
        h += `<line class="gl" x1="${L}" y1="${y(v).toFixed(1)}" x2="${W - R}" y2="${y(v).toFixed(1)}"/>`;
        h += `<text class="lbl" x="1" y="${(y(v) + 2.5).toFixed(1)}">${v.toFixed(1)}</text>`;
    }
    h += `<line class="ax" x1="${L}" y1="${y(0)}" x2="${W - R}" y2="${y(0)}"/>`;
    h += `<text class="lbl" x="${L}" y="${H - 2}">1</text>`;
    h += `<text class="lbl" x="${W - R - 8}" y="${H - 2}">${CONFIG.T}</text>`;

    const pts = S.history.map(d => [x(d.round), y(d.alpha)]);
    if (pts.length) {
        const line = pts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
        h += `<polygon class="area" points="${pts[0][0].toFixed(1)},${y(0).toFixed(1)} ${line} ${pts[pts.length - 1][0].toFixed(1)},${y(0).toFixed(1)}"/>`;
        h += `<polyline class="line" points="${line}"/>`;
        if (pts.length <= 30) for (const p of pts) h += `<circle class="dot" cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="1.9"/>`;
    } else {
        h += `<text class="lbl" x="${W / 2 - 26}" y="${H / 2}">no rounds yet</text>`;
    }
    svg.innerHTML = h;
}

function renderPath() {
    const t = S.telemetry;
    if (!t) { $('path-line').textContent = '—'; $('path-cost').textContent = '—'; return; }
    $('path-line').textContent = t.search.path.map(cellLabel).join('  →  ');
    $('path-cost').textContent = `${t.search.cost} steps · ${t.search.expansions} expansions`;
}

/* ==========================================================================
   Chrome
   ========================================================================== */

function pushEvent(text, kind) {
    const feed = $('feed');
    const el = document.createElement('div');
    el.className = `ev ${kind || 'system'}`;
    el.textContent = text;
    feed.appendChild(el);
    while (feed.children.length > 120) feed.removeChild(feed.firstChild);
    feed.scrollTop = feed.scrollHeight;
}

let toastTimer;
function toast(msg, bad) {
    const el = $('toast');
    el.textContent = msg;
    el.className = 'toast show' + (bad ? ' bad' : '');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = 'toast'; }, 2600);
}

function toggleRules() { $('rules-modal').classList.toggle('active'); }

function toggleBeliefOverlay() {
    overlayOn = !overlayOn;
    document.body.classList.toggle('overlay', overlayOn);
    $('overlay-toggle').setAttribute('aria-pressed', String(overlayOn));
}

function showGameOver() {
    const winner = S.winner;
    $('over-title').textContent =
        winner === 'HUMAN' ? 'Player wins' : winner === 'BOT' ? 'SHCBO wins' : 'Draw';
    $('over-human').textContent = S.scoreHuman;
    $('over-bot').textContent = S.scoreBot;

    renderAlphaChart($('over-chart'), 300, 84);

    const total = ACTIONS.reduce((s, a) => s + botActionCounts[a], 0) || 1;
    $('over-mix').innerHTML = ACTIONS.map(a => {
        const pct = 100 * botActionCounts[a] / total;
        return `<div class="bar-row">
            <span>${a}</span>
            <span class="bar-track"><span class="bar-fill f-${a.toLowerCase()}" style="width:${pct.toFixed(0)}%"></span></span>
            <b>${pct.toFixed(0)}%</b>
        </div>`;
    }).join('');

    $('over-timeline').innerHTML = S.history.map(h => {
        const cls = h.hill === 'HUMAN' ? 't-p' : h.hill === 'BOT' ? 't-a' : h.hill === 'CONTESTED' ? 't-c' : '';
        return `<div class="tl ${cls}" title="Round ${h.round}: ${h.hill.toLowerCase()}"></div>`;
    }).join('');

    const mean = beliefSamples.length
        ? beliefSamples.reduce((a, b) => a + b, 0) / beliefSamples.length : 0;
    $('over-belief').textContent = mean.toFixed(3);
    $('over-switches').textContent = regimeSwitches;

    $('over-modal').classList.add('active');
}

/* ── Boot ──────────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
    renderCoordinateRails();
    setupKeys();
    newMatch();
    $('start-screen').classList.add('active');
    setTimeout(() => positionTokens(false), 60);
});
