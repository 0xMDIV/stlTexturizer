// Standalone harness for the undo/redo state machine in js/main.js.
// Mirrors the algorithm exactly so we can exercise the invariants in Node:
//   - push only when next != baseline (dedup)
//   - LIFO undo / redo cursors
//   - any new edit clears redo stack
//   - cap at UNDO_LIMIT, drop oldest
//   - applyDepth suppresses re-capture (simulating the input/change synthetic events)
//
// To match behavior, capture & comparison use only fields the real impl uses
// (PERSISTED_KEYS subset, mask {selectionMode, excluded[]}).

const PERSISTED_KEYS = ['amplitude', 'scaleU', 'rotation', 'mappingMode'];
const UNDO_LIMIT = 50;

// Simulated app state.
const state = {
  settings: { amplitude: 0.5, scaleU: 1, rotation: 0, mappingMode: 5, activeMapName: 'Crystal' },
  excludedFaces: new Set(),
  selectionMode: false,
};

function getSettingsSnapshot() {
  const snap = {};
  for (const k of PERSISTED_KEYS) snap[k] = state.settings[k];
  snap.activeMapName = state.settings.activeMapName;
  return snap;
}
function _collectCurrentMask() {
  if (state.excludedFaces.size === 0 && !state.selectionMode) return null;
  return { selectionMode: state.selectionMode, excluded: [...state.excludedFaces] };
}
function applySettingsSnapshot(snap) {
  if (!snap) return;
  for (const k of PERSISTED_KEYS) if (snap[k] != null) state.settings[k] = snap[k];
  if (snap.activeMapName != null) state.settings.activeMapName = snap.activeMapName;
}
function _restoreMask(mask) {
  if (!mask) { state.excludedFaces = new Set(); state.selectionMode = false; return; }
  state.selectionMode = !!mask.selectionMode;
  state.excludedFaces = new Set(mask.excluded);
}

// ── Undo module (mirrors js/main.js) ─────────────────────────────────────────
let _undoStack = [];
let _redoStack = [];
let _baselineSnapshot = null;
let _undoApplyDepth = 0;

function _captureUndoSnapshot() {
  return { settings: getSettingsSnapshot(), mask: _collectCurrentMask() };
}
function _undoSnapshotsEqual(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  for (const k of PERSISTED_KEYS) if (a.settings[k] !== b.settings[k]) return false;
  if ((a.settings.activeMapName || null) !== (b.settings.activeMapName || null)) return false;
  const ma = a.mask, mb = b.mask;
  if (!ma && !mb) return true;
  if (!ma || !mb) return false;
  if (ma.selectionMode !== mb.selectionMode) return false;
  if (ma.excluded.length !== mb.excluded.length) return false;
  const sb = new Set(mb.excluded);
  for (const v of ma.excluded) if (!sb.has(v)) return false;
  return true;
}
function _commitUndoCapture() {
  if (_undoApplyDepth > 0) return;
  const next = _captureUndoSnapshot();
  if (_baselineSnapshot && _undoSnapshotsEqual(_baselineSnapshot, next)) return;
  if (_baselineSnapshot) {
    _undoStack.push(_baselineSnapshot);
    if (_undoStack.length > UNDO_LIMIT) _undoStack.shift();
  }
  _redoStack.length = 0;
  _baselineSnapshot = next;
}
function _clearUndoStacks() {
  _undoStack.length = 0;
  _redoStack.length = 0;
  _baselineSnapshot = _captureUndoSnapshot();
}
function _applyUndoSnapshot(snap) {
  _undoApplyDepth++;
  try {
    applySettingsSnapshot(snap.settings);
    _restoreMask(snap.mask);
  } finally {
    _undoApplyDepth--;
  }
}
function _undo() {
  _commitUndoCapture();
  if (!_undoStack.length) return false;
  const prev = _undoStack.pop();
  if (_baselineSnapshot) _redoStack.push(_baselineSnapshot);
  _applyUndoSnapshot(prev);
  _baselineSnapshot = prev;
  return true;
}
function _redo() {
  _commitUndoCapture();
  if (!_redoStack.length) return false;
  const next = _redoStack.pop();
  if (_baselineSnapshot) _undoStack.push(_baselineSnapshot);
  _applyUndoSnapshot(next);
  _baselineSnapshot = next;
  return true;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
let failed = 0;
function check(label, cond, extra) {
  const ok = !!cond;
  if (!ok) { failed++; console.error('FAIL:', label, extra ?? ''); }
  else console.log('  ok  ', label);
}

// ── Init ─────────────────────────────────────────────────────────────────────
_baselineSnapshot = _captureUndoSnapshot();

// ── 1) Empty stack: undo/redo are no-ops ─────────────────────────────────────
console.log('\n[1] Empty-stack no-ops');
check('undo on empty stack returns false', _undo() === false);
check('redo on empty stack returns false', _redo() === false);
check('settings unchanged', state.settings.amplitude === 0.5);

// ── 2) Single edit + undo + redo round-trip ──────────────────────────────────
console.log('\n[2] Single edit round-trip');
state.settings.amplitude = 0.8;
_commitUndoCapture();
check('undo stack has one entry', _undoStack.length === 1);
check('redo stack empty after edit', _redoStack.length === 0);
_undo();
check('after undo, amplitude reverts to 0.5', state.settings.amplitude === 0.5);
check('redo stack now has one entry', _redoStack.length === 1);
_redo();
check('after redo, amplitude restored to 0.8', state.settings.amplitude === 0.8);
check('redo stack empty again', _redoStack.length === 0);

// ── 3) Dedup: identical commit is a no-op ────────────────────────────────────
console.log('\n[3] Dedup of identical commits');
const beforeLen = _undoStack.length;
_commitUndoCapture(); // no state change
check('committing without change does not push', _undoStack.length === beforeLen);

// ── 4) New edit after undo clears redo stack ─────────────────────────────────
console.log('\n[4] New edit clears redo stack');
state.settings.amplitude = 0.3;
_commitUndoCapture();
_undo();                        // back to 0.8
check('redo stack populated', _redoStack.length === 1);
state.settings.scaleU = 2.0;
_commitUndoCapture();
check('new edit clears redo stack', _redoStack.length === 0);

// ── 5) Stack cap: oldest dropped ─────────────────────────────────────────────
console.log('\n[5] Stack cap of UNDO_LIMIT');
_clearUndoStacks();
for (let i = 0; i < UNDO_LIMIT + 10; i++) {
  state.settings.rotation = i;
  _commitUndoCapture();
}
check('stack length capped at UNDO_LIMIT', _undoStack.length === UNDO_LIMIT, `got ${_undoStack.length}`);
// Oldest entry should be rotation = 9 (we committed 60 times: state goes 0..59,
// and stack stores BEFORE-states, so first push was rotation=0 baseline. After
// 60 pushes with cap=50, oldest 10 dropped => oldest stored is rotation=9).
const oldest = _undoStack[0];
check('oldest baseline dropped', oldest.settings.rotation === 9, `oldest rotation = ${oldest.settings.rotation}`);

// ── 6) applyDepth suppresses capture ─────────────────────────────────────────
console.log('\n[6] applyDepth suppresses capture');
_clearUndoStacks();
const beforeDepth = _undoStack.length;
_undoApplyDepth++;
state.settings.amplitude = 99;
_commitUndoCapture();          // suppressed
_undoApplyDepth--;
check('no capture during applyDepth>0', _undoStack.length === beforeDepth);

// ── 7) Mask paint round-trip ─────────────────────────────────────────────────
console.log('\n[7] Mask paint round-trip');
_clearUndoStacks();
state.excludedFaces.add(42);
state.excludedFaces.add(7);
_commitUndoCapture();
state.excludedFaces.add(99);
_commitUndoCapture();
_undo();
check('after undo, face 99 removed', !state.excludedFaces.has(99));
check('faces 42 and 7 retained', state.excludedFaces.has(42) && state.excludedFaces.has(7));
_undo();
check('after second undo, mask empty', state.excludedFaces.size === 0);
_redo();
check('after redo, faces 42 and 7 restored', state.excludedFaces.has(42) && state.excludedFaces.has(7) && !state.excludedFaces.has(99));

// ── 8) clearUndoStacks rebaselines ───────────────────────────────────────────
console.log('\n[8] clearUndoStacks rebaselines');
state.settings.amplitude = 0.111;
_clearUndoStacks();
check('stacks empty after clear', _undoStack.length === 0 && _redoStack.length === 0);
state.settings.amplitude = 0.222;
_commitUndoCapture();
_undo();
check('after undo, returns to 0.111 (the new baseline)', state.settings.amplitude === 0.111);

// ── 9) Regression: _restoreMask(null) must clear the mask ────────────────────
// Bug: an earlier main.js _restoreMask did `if (!mask) return;`, leaving the
// previous stroke's faces on screen when undoing back to the empty baseline.
console.log('\n[9] _restoreMask(null) clears state');
state.excludedFaces = new Set([1, 2, 3]);
state.selectionMode = false;
_restoreMask(null);
check('null mask clears excludedFaces', state.excludedFaces.size === 0);
check('null mask leaves selectionMode false', state.selectionMode === false);

state.excludedFaces = new Set([4, 5]);
state.selectionMode = true;
_restoreMask(null);
check('null mask resets include→exclude mode', state.selectionMode === false);
check('null mask clears excludedFaces (from include mode)', state.excludedFaces.size === 0);

// ── 10) Regression: many brush strokes, undo all the way back ───────────────
// Mirrors the user's report: paint several strokes, then undo each one. Each
// undo must visibly change the mask, and the final undo must end at empty.
console.log('\n[10] Multi-stroke paint: undo back to empty');
_clearUndoStacks();
state.excludedFaces = new Set();
state.selectionMode = false;
_baselineSnapshot = _captureUndoSnapshot();

const strokes = [[10,11,12], [20,21,22], [30,31,32], [40,41,42], [50,51,52]];
for (const s of strokes) { for (const f of s) state.excludedFaces.add(f); _commitUndoCapture(); }
check('5 strokes → 5 undo entries', _undoStack.length === 5, `got ${_undoStack.length}`);

for (let i = 0; i < 5; i++) _undo();
check('after 5 undos, mask is empty', state.excludedFaces.size === 0,
  `still have ${state.excludedFaces.size} faces`);
check('redo stack has 5 entries', _redoStack.length === 5);

for (let i = 0; i < 5; i++) _redo();
check('after 5 redos, all 15 faces present', state.excludedFaces.size === 15,
  `got ${state.excludedFaces.size}`);
check('redo stack empty after fully redoing', _redoStack.length === 0);

// ── Result ───────────────────────────────────────────────────────────────────
console.log(`\n${failed === 0 ? 'PASS' : 'FAIL'}: ${failed} failure(s)`);
process.exit(failed === 0 ? 0 : 1);
