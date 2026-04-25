// Standalone harness for the BFS circle brush in js/main.js → bfsBrushSelect.
// Mirrors the algorithm exactly so we can exercise it in Node without three.js
// or a browser. After PrusaSlicer's TriangleSelector::select_patch.
//
// Verifies the bug the user reported is fixed: when the brush circle is wider
// than a thin shell's wall thickness, triangles on the BACK side (whether
// opposite-normal or same-normal hidden layer) must NOT be selected. With BFS
// that comes for free — the walk can't traverse a back-facing wall to reach
// the other side, and disconnected components are never visited at all.

// ── Tiny linear-algebra primitives ────────────────────────────────────────────
const sub  = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot  = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const norm = (a) => { const m = Math.hypot(a[0],a[1],a[2]); return [a[0]/m, a[1]/m, a[2]/m]; };
const dist = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]);

// Closest-point-on-triangle squared distance (same as main.js distSqPointToTri)
function distSqPointToTri(p, a, b, c) {
  const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return dot(ap, ap);
  const bp = sub(p, b);
  const d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return dot(bp, bp);
  const cp = sub(p, c);
  const d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return dot(cp, cp);
  const vc = d1*d4 - d3*d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    const q = [a[0]+v*ab[0], a[1]+v*ab[1], a[2]+v*ab[2]];
    const d = sub(q, p); return dot(d, d);
  }
  const vb = d5*d2 - d1*d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const q = [a[0]+w*ac[0], a[1]+w*ac[1], a[2]+w*ac[2]];
    const d = sub(q, p); return dot(d, d);
  }
  const va = d3*d6 - d5*d4;
  if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
    const w = (d4-d3) / ((d4-d3) + (d5-d6));
    const q = [b[0]+w*(c[0]-b[0]), b[1]+w*(c[1]-b[1]), b[2]+w*(c[2]-b[2])];
    const d = sub(q, p); return dot(d, d);
  }
  const den = 1 / (va + vb + vc);
  const v = vb * den, w = vc * den;
  const q = [a[0]+ab[0]*v+ac[0]*w, a[1]+ab[1]*v+ac[1]*w, a[2]+ab[2]*v+ac[2]*w];
  const d = sub(q, p); return dot(d, d);
}

// ── Mesh helpers ─────────────────────────────────────────────────────────────
// Each tri: { v: [a,b,c] }. Adjacency built by sharing two vertex positions
// (after rounding). Face normal from CCW vertex order.

const QUANT = 1e4;
const key = ([x,y,z]) => `${Math.round(x*QUANT)}|${Math.round(y*QUANT)}|${Math.round(z*QUANT)}`;

function annotate(tris) {
  for (const t of tris) {
    const [a, b, c] = t.v;
    const e1 = sub(b, a), e2 = sub(c, a);
    t.normal = norm([
      e1[1]*e2[2] - e1[2]*e2[1],
      e1[2]*e2[0] - e1[0]*e2[2],
      e1[0]*e2[1] - e1[1]*e2[0],
    ]);
    t.centroid = [(a[0]+b[0]+c[0])/3, (a[1]+b[1]+c[1])/3, (a[2]+b[2]+c[2])/3];
  }
}

function buildAdjacency(tris) {
  // Build per-edge → triangles list, then per-triangle neighbors.
  const edgeMap = new Map();
  for (let i = 0; i < tris.length; i++) {
    const [a, b, c] = tris[i].v;
    const ka = key(a), kb = key(b), kc = key(c);
    const edges = [[ka, kb], [kb, kc], [kc, ka]];
    for (const [u, v] of edges) {
      const ek = u < v ? `${u}__${v}` : `${v}__${u}`;
      let arr = edgeMap.get(ek);
      if (!arr) { arr = []; edgeMap.set(ek, arr); }
      arr.push(i);
    }
  }
  const adj = tris.map(() => []);
  for (const [, list] of edgeMap) {
    if (list.length === 2) {
      adj[list[0]].push(list[1]);
      adj[list[1]].push(list[0]);
    }
  }
  return adj;
}

// ── BFS circle brush (mirrors bfsBrushSelect in main.js) ─────────────────────
function bfsBrushSelect({ tris, adjacency, seed, hitPt, brushR, viewDir }) {
  const r2 = brushR * brushR;
  const visited = new Uint8Array(tris.length);
  visited[seed] = 1;
  const queue = [seed];
  const out = new Set();
  let head = 0;

  while (head < queue.length) {
    const cur = queue[head++];
    const tri = tris[cur];

    // Inside-test: project each vertex onto plane through hitPt perpendicular
    // to viewDir, then point-to-triangle distance to projected triangle.
    const projected = tri.v.map(v => {
      const d = dot(sub(v, hitPt), viewDir);
      return [v[0] - d*viewDir[0], v[1] - d*viewDir[1], v[2] - d*viewDir[2]];
    });
    const d2 = distSqPointToTri(hitPt, ...projected);
    if (d2 > r2) continue;

    out.add(cur);

    for (const nb of adjacency[cur]) {
      if (visited[nb]) continue;
      visited[nb] = 1;
      // Cull back-facing neighbors (front-facing has normal opposing view dir)
      if (dot(tris[nb].normal, viewDir) >= 0) continue;
      queue.push(nb);
    }
  }
  return out;
}

// ── Mesh builders ────────────────────────────────────────────────────────────
// Closed thin shell (front + back planes joined by 4 side walls). Front/back
// have OPPOSITE normals — the user's original "back face gets selected" case.
function buildClosedThinShell(gap, sideTris = 2) {
  const tris = [];
  const step = 1.0;
  const n = sideTris;

  // Front face at z=0, normal +Z (CCW seen from +Z).
  for (let iy = 0; iy < n; iy++) {
    for (let ix = 0; ix < n; ix++) {
      const x0 = ix*step, x1 = x0+step, y0 = iy*step, y1 = y0+step;
      tris.push({ v: [[x0,y0,0],[x1,y0,0],[x1,y1,0]], side: 'front' });
      tris.push({ v: [[x0,y0,0],[x1,y1,0],[x0,y1,0]], side: 'front' });
    }
  }
  // Back face at z=-gap, normal -Z (CCW seen from -Z).
  for (let iy = 0; iy < n; iy++) {
    for (let ix = 0; ix < n; ix++) {
      const x0 = ix*step, x1 = x0+step, y0 = iy*step, y1 = y0+step;
      tris.push({ v: [[x0,y0,-gap],[x1,y1,-gap],[x1,y0,-gap]], side: 'back' });
      tris.push({ v: [[x0,y0,-gap],[x0,y1,-gap],[x1,y1,-gap]], side: 'back' });
    }
  }
  // Side walls (4 sides), each made of 2 tris. Normals point outward.
  const W = n * step;
  // -Y wall (normal -Y)
  tris.push({ v: [[0,0,0],[W,0,-gap],[W,0,0]], side: 'wall' });
  tris.push({ v: [[0,0,0],[0,0,-gap],[W,0,-gap]], side: 'wall' });
  // +Y wall (normal +Y)
  tris.push({ v: [[0,W,0],[W,W,0],[W,W,-gap]], side: 'wall' });
  tris.push({ v: [[0,W,0],[W,W,-gap],[0,W,-gap]], side: 'wall' });
  // -X wall (normal -X)
  tris.push({ v: [[0,0,0],[0,W,0],[0,W,-gap]], side: 'wall' });
  tris.push({ v: [[0,0,0],[0,W,-gap],[0,0,-gap]], side: 'wall' });
  // +X wall (normal +X)
  tris.push({ v: [[W,0,0],[W,0,-gap],[W,W,-gap]], side: 'wall' });
  tris.push({ v: [[W,0,0],[W,W,-gap],[W,W,0]], side: 'wall' });
  return tris;
}

// Two disconnected parallel layers, both facing +Z (the harder thin-gap case
// where normals don't help — only topology / occlusion can rescue).
function buildDoubleFront(gap, sideTris = 2) {
  const tris = [];
  const step = 1.0;
  const n = sideTris;
  for (const z of [0, -gap]) {
    for (let iy = 0; iy < n; iy++) {
      for (let ix = 0; ix < n; ix++) {
        const x0 = ix*step, x1 = x0+step, y0 = iy*step, y1 = y0+step;
        tris.push({ v: [[x0,y0,z],[x1,y0,z],[x1,y1,z]], side: z===0?'front':'back' });
        tris.push({ v: [[x0,y0,z],[x1,y1,z],[x0,y1,z]], side: z===0?'front':'back' });
      }
    }
  }
  return tris;
}

// ── Tests ────────────────────────────────────────────────────────────────────
const cases = [];
let pass = 0, fail = 0;

function check(name, cond, info = '') {
  if (cond) { pass++; cases.push(`  ✓ ${name}`); }
  else      { fail++; cases.push(`  ✗ ${name}${info ? ` — ${info}` : ''}`); }
}

// Camera looking down -Z; brush wider than the shell gap.
const camPos = [1.0, 1.0, 10.0];
const hitPt  = [1.0, 1.0, 0.0];
const viewDir = norm(sub(hitPt, camPos)); // (0,0,-1)

// Helper: find tri index for hit (front face containing hitPt at z=0)
function seedAtHit(tris) {
  // Find a front-side triangle whose centroid is close to hitPt in XY
  let best = -1, bestD = Infinity;
  for (let i = 0; i < tris.length; i++) {
    if (tris[i].side !== 'front') continue;
    const dx = tris[i].centroid[0] - hitPt[0];
    const dy = tris[i].centroid[1] - hitPt[1];
    const d = dx*dx + dy*dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

// ── Case 1: closed thin shell, brush wider than gap ──────────────────────────
{
  const tris = buildClosedThinShell(0.5, 2);
  annotate(tris);
  const adj = buildAdjacency(tris);
  const seed = seedAtHit(tris);
  const sel = bfsBrushSelect({ tris, adjacency: adj, seed, hitPt, brushR: 1.5, viewDir });

  const sides = [...sel].map(i => tris[i].side);
  const front = sides.filter(s => s === 'front').length;
  const back  = sides.filter(s => s === 'back').length;
  const wall  = sides.filter(s => s === 'wall').length;

  check('closed shell: front faces selected', front === 8, `got ${front}/8`);
  check('closed shell: NO back faces selected', back === 0, `got ${back} back tris`);
  check('closed shell: side walls (perpendicular to view) NOT selected', wall === 0, `got ${wall} walls`);
}

// ── Case 2: disconnected double-front layer (BFS topology saves us) ──────────
{
  const tris = buildDoubleFront(0.5, 2);
  annotate(tris);
  const adj = buildAdjacency(tris);
  const seed = seedAtHit(tris);
  const sel = bfsBrushSelect({ tris, adjacency: adj, seed, hitPt, brushR: 1.5, viewDir });

  const sides = [...sel].map(i => tris[i].side);
  const front = sides.filter(s => s === 'front').length;
  const back  = sides.filter(s => s === 'back').length;

  check('double-layer: front faces selected', front === 8, `got ${front}/8`);
  check('double-layer: NO back layer reached (disconnected)', back === 0, `got ${back} back tris`);
}

// ── Case 3: small brush stays local ──────────────────────────────────────────
{
  const tris = buildClosedThinShell(5.0, 4);
  annotate(tris);
  const adj = buildAdjacency(tris);
  const seed = seedAtHit(tris);
  const sel = bfsBrushSelect({ tris, adjacency: adj, seed, hitPt, brushR: 0.4, viewDir });

  const sides = [...sel].map(i => tris[i].side);
  const front = sides.filter(s => s === 'front').length;
  const back  = sides.filter(s => s === 'back').length;
  const wall  = sides.filter(s => s === 'wall').length;

  check('tight brush: at least one front face selected', front >= 1);
  check('tight brush: no back faces', back === 0);
  check('tight brush: no walls', wall === 0);
}

// ── Case 4: BFS work bounded by painted area, not mesh size ──────────────────
// Expand the front face mesh to 50×50 = 2500 front tris; tiny brush should
// still only walk a handful of triangles.
{
  const tris = buildClosedThinShell(0.5, 50); // 50*50*2 = 5000 front tris
  annotate(tris);
  const adj = buildAdjacency(tris);
  // Aim hit at center
  const center = [25.0, 25.0, 0.0];
  let best = -1, bestD = Infinity;
  for (let i = 0; i < tris.length; i++) {
    if (tris[i].side !== 'front') continue;
    const dx = tris[i].centroid[0] - center[0], dy = tris[i].centroid[1] - center[1];
    const d = dx*dx + dy*dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  let visits = 0;
  const sel = bfsBrushSelect({
    tris, adjacency: adj, seed: best, hitPt: center, brushR: 1.0, viewDir,
  });
  check('big mesh + small brush: only local tris selected (≤ 50)',
    sel.size <= 50, `selected ${sel.size}`);
  check('big mesh + small brush: NO back tris', [...sel].every(i => tris[i].side === 'front'));
}

console.log(cases.join('\n'));
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
