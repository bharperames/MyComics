// The comic's four corners in a picture, for the phone scan (scripts/export_static.py SCAN_PAGE): a port of ingest's
// edge-line finder (local_read.find_comic, refined by trim.locate, kept only when trim.comic_shaped) to OpenCV.js.
// studies/phone_edges.py measured the Python finder on 416 owned fronts: the scan's first guess 86.1% -> 90.6% from a
// guide-box shot, 80.0% -> 90.4% from a library photo, its "sure" answers 24% -> 70% and 6% -> 67%; the same study
// runs this file in headless Chromium (find --js) to check the port gives the same.
//
//   ScanEdges.find(cv, canvasOrImage) -> [[x, y] x4] (top-left, top-right, bottom-right, bottom-left) in its pixels, or null
//   ScanEdges.flatten(cv, source, quad, height) -> a canvas of the comic, flattened (0.68 wide per 1 tall)
(function (root) {
  'use strict';
  const WORK = 800, MASK_WORK = 1000, BORDER = 0.04, BG_SIGMA = 4.0, GROW = 0.02, AGREE = 0.03, SHAPES = [1.30, 1.60];

  // ---- small helpers ------------------------------------------------------------------------------------------
  const hyp = (x, y) => Math.sqrt(x * x + y * y);
  function rng(seed) {                    // a seeded generator (mulberry32), so a picture always gives the same lines
    let a = seed >>> 0;
    return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function percentile(arr, p) {          // numpy's default (linear) percentile
    const a = Float64Array.from(arr).sort(), k = (a.length - 1) * p / 100, f = Math.floor(k), c = Math.ceil(k);
    return f === c ? a[f] : a[f] + (a[c] - a[f]) * (k - f);
  }
  function orderCorners(q) {             // trim.order_corners
    const cx = (q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4, cy = (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4;
    const s = q.map(p => [p[0], p[1]]).sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
    let start = 0; for (let i = 1; i < 4; i++) if (s[i][0] + s[i][1] < s[start][0] + s[start][1]) start = i;
    return s.slice(start).concat(s.slice(0, start));
  }
  const polyArea = q => { let a = 0; for (let i = 0; i < q.length; i++) { const j = (i + 1) % q.length; a += q[i][0] * q[j][1] - q[j][0] * q[i][1]; } return Math.abs(a) / 2; };
  function solve(A, b) {                 // Gaussian elimination, A n x n (rows), b n
    const n = b.length, M = A.map((r, i) => r.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    return M.map((r, i) => r[n] / r[i]);
  }
  function inv3(m) {
    const [a, b, c, d, e, f, g, h, i] = [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]];
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
    return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det], [B / det, (a * i - c * g) / det, -(a * f - c * d) / det], [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
  }
  function matFrom(cv, source) {          // an RGB Mat of a canvas or image
    let c = source;
    if (!(source instanceof HTMLCanvasElement)) {
      c = document.createElement('canvas'); c.width = source.naturalWidth || source.videoWidth || source.width; c.height = source.naturalHeight || source.videoHeight || source.height;
      c.getContext('2d').drawImage(source, 0, 0);
    }
    const rgba = cv.imread(c), rgb = new cv.Mat(); cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB); rgba.delete();
    return rgb;
  }
  function resized(cv, rgb, long, interp) {
    const s = long / Math.max(rgb.rows, rgb.cols), out = new cv.Mat();
    cv.resize(rgb, out, new cv.Size(Math.round(rgb.cols * s), Math.round(rgb.rows * s)), 0, 0, interp);
    return [out, s];
  }
  const ones = (cv, k) => cv.Mat.ones(k, k, cv.CV_8U);

  // ---- trim.object_mask: everything unlike the table, the table learned from the picture's border ---------------
  function objectMask(cv, rgb) {
    const [small, s] = resized(cv, rgb, MASK_WORK, cv.INTER_AREA);
    const bl = new cv.Mat(); cv.GaussianBlur(small, bl, new cv.Size(5, 5), 0); small.delete();
    const labM = new cv.Mat(); cv.cvtColor(bl, labM, cv.COLOR_RGB2Lab); bl.delete();
    const sh = labM.rows, sw = labM.cols, lab = labM.data, b = Math.max(4, Math.round(BORDER * Math.min(sh, sw)));
    const basis = (x, y) => [1, x, y, x * x, y * y, x * y];
    const bx = [], by = [], bi = [];
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++)
      if (y < b || y >= sh - b || x < b || x >= sw - b) { bx.push(x / sw - 0.5); by.push(y / sh - 0.5); bi.push(y * sw + x); }
    const n = bi.length; let keep = new Uint8Array(n).fill(1), coef = null;
    const Y = k => [lab[bi[k] * 3], lab[bi[k] * 3 + 1], lab[bi[k] * 3 + 2]];
    for (let it = 0; it < 3; it++) {     // refit without outliers (a comic corner or board in the band)
      const AtA = Array.from({ length: 6 }, () => new Array(6).fill(0)), AtY = [0, 1, 2].map(() => new Array(6).fill(0));
      for (let k = 0; k < n; k++) if (keep[k]) {
        const a = basis(bx[k], by[k]), yv = Y(k);
        for (let i = 0; i < 6; i++) { for (let j = 0; j < 6; j++) AtA[i][j] += a[i] * a[j]; for (let ch = 0; ch < 3; ch++) AtY[ch][i] += a[i] * yv[ch]; }
      }
      coef = [0, 1, 2].map(ch => solve(AtA, AtY[ch]));
      const r = new Float64Array(n), kept = [];
      for (let k = 0; k < n; k++) {
        const a = basis(bx[k], by[k]), yv = Y(k); let d = 0;
        for (let ch = 0; ch < 3; ch++) { let p = 0; for (let i = 0; i < 6; i++) p += a[i] * coef[ch][i]; d += (yv[ch] - p) ** 2; }
        r[k] = Math.sqrt(d); if (keep[k]) kept.push(r[k]);
      }
      const lim = percentile(kept, 80); keep = new Uint8Array(n); for (let k = 0; k < n; k++) keep[k] = r[k] <= lim ? 1 : 0;
    }
    const resid = new Float32Array(sh * sw * 3);
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
      const a = basis(x / sw - 0.5, y / sh - 0.5), i = y * sw + x;
      for (let ch = 0; ch < 3; ch++) { let p = 0; for (let k = 0; k < 6; k++) p += a[k] * coef[ch][k]; resid[i * 3 + ch] = lab[i * 3 + ch] - p; }
    }
    labM.delete();
    // covariance of the kept band residuals (numpy's np.cov: n - 1), plus the identity
    const mean = [0, 0, 0]; let m = 0;
    for (let k = 0; k < n; k++) if (keep[k]) { for (let ch = 0; ch < 3; ch++) mean[ch] += resid[bi[k] * 3 + ch]; m++; }
    for (let ch = 0; ch < 3; ch++) mean[ch] /= m;
    const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let k = 0; k < n; k++) if (keep[k]) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] += (resid[bi[k] * 3 + i] - mean[i]) * (resid[bi[k] * 3 + j] - mean[j]);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] = cov[i][j] / (m - 1) + (i === j ? 1 : 0);
    const iv = inv3(cov), obj = new cv.Mat(sh, sw, cv.CV_8U), od = obj.data;
    for (let i = 0; i < sh * sw; i++) {
      const r0 = resid[i * 3], r1 = resid[i * 3 + 1], r2 = resid[i * 3 + 2];
      const d2 = r0 * (iv[0][0] * r0 + iv[0][1] * r1 + iv[0][2] * r2) + r1 * (iv[1][0] * r0 + iv[1][1] * r1 + iv[1][2] * r2) + r2 * (iv[2][0] * r0 + iv[2][1] * r1 + iv[2][2] * r2);
      od[i] = Math.sqrt(Math.max(0, d2)) > BG_SIGMA ? 1 : 0;
    }
    const k9 = ones(cv, 9), k5 = ones(cv, 5);
    cv.morphologyEx(obj, obj, cv.MORPH_CLOSE, k9); cv.morphologyEx(obj, obj, cv.MORPH_OPEN, k5); k9.delete(); k5.delete();
    return [obj, s];
  }

  // ---- straight lines -----------------------------------------------------------------------------------------
  function fitLines(cv, pts, nLines = 8, tol = 1.5, iters = 400) {   // local_read._lines: RANSAC, then a Huber fit
    const out = [], rand = rng(0); let P = pts;
    for (let l = 0; l < nLines; l++) {
      if (P.length < 30) break;
      let best = null, bestN = -1;
      for (let it = 0; it < iters; it++) {
        const i = Math.floor(rand() * P.length); let j = Math.floor(rand() * (P.length - 1)); if (j >= i) j++;
        const dx = P[j][0] - P[i][0], dy = P[j][1] - P[i][1], nr = hyp(dx, dy);
        if (nr < 10) continue;
        const nx = -dy / nr, ny = dx / nr; let c = 0;
        for (const p of P) if (Math.abs((p[0] - P[i][0]) * nx + (p[1] - P[i][1]) * ny) < tol) c++;
        if (c > bestN) { bestN = c; best = [P[i], nx, ny]; }
      }
      if (!best || bestN < 30) break;
      const inl = P.filter(p => Math.abs((p[0] - best[0][0]) * best[1] + (p[1] - best[0][1]) * best[2]) < tol);
      const m = cv.matFromArray(inl.length, 1, cv.CV_32FC2, inl.flat()), line = new cv.Mat();
      cv.fitLine(m, line, cv.DIST_HUBER, 0, 0.01, 0.01);
      const [vx, vy, x0, y0] = line.data32F; m.delete(); line.delete();
      out.push([[x0, y0], [vx, vy], inl.length]);
      P = P.filter(p => Math.abs((p[0] - x0) * -vy + (p[1] - y0) * vx) > 3 * tol);
    }
    return out;
  }
  function houghLines(cv, small, n = 14) {                          // local_read._hough_lines
    const g = new cv.Mat(), bl = new cv.Mat(), edges = new cv.Mat(), hl = new cv.Mat();
    cv.cvtColor(small, g, cv.COLOR_RGB2GRAY); cv.GaussianBlur(g, bl, new cv.Size(5, 5), 0); cv.Canny(bl, edges, 40, 120);
    cv.HoughLines(edges, hl, 1, Math.PI / 360, Math.floor(0.12 * Math.min(g.rows, g.cols)), 0, 0, 0, Math.PI);
    const out = [], seen = [];
    for (let k = 0; k < hl.rows && out.length < n; k++) {
      const rho = hl.data32F[k * 2], th = hl.data32F[k * 2 + 1];
      if (seen.some(([r, t]) => Math.abs(rho - r) < 8 && Math.min(Math.abs(th - t), Math.PI - Math.abs(th - t)) < 3 * Math.PI / 180)) continue;
      seen.push([rho, th]);
      const c = Math.cos(th), s = Math.sin(th);
      out.push([[c * rho, s * rho], [-s, c], 0]);
    }
    [g, bl, edges, hl].forEach(m => m.delete());
    return out;
  }
  function intersect(l1, l2) {
    const [p, d] = l1, [q, e] = l2, det = d[0] * -e[1] - (-e[0]) * d[1];
    if (Math.abs(det) < 1e-6) return null;
    const rx = q[0] - p[0], ry = q[1] - p[1], t = (rx * -e[1] - (-e[0]) * ry) / det;
    return [p[0] + t * d[0], p[1] + t * d[1]];
  }
  function quads(lines, sh, sw) {                                   // local_read._quads (its angle test kept as it is)
    const ang = lines.map(l => ((Math.atan2(l[1][1], l[1][0]) % Math.PI) + Math.PI) % Math.PI);
    const near = (a, b) => Math.min(Math.abs(a - b), Math.PI - Math.abs(a - b)) < 15 * Math.PI / 180;
    const sep = (a, b) => { const nx = -lines[a][1][1], ny = lines[a][1][0]; return Math.abs((lines[b][0][0] - lines[a][0][0]) * nx + (lines[b][0][1] - lines[a][0][1]) * ny); };
    const lim = 0.15 * Math.min(sh, sw), out = [], L = lines.length;
    for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++) {
      if (!near(ang[i], ang[j]) || sep(i, j) < lim) continue;
      for (let k = 0; k < L; k++) for (let m = k + 1; m < L; m++) {
        if (k === i || k === j || m === i || m === j || !near(ang[k], ang[m]) || !near(ang[k], ang[i] + Math.PI / 2)) continue;
        if (sep(k, m) < lim) continue;
        const cs = [intersect(lines[i], lines[k]), intersect(lines[i], lines[m]), intersect(lines[j], lines[k]), intersect(lines[j], lines[m])];
        if (cs.some(c => !c)) continue;
        out.push(orderCorners(cs));
      }
    }
    return out;
  }

  // ---- scoring a quad -----------------------------------------------------------------------------------------
  function rowSpans(q, h, w) {          // the pixels fillPoly would fill, as [y, x0, x1] (inclusive), by pixel centers
    const r = q.map(p => [Math.round(p[0]), Math.round(p[1])]);     // fillPoly is given the rounded corners
    const ys = r.map(p => p[1]), y0 = Math.max(0, Math.min(...ys)), y1 = Math.min(h - 1, Math.max(...ys));
    const spans = [];
    for (let y = y0; y <= y1; y++) {
      const xs = [];
      for (let k = 0; k < 4; k++) {
        const a = r[k], b = r[(k + 1) % 4];
        if ((a[1] <= y && b[1] > y) || (b[1] <= y && a[1] > y)) xs.push(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
        else if (a[1] === y && b[1] === y) xs.push(a[0], b[0]);   // a level edge on this row
      }
      if (xs.length < 2) continue;
      const x0 = Math.max(0, Math.ceil(Math.min(...xs))), x1 = Math.min(w - 1, Math.floor(Math.max(...xs)));
      if (x1 >= x0) spans.push([y, x0, x1]);
    }
    return spans;
  }
  function prefix(mask, h, w) {         // row prefix sums: P[y*(w+1)+x] = sum of mask[y, 0..x-1]
    const P = new Int32Array(h * (w + 1));
    for (let y = 0; y < h; y++) { let s = 0; for (let x = 0; x < w; x++) { s += mask[y * w + x]; P[y * (w + 1) + x + 1] = s; } }
    return P;
  }
  const labAt = (lab, w, h, x, y) => { const xi = Math.min(w - 1, Math.max(0, Math.round(x))), yi = Math.min(h - 1, Math.max(0, Math.round(y))), i = (yi * w + xi) * 3; return [lab[i], lab[i + 1], lab[i + 2]]; };
  function sideContrast(lab, w, h, a, b, c, off = 4) {               // local_read._side_contrast
    const dx = b[0] - a[0], dy = b[1] - a[1], nr = hyp(dx, dy) + 1e-9; let nx = -dy / nr, ny = dx / nr;
    if ((c[0] - (a[0] + b[0]) / 2) * nx + (c[1] - (a[1] + b[1]) / 2) * ny > 0) { nx = -nx; ny = -ny; }
    const d = [];
    for (let k = 0; k < 60; k++) {
      const t = 0.08 + (0.92 - 0.08) * k / 59, px = a[0] + t * dx, py = a[1] + t * dy;
      const u = labAt(lab, w, h, px - off * nx, py - off * ny), v = labAt(lab, w, h, px + off * nx, py + off * ny);
      d.push(hyp(hyp(u[0] - v[0], u[1] - v[1]), u[2] - v[2]));
    }
    return percentile(d, 25);
  }
  function insideBoard(lab, w, h, a, b, c, depth = 10) {             // local_read._inside_board
    const dx = b[0] - a[0], dy = b[1] - a[1], nr = hyp(dx, dy) + 1e-9; let nx = -dy / nr, ny = dx / nr;
    if ((c[0] - (a[0] + b[0]) / 2) * nx + (c[1] - (a[1] + b[1]) / 2) * ny < 0) { nx = -nx; ny = -ny; }
    let white = 0;
    for (let k = 0; k < 40; k++) {
      const t = 0.1 + 0.8 * k / 39; let all = true;
      for (const dd of [3, Math.floor(depth / 2) + 3, depth + 3]) {
        const v = labAt(lab, w, h, a[0] + t * dx + dd * nx, a[1] + t * dy + dd * ny);
        if (!(v[0] > 200 && Math.abs(v[1] - 128) < 7 && Math.abs(v[2] - 128) < 9)) { all = false; break; }
      }
      if (all) white++;
    }
    return white / 40;
  }
  function bestQuad(qs, lab, fg, bg, h, w) {                          // local_read._best_quad
    const Pf = prefix(fg, h, w), Pb = prefix(bg, h, w); let fgArea = 0; for (const v of fg) fgArea += v;
    let best = null, bestS = -1;
    for (const q of qs) {
      if (q.some(p => p[0] < -0.02 * w || p[0] > 1.02 * w || p[1] < -0.02 * h || p[1] > 1.02 * h)) continue;
      const sides = [0, 1, 2, 3].map(k => hyp(q[(k + 1) % 4][0] - q[k][0], q[(k + 1) % 4][1] - q[k][1]));
      const ratio = Math.max(sides[0] + sides[2], sides[1] + sides[3]) / Math.max(1, Math.min(sides[0] + sides[2], sides[1] + sides[3]));
      const shape = ratio >= SHAPES[0] && ratio <= SHAPES[1] ? 1 : Math.exp(-8 * Math.min(Math.abs(ratio - SHAPES[0]), Math.abs(ratio - SHAPES[1])));
      let area = 0, inFg = 0, inBg = 0;
      for (const [y, x0, x1] of rowSpans(q, h, w)) { const o = y * (w + 1); area += x1 - x0 + 1; inFg += Pf[o + x1 + 1] - Pf[o + x0]; inBg += Pb[o + x1 + 1] - Pb[o + x0]; }
      area = Math.max(1, area);
      const covered = inFg / Math.max(1, fgArea), table = inBg / area;
      const c = [(q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4, (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4];
      let edge = Infinity, board = 0;
      for (let k = 0; k < 4; k++) { edge = Math.min(edge, sideContrast(lab, w, h, q[k], q[(k + 1) % 4], c)); board = Math.max(board, insideBoard(lab, w, h, q[k], q[(k + 1) % 4], c)); }
      const score = shape * Math.min(covered / 0.9, 1) ** 3 * (1 - table) ** 2 * (1 - board) ** 2 * Math.min(edge, 40);
      if (score > bestS) { best = q; bestS = score; }
    }
    return best;
  }
  function fitQuad(cv, mask) {                                        // trim._fit_quad
    const contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(mask, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let big = -1, bigA = -1;
    for (let i = 0; i < contours.size(); i++) { const a = cv.contourArea(contours.get(i)); if (a > bigA) { bigA = a; big = i; } }
    let out = null;
    if (big >= 0) {
      const hull = new cv.Mat(); cv.convexHull(contours.get(big), hull, false, true);
      const peri = cv.arcLength(hull, true);
      for (let k = 0; k < 23 && !out; k++) {
        const eps = 0.005 + (0.06 - 0.005) * k / 22, ap = new cv.Mat();
        cv.approxPolyDP(hull, ap, eps * peri, true);
        if (ap.rows === 4) out = [0, 1, 2, 3].map(i => [ap.data32S[i * 2], ap.data32S[i * 2 + 1]]);
        ap.delete();
      }
      hull.delete();
    }
    contours.delete(); hier.delete();
    return out;
  }

  // ---- local_read.find_comic --------------------------------------------------------------------------------
  function findComic(cv, rgb, obj0) {
    const [small, s] = resized(cv, rgb, WORK, cv.INTER_AREA), sh = small.rows, sw = small.cols;
    const obj = new cv.Mat(); cv.resize(obj0, obj, new cv.Size(sw, sh), 0, 0, cv.INTER_NEAREST);
    const gc = new cv.Mat(sh, sw, cv.CV_8U, new cv.Scalar(cv.GC_PR_BGD)), g = gc.data, o = obj.data;
    for (let i = 0; i < sh * sw; i++) if (o[i] > 0) g[i] = cv.GC_PR_FGD;
    const core = new cv.Mat(), k41 = ones(cv, 41); cv.erode(obj, core, k41); k41.delete();
    const cc = new cv.Mat(), stats = new cv.Mat(), cent = new cv.Mat();
    const n = cv.connectedComponentsWithStats(core, cc, stats, cent);
    if (n > 1) {
      let k = 1; for (let i = 2; i < n; i++) if (stats.intAt(i, cv.CC_STAT_AREA) > stats.intAt(k, cv.CC_STAT_AREA)) k = i;
      const cd = cc.data32S; for (let i = 0; i < sh * sw; i++) if (cd[i] === k) g[i] = cv.GC_FGD;
    }
    [core, cc, stats, cent, obj].forEach(m => m.delete());
    const b = Math.max(6, Math.round(0.02 * Math.min(sh, sw)));
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) if (y < b || y >= sh - b || x < b || x >= sw - b) g[y * sw + x] = cv.GC_BGD;
    const bgr = new cv.Mat(); cv.cvtColor(small, bgr, cv.COLOR_RGB2BGR);
    const bgm = new cv.Mat(), fgm = new cv.Mat();
    try { cv.grabCut(bgr, gc, new cv.Rect(0, 0, 1, 1), bgm, fgm, 4, cv.GC_INIT_WITH_MASK); }
    catch (e) { [bgr, bgm, fgm, gc, small].forEach(m => m.delete()); return null; }
    [bgr, bgm, fgm].forEach(m => m.delete());
    const fg = new cv.Mat(sh, sw, cv.CV_8U), fd = fg.data;
    for (let i = 0; i < sh * sw; i++) fd[i] = g[i] === cv.GC_FGD || g[i] === cv.GC_PR_FGD ? 1 : 0;
    gc.delete();
    const k5 = ones(cv, 5); cv.morphologyEx(fg, fg, cv.MORPH_OPEN, k5); k5.delete();
    const contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(fg, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
    let big = -1, bigA = -1;
    for (let i = 0; i < contours.size(); i++) { const a = cv.contourArea(contours.get(i)); if (a > bigA) { bigA = a; big = i; } }
    if (big < 0) { contours.delete(); hier.delete(); fg.delete(); small.delete(); return null; }
    const cd = contours.get(big).data32S, pts = [];
    for (let i = 0; i < cd.length; i += 2) {
      const x = cd[i], y = cd[i + 1];
      if (x > b + 2 && x < sw - b - 3 && y > b + 2 && y < sh - b - 3) pts.push([x, y]);
    }
    contours.delete(); hier.delete();
    const bl = new cv.Mat(); cv.GaussianBlur(small, bl, new cv.Size(3, 3), 0);
    const labM = new cv.Mat(); cv.cvtColor(bl, labM, cv.COLOR_RGB2Lab); bl.delete();
    const lab = Float32Array.from(labM.data); labM.delete();
    const fgArr = Uint8Array.from(fg.data), bgArr = new Uint8Array(sh * sw);
    for (let i = 0; i < sh * sw; i++) {
      const white = lab[i * 3] > 200 && Math.abs(lab[i * 3 + 1] - 128) < 7 && Math.abs(lab[i * 3 + 2] - 128) < 9;
      bgArr[i] = !fgArr[i] && !white ? 1 : 0;
    }
    const lines = fitLines(cv, pts, 8).concat(houghLines(cv, small));
    small.delete();
    let q = bestQuad(quads(lines, sh, sw), lab, fgArr, bgArr, sh, sw);
    if (!q) q = fitQuad(cv, fg);
    fg.delete();
    return q ? q.map(p => [p[0] / s, p[1] / s]) : null;
  }

  // ---- trim.locate and trim.comic_shaped --------------------------------------------------------------------
  function locate(cv, obj, s, corners) {
    const g = orderCorners(corners), gs = g.map(p => [p[0] * s, p[1] * s]), size = Math.sqrt(polyArea(gs));
    const cx = gs.reduce((a, p) => a + p[0], 0) / 4, cy = gs.reduce((a, p) => a + p[1], 0) / 4, d = GROW * size;
    const grown = gs.map(p => { const vx = p[0] - cx, vy = p[1] - cy, nr = Math.max(hyp(vx, vy), 1e-6); return [p[0] + vx / nr * d * 1.4142, p[1] + vy / nr * d * 1.4142]; });
    const region = cv.Mat.zeros(obj.rows, obj.cols, cv.CV_8U), pv = new cv.MatVector();
    const poly = cv.matFromArray(4, 1, cv.CV_32SC2, grown.flatMap(p => [Math.trunc(p[0]), Math.trunc(p[1])]));
    pv.push_back(poly); cv.fillPoly(region, pv, new cv.Scalar(1)); poly.delete(); pv.delete();
    const both = new cv.Mat(); cv.bitwise_and(obj, region, both); region.delete();
    let fit = fitQuad(cv, both); both.delete();
    if (fit) {
      fit = orderCorners(fit);
      if (Math.max(...fit.map((p, i) => hyp(p[0] - gs[i][0], p[1] - gs[i][1]))) <= AGREE * size * 1.5) return fit.map(p => [p[0] / s, p[1] / s]);
    }
    return g;
  }
  function comicShaped(quad, w, h) {
    const q = orderCorners(quad);
    for (let k = 0; k < 4; k++) {
      const a = q[(k + 3) % 4], b = q[k], c = q[(k + 1) % 4], v1 = [a[0] - b[0], a[1] - b[1]], v2 = [c[0] - b[0], c[1] - b[1]];
      const cos = Math.max(-1, Math.min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / (hyp(...v1) * hyp(...v2))));
      if (Math.abs(Math.acos(cos) * 180 / Math.PI - 90) > 6) return false;
    }
    const ww = (hyp(q[1][0] - q[0][0], q[1][1] - q[0][1]) + hyp(q[2][0] - q[3][0], q[2][1] - q[3][1])) / 2;
    const hh = (hyp(q[3][0] - q[0][0], q[3][1] - q[0][1]) + hyp(q[2][0] - q[1][0], q[2][1] - q[1][1])) / 2;
    const area = polyArea(q) / (w * h), r = Math.min(ww, hh) / Math.max(ww, hh);
    return r >= 0.60 && r <= 0.82 && area >= 0.12 && area <= 0.9;
  }

  function find(cv, source) {
    const rgb = matFrom(cv, source);
    try {
      const [obj, s] = objectMask(cv, rgb);
      let q = findComic(cv, rgb, obj);
      if (q) q = locate(cv, obj, s, q);
      obj.delete();
      return q && comicShaped(q, rgb.cols, rgb.rows) ? orderCorners(q) : null;
    } finally { rgb.delete(); }
  }
  function flatten(cv, source, quad, height = 1000) {
    const W = Math.round(height * 0.68), rgb = matFrom(cv, source), out = new cv.Mat();
    const src = cv.matFromArray(4, 1, cv.CV_32FC2, orderCorners(quad).flat());
    const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, W - 1, 0, W - 1, height - 1, 0, height - 1]);
    const M = cv.getPerspectiveTransform(src, dst);
    cv.warpPerspective(rgb, out, M, new cv.Size(W, height), cv.INTER_AREA);
    const c = document.createElement('canvas'); c.width = W; c.height = height;
    const rgba = new cv.Mat(); cv.cvtColor(out, rgba, cv.COLOR_RGB2RGBA); cv.imshow(c, rgba);
    [rgb, out, src, dst, M, rgba].forEach(m => m.delete());
    return c;
  }
  root.ScanEdges = { find, flatten, comicShaped, orderCorners };
})(typeof window !== 'undefined' ? window : globalThis);
