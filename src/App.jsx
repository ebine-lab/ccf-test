import React, { useReducer, useState, useMemo, useRef, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine, Cell, ErrorBar } from 'recharts';
import Papa from 'papaparse';

const X_CLR = '#3b82f6';
const Y_CLR = '#ef4444';
const P_CLR = '#059669';
const V_CLR = '#8b5cf6';

/* ================================================================
   History reducer
   ================================================================ */
function histReducer(s, a) {
  switch (a.type) {
    case 'init': return { stack: [a.data], idx: 0 };
    case 'push': return { stack: [...s.stack.slice(0, s.idx + 1), a.data], idx: s.idx + 1 };
    case 'undo': return s.idx > 0 ? { ...s, idx: s.idx - 1 } : s;
    case 'redo': return s.idx < s.stack.length - 1 ? { ...s, idx: s.idx + 1 } : s;
    default: return s;
  }
}

/* ================================================================
   Math helpers
   ================================================================ */
function dft(sig) {
  const N = sig.length, re = new Float64Array(N), im = new Float64Array(N);
  for (let k = 0; k < N; k++)
    for (let n = 0; n < N; n++) {
      const a = (-2 * Math.PI * k * n) / N;
      re[k] += sig[n] * Math.cos(a);
      im[k] += sig[n] * Math.sin(a);
    }
  return { re, im };
}
function idft(re, im) {
  const N = re.length, out = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    for (let k = 0; k < N; k++) {
      const a = (2 * Math.PI * k * n) / N;
      out[n] += re[k] * Math.cos(a) - im[k] * Math.sin(a);
    }
    out[n] /= N;
  }
  return Array.from(out);
}
function lowpass(signal, cutoffDays) {
  const N = signal.length;
  if (N < 4 || cutoffDays >= N) return [...signal];
  const { re, im } = dft(signal);
  const kMax = Math.floor(N / cutoffDays);
  for (let k = 0; k < N; k++) {
    const kEff = k <= N / 2 ? k : N - k;
    if (kEff > kMax) { re[k] = 0; im[k] = 0; }
  }
  return idft(re, im);
}
function medianFilt(signal, win) {
  const half = Math.floor(win / 2);
  return signal.map((_, i) => {
    const s = Math.max(0, i - half), e = Math.min(signal.length, i + half + 1);
    const w = signal.slice(s, e).sort((a, b) => a - b);
    return w[Math.floor(w.length / 2)];
  });
}

function ccfOne(x, y, maxLag) {
  const n = x.length;
  if (n < 4) return null;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  const sx = Math.sqrt(x.reduce((s, v) => s + (v - mx) ** 2, 0) / n);
  const sy = Math.sqrt(y.reduce((s, v) => s + (v - my) ** 2, 0) / n);
  if (sx < 1e-12 || sy < 1e-12) return null;
  const out = [];
  for (let tau = -maxLag; tau <= maxLag; tau++) {
    let sum = 0, cnt = 0;
    for (let t = 0; t < n; t++) {
      const ty = t + tau;
      if (ty >= 0 && ty < n) { sum += (x[t] - mx) * (y[ty] - my); cnt++; }
    }
    if (cnt > 2) out.push({ tau, r: sum / (cnt * sx * sy) });
  }
  return out;
}

function fitAR1(series) {
  const n = series.length;
  if (n < 5) return null;
  const mean = series.reduce((a, b) => a + b, 0) / n;
  const c = series.map(v => v - mean);
  let num = 0, den = 0;
  for (let t = 1; t < n; t++) { num += c[t] * c[t - 1]; den += c[t - 1] ** 2; }
  const phi = den > 1e-12 ? num / den : 0;
  const res = [];
  for (let t = 1; t < n; t++) res.push(c[t] - phi * c[t - 1]);
  return { phi, residuals: res };
}

function applyFilter(patients, fn) {
  return patients.map(p => {
    const xF = fn(p.points.map(pt => pt.x));
    const yF = fn(p.points.map(pt => pt.y));
    return { ...p, points: p.points.map((pt, i) => ({ t: pt.t, x: xF[i], y: yF[i] })) };
  });
}

/* Aggregate: compute mean ± SE across patients at each lag */
function aggregateLag(allArrays, maxLag) {
  const out = [];
  for (let tau = -maxLag; tau <= maxLag; tau++) {
    const vals = allArrays
      .map(arr => arr?.find(d => d.tau === tau)?.r)
      .filter(v => v != null);
    if (vals.length === 0) { out.push({ tau, mean: 0, se: 0, n: 0 }); continue; }
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (vals.length - 1 || 1);
    out.push({ tau, mean, se: Math.sqrt(variance / vals.length), n: vals.length });
  }
  return out;
}

/* Find τ* and r* per patient */
function findPeak(ccfArr) {
  if (!ccfArr || ccfArr.length === 0) return null;
  let best = ccfArr[0];
  for (const d of ccfArr) if (Math.abs(d.r) > Math.abs(best.r)) best = d;
  return { tau: best.tau, r: best.r };
}

/* ================================================================
   CSV parser
   ================================================================ */
function parseCSV(text) {
  const result = Papa.parse(text, { header: false, skipEmptyLines: true, dynamicTyping: true });
  const rows = result.data.slice(1);
  const grouped = {};
  for (const row of rows) {
    if (!row || row.length < 12) continue;
    const id = String(row[0] ?? '').trim();
    const t = Number(row[1]), x = Number(row[2]), y = Number(row[11]);
    const k = Number(row[10]);
    if (!id || isNaN(t) || isNaN(x) || isNaN(y)) continue;
    if (!grouped[id]) grouped[id] = [];
    grouped[id].push({ t, x, y, k: isNaN(k) ? 0 : k });
  }
  return Object.entries(grouped)
    .map(([id, points]) => ({ id, points: points.sort((a, b) => a.t - b.t) }))
    .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));
}

/* ================================================================
   Sparkline
   ================================================================ */
function Sparkline({ patient, isSelected, onClick }) {
  const W = 168, H = 58, P = 3;
  const pts = patient.points;
  if (pts.length < 2) return (
    <div style={{ width: W + 8, height: H + 20, background: '#fff', borderRadius: 4, border: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#94a3b8' }}>
      {patient.id}: n&lt;2
    </div>
  );
  const xV = pts.map(p => p.x), yV = pts.map(p => p.y);
  const tR = [pts[0].t, pts[pts.length - 1].t];
  const xR = [Math.min(...xV), Math.max(...xV)], yR = [Math.min(...yV), Math.max(...yV)];
  const st = t => P + (t - tR[0]) / ((tR[1] - tR[0]) || 1) * (W - 2 * P);
  const sxV = v => H - P - (v - xR[0]) / ((xR[1] - xR[0]) || 1) * (H - 2 * P);
  const syV = v => H - P - (v - yR[0]) / ((yR[1] - yR[0]) || 1) * (H - 2 * P);
  const xPts = pts.map(p => `${st(p.t)},${sxV(p.x)}`).join(' ');
  const yPts = pts.map(p => `${st(p.t)},${syV(p.y)}`).join(' ');
  return (
    <div onClick={onClick} style={{ cursor: 'pointer', border: isSelected ? '2px solid #3b82f6' : '1px solid #e2e8f0', borderRadius: 4, padding: '2px 4px', background: isSelected ? '#eff6ff' : '#fff', transition: 'border-color 0.15s' }}>
      <div style={{ fontSize: 9, color: '#475569', fontFamily: 'monospace', display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ fontWeight: 600 }}>{patient.id}</span>
        <span style={{ color: '#94a3b8' }}>d{tR[0]}–{tR[1]}</span>
      </div>
      <svg width={W} height={H} style={{ display: 'block' }}>
        <polyline points={xPts} fill="none" stroke={X_CLR} strokeWidth="1.5" strokeLinejoin="round" />
        <polyline points={yPts} fill="none" stroke={Y_CLR} strokeWidth="1.5" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

/* ================================================================
   Detail panel (selected patient)
   ================================================================ */
function DetailPanel({ patient, maxLag }) {
  const pts = patient.points;
  const x = pts.map(p => p.x), y = pts.map(p => p.y);
  const effLag = Math.min(maxLag, Math.floor(pts.length / 2) - 1);
  const c = effLag >= 1 ? ccfOne(x, y, effLag) : null;
  const chartData = pts.map(p => ({ t: p.t, x: p.x, y: p.y }));
  const ccfData = c ? c.map(d => ({ tau: d.tau, r: d.r })) : [];

  // partial CCF for this patient
  const arX = fitAR1(x), arY = fitAR1(y);
  const pc = (arX && arY && effLag >= 1) ? ccfOne(arX.residuals, arY.residuals, Math.min(effLag, Math.floor(arX.residuals.length / 2) - 1)) : null;
  const pccfData = pc ? pc.map(d => ({ tau: d.tau, r: d.r })) : [];

  return (
    <div style={{ background: '#fff', border: '1px solid #cbd5e1', borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 8, color: '#0f172a' }}>
        {patient.id} — 詳細 (n={pts.length})
        {arX && <span style={{ fontWeight: 400, fontSize: 11, color: '#64748b', marginLeft: 12 }}>φ(X)={arX.phi.toFixed(3)}, φ(Y)={arY?.phi.toFixed(3)}</span>}
      </div>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 320px', minWidth: 260 }}>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
            <span style={{ color: X_CLR }}>■</span> X 離床(min)　<span style={{ color: Y_CLR }}>■</span> Y 心拍(bpm)
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="t" tick={{ fontSize: 10 }} />
              <YAxis yAxisId="x" tick={{ fontSize: 10 }} stroke={X_CLR} width={36} />
              <YAxis yAxisId="y" orientation="right" tick={{ fontSize: 10 }} stroke={Y_CLR} width={36} />
              <Tooltip contentStyle={{ fontSize: 11 }} />
              <Bar yAxisId="x" dataKey="x" fill={X_CLR} opacity={0.5} />
              <Bar yAxisId="y" dataKey="y" fill={Y_CLR} opacity={0.5} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        {ccfData.length > 0 && (
          <div style={{ flex: '1 1 260px', minWidth: 220 }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>個別 CCF(τ)</div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={ccfData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="tau" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} domain={[-1, 1]} width={32} />
                <Tooltip contentStyle={{ fontSize: 11 }} formatter={v => [Number(v).toFixed(3), 'CCF']} />
                <ReferenceLine y={0} stroke="#94a3b8" />
                <Bar dataKey="r">{ccfData.map((d, i) => <Cell key={i} fill={d.r >= 0 ? X_CLR : Y_CLR} opacity={0.8} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        {pccfData.length > 0 && (
          <div style={{ flex: '1 1 260px', minWidth: 220 }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>個別 偏CCF(τ) <span style={{ color: P_CLR }}>AR(1)残差</span></div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={pccfData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="tau" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} domain={[-1, 1]} width={32} />
                <Tooltip contentStyle={{ fontSize: 11 }} formatter={v => [Number(v).toFixed(3), 'pCCF']} />
                <ReferenceLine y={0} stroke="#94a3b8" />
                <Bar dataKey="r">{pccfData.map((d, i) => <Cell key={i} fill={d.r >= 0 ? P_CLR : Y_CLR} opacity={0.8} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}

/* ================================================================
   Mini bar chart component (reusable)
   ================================================================ */
function LagBarChart({ data, title, subtitle, color, yDomain, showErrorBar }) {
  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2, color: '#334155' }}>{title}</div>
      {subtitle && <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>{subtitle}</div>}
      <ResponsiveContainer width="100%" height={190}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="tau" tick={{ fontSize: 10 }} label={{ value: 'τ (day)', position: 'insideBottom', offset: -2, fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} width={40} domain={yDomain} />
          <Tooltip contentStyle={{ fontSize: 11 }} formatter={(v, name) => [Number(v).toFixed(4), name]} />
          <ReferenceLine y={0} stroke="#64748b" strokeWidth={1} />
          <Bar dataKey="mean">
            {data.map((d, i) => <Cell key={i} fill={d.mean >= 0 ? (color || X_CLR) : Y_CLR} opacity={0.8} />)}
            {showErrorBar && <ErrorBar dataKey="se" width={2} strokeWidth={1} stroke="#475569" />}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function HistChart({ data, dataKey, xKey, title, color, xLabel }) {
  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4, color: '#334155' }}>{title}</div>
      <ResponsiveContainer width="100%" height={190}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey={xKey} tick={{ fontSize: 9 }} interval={xKey === 'label' ? 3 : 0} label={xLabel ? { value: xLabel, position: 'insideBottom', offset: -2, fontSize: 10 } : undefined} />
          <YAxis tick={{ fontSize: 10 }} width={28} allowDecimals={false} />
          <Tooltip contentStyle={{ fontSize: 11 }} />
          <Bar dataKey={dataKey}>
            {data.map((d, i) => {
              let fill = color || V_CLR;
              if (xKey === 'label') {
                const mid = (d.lo + d.hi) / 2;
                fill = mid >= 0 ? X_CLR : Y_CLR;
              }
              return <Cell key={i} fill={fill} opacity={0.75} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ================================================================
   CCF summary panel
   ================================================================ */
function CCFSummary({ results, maxLag }) {
  const { meanCCF, meanACFx, meanACFy, meanPCCF, tauStarsCCF, rStarsCCF, tauStarsPCCF, rStarsPCCF, validN, validNp, perPatient, fixedLagCCF, fixedLagPCCF } = results;

  const tauHistCCF = useMemo(() => {
    const c = {}; for (let t = -maxLag; t <= maxLag; t++) c[t] = 0;
    tauStarsCCF.forEach(t => { if (c[t] !== undefined) c[t]++; });
    return Object.entries(c).map(([tau, count]) => ({ tau: Number(tau), count }));
  }, [tauStarsCCF, maxLag]);

  const tauHistPCCF = useMemo(() => {
    const c = {}; for (let t = -maxLag; t <= maxLag; t++) c[t] = 0;
    tauStarsPCCF.forEach(t => { if (c[t] !== undefined) c[t]++; });
    return Object.entries(c).map(([tau, count]) => ({ tau: Number(tau), count }));
  }, [tauStarsPCCF, maxLag]);

  const binR = useCallback((vals) => {
    const nBins = 20, lo = -1, hi = 1, bw = (hi - lo) / nBins;
    const bins = Array.from({ length: nBins }, (_, i) => ({
      label: (lo + (i + 0.5) * bw).toFixed(2), lo: lo + i * bw, hi: lo + (i + 1) * bw, count: 0,
    }));
    vals.forEach(r => { const idx = Math.min(Math.floor((r - lo) / bw), nBins - 1); if (idx >= 0) bins[idx].count++; });
    return bins;
  }, []);

  const rHistCCF = useMemo(() => binR(rStarsCCF), [rStarsCCF, binR]);
  const rHistPCCF = useMemo(() => binR(rStarsPCCF), [rStarsPCCF, binR]);

  const peakCCF = meanCCF.reduce((b, d) => Math.abs(d.mean) > Math.abs(b.mean) ? d : b, { mean: 0, tau: 0 });
  const peakPCCF = meanPCCF.reduce((b, d) => Math.abs(d.mean) > Math.abs(b.mean) ? d : b, { mean: 0, tau: 0 });

  /* CSV export */
  const exportCSV = useCallback(() => {
    const lines = [];
    lines.push('# CCF Analysis Results');
    lines.push(`# Max Lag: ${maxLag}`);
    lines.push(`# Valid patients (CCF): ${validN}`);
    lines.push(`# Valid patients (partial CCF): ${validNp}`);
    lines.push('');
    lines.push('## Lag-level Statistics');
    lines.push('tau,mean_CCF,SE_CCF,n_CCF,mean_ACF_X,SE_ACF_X,mean_ACF_Y,SE_ACF_Y,mean_partialCCF,SE_partialCCF,n_partialCCF');
    for (let tau = -maxLag; tau <= maxLag; tau++) {
      const c = meanCCF.find(d => d.tau === tau) || { mean: '', se: '', n: '' };
      const ax = meanACFx.find(d => d.tau === tau) || { mean: '', se: '' };
      const ay = meanACFy.find(d => d.tau === tau) || { mean: '', se: '' };
      const pc = meanPCCF.find(d => d.tau === tau) || { mean: '', se: '', n: '' };
      lines.push([tau, c.mean, c.se, c.n, ax.mean, ax.se, ay.mean, ay.se, pc.mean, pc.se, pc.n].join(','));
    }
    lines.push('');
    lines.push('## Per-patient Statistics');
    lines.push('patient_id,n_days,tau_star_CCF,r_star_CCF,tau_star_partialCCF,r_star_partialCCF,phi_X,phi_Y');
    perPatient.forEach(p => {
      lines.push([p.id, p.nDays, p.tauCCF ?? '', p.rCCF ?? '', p.tauPCCF ?? '', p.rPCCF ?? '', p.phiX ?? '', p.phiY ?? ''].join(','));
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'ccf_analysis_results.csv'; a.click();
    URL.revokeObjectURL(url);
  }, [meanCCF, meanACFx, meanACFy, meanPCCF, perPatient, maxLag, validN, validNp]);

  const G = { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 };

  return (
    <div style={{ background: '#fff', border: '1px solid #cbd5e1', borderRadius: 8, padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4, flexWrap: 'wrap' }}>
        <div style={{ fontWeight: 700, fontSize: 15, color: '#0f172a' }}>CCF 集団解析結果</div>
        <button onClick={exportCSV} style={{ marginLeft: 'auto', padding: '5px 14px', fontSize: 12, fontWeight: 600, borderRadius: 6, border: '1px solid #cbd5e1', background: '#fff', color: '#334155', cursor: 'pointer' }}>
          📥 CSVエクスポート
        </button>
      </div>
      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 14 }}>
        有効患者 CCF:{validN} / 偏CCF:{validNp} | τ∈[−{maxLag}, +{maxLag}] | CCF(τ)=Corr(X<sub>t</sub>, Y<sub>t+τ</sub>) → τ&gt;0: 活動→心拍, τ&lt;0: 心拍→活動
      </div>

      {/* Row 1: CCF, ACF(X), ACF(Y), Partial CCF */}
      <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a', marginBottom: 8, borderBottom: '1px solid #e2e8f0', paddingBottom: 4 }}>
        ラグ構造の診断
      </div>
      <div style={G}>
        <LagBarChart data={meanCCF} title="① 集団平均 CCF(τ) ± SE" subtitle={`peak: τ=${peakCCF.tau}, r̄=${peakCCF.mean.toFixed(3)}`} color={X_CLR} showErrorBar />
        <LagBarChart data={meanACFx} title="② ACF(X) 離床時間の自己相関" subtitle="X系列の持続性/反転傾向" color={X_CLR} showErrorBar />
        <LagBarChart data={meanACFy} title="③ ACF(Y) 心拍の自己相関" subtitle="Y系列の持続性/反転傾向" color={Y_CLR} showErrorBar />
        <LagBarChart data={meanPCCF} title="④ 偏CCF(τ) AR(1)残差 ± SE" subtitle={`peak: τ=${peakPCCF.tau}, r̄=${peakPCCF.mean.toFixed(3)}　※自己相関除去後`} color={P_CLR} showErrorBar />
      </div>

      {/* Row 2: CCF histograms */}
      <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a', marginTop: 20, marginBottom: 8, borderBottom: '1px solid #e2e8f0', paddingBottom: 4 }}>
        患者別ピーク分布 — CCF
      </div>
      <div style={G}>
        <HistChart data={tauHistCCF} dataKey="count" xKey="tau" title="⑤ τ* 分布（|CCF|最大ラグ）" color={V_CLR} xLabel="τ* (day)" />
        <HistChart data={rHistCCF} dataKey="count" xKey="label" title="⑥ r* 分布（τ*でのCCF値）" xLabel="r*" />
      </div>

      {/* Row 3: Partial CCF histograms */}
      <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a', marginTop: 20, marginBottom: 8, borderBottom: '1px solid #e2e8f0', paddingBottom: 4 }}>
        患者別ピーク分布 — 偏CCF（AR(1)残差）
      </div>
      <div style={G}>
        <HistChart data={tauHistPCCF} dataKey="count" xKey="tau" title="⑦ τ* 分布（|偏CCF|最大ラグ）" color={P_CLR} xLabel="τ* (day)" />
        <HistChart data={rHistPCCF} dataKey="count" xKey="label" title="⑧ r* 分布（τ*での偏CCF値）" xLabel="r*" />
      </div>

      {/* Row 4: Fixed-lag CCF distributions */}
      {fixedLagCCF && (
        <>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a', marginTop: 20, marginBottom: 4, borderBottom: '1px solid #e2e8f0', paddingBottom: 4 }}>
            固定ラグ CCF(τ) の患者別分布 — 選択バイアスなし
          </div>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 8 }}>
            各τを固定し、96名の CCF(τ) 値をヒストグラム化。二峰ならサブポピュレーション、単峰なら max|CCF| の選択アーティファクト。
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 10 }}>
            {[-5,-4,-3,-2,-1,0,1,2,3,4,5].map(tau => {
              const vals = fixedLagCCF[tau] || [];
              if (vals.length === 0) return null;
              const bins = binR(vals);
              const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
              const sd = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (vals.length - 1 || 1));
              return (
                <div key={tau}>
                  <HistChart
                    data={bins} dataKey="count" xKey="label"
                    title={`CCF τ=${tau} (n=${vals.length}, r̄=${mean.toFixed(3)}±${(sd / Math.sqrt(vals.length)).toFixed(3)})`}
                    xLabel="r"
                  />
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Row 5: Fixed-lag partial CCF distributions */}
      {fixedLagPCCF && (
        <>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a', marginTop: 20, marginBottom: 4, borderBottom: '1px solid #e2e8f0', paddingBottom: 4 }}>
            固定ラグ 偏CCF(τ) の患者別分布 — AR(1)残差
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 10 }}>
            {[-5,-4,-3,-2,-1,0,1,2,3,4,5].map(tau => {
              const vals = fixedLagPCCF[tau] || [];
              if (vals.length === 0) return null;
              const bins = binR(vals);
              const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
              const sd = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (vals.length - 1 || 1));
              return (
                <div key={tau}>
                  <HistChart
                    data={bins} dataKey="count" xKey="label"
                    title={`pCCF τ=${tau} (n=${vals.length}, r̄=${mean.toFixed(3)}±${(sd / Math.sqrt(vals.length)).toFixed(3)})`}
                    xLabel="r"
                  />
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

/* ================================================================
   Main Dashboard
   ================================================================ */
export default function CCFDashboard() {
  const [hist, dispatch] = useReducer(histReducer, { stack: [], idx: -1 });
  const [ccfResult, setCcfResult] = useState(null);
  const [cutoff, setCutoff] = useState('5');
  const [medWin, setMedWin] = useState('3');
  const [lagMax, setLagMax] = useState('7');
  const [kThresh, setKThresh] = useState('100');
  const [selected, setSelected] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [computing, setComputing] = useState(false);
  const fileRef = useRef(null);

  const data = hist.idx >= 0 ? hist.stack[hist.idx] : [];
  const canUndo = hist.idx > 0, canRedo = hist.idx < hist.stack.length - 1;

  const handleFile = useCallback((file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
      const patients = parseCSV(e.target.result);
      if (patients.length === 0) { alert('有効なデータが見つかりません'); return; }
      dispatch({ type: 'init', data: patients });
      setCcfResult(null); setSelected(null);
    };
    reader.readAsText(file);
  }, []);

  const doLowpass = useCallback(() => {
    const c = Number(cutoff);
    if (isNaN(c) || c < 2) { alert('カットオフ≥2日'); return; }
    dispatch({ type: 'push', data: applyFilter(data, s => lowpass(s, c)) });
    setCcfResult(null);
  }, [data, cutoff]);

  const doMedian = useCallback(() => {
    const w = Math.round(Number(medWin));
    if (isNaN(w) || w < 2) { alert('窓幅≥2'); return; }
    dispatch({ type: 'push', data: applyFilter(data, s => medianFilt(s, w)) });
    setCcfResult(null);
  }, [data, medWin]);

  const doTrimEdges = useCallback(() => {
    const trimmed = data
      .map(p => ({ ...p, points: p.points.length > 2 ? p.points.slice(1, -1) : p.points }))
      .filter(p => p.points.length >= 2);
    if (trimmed.length === 0) { alert('カット後にデータが残りません'); return; }
    dispatch({ type: 'push', data: trimmed });
    setCcfResult(null);
  }, [data]);

  const doInterpY = useCallback(() => {
    let totalFixed = 0;
    const patched = data.map(p => {
      const pts = p.points.map(pt => ({ ...pt }));
      // collect indices where y is 0 (or essentially 0)
      const zeroIdx = [];
      pts.forEach((pt, i) => { if (pt.y === 0 || pt.y < 1) zeroIdx.push(i); });
      if (zeroIdx.length === 0) return p;
      if (zeroIdx.length === pts.length) return p; // all zero, can't fix
      totalFixed += zeroIdx.length;
      for (const i of zeroIdx) {
        // find nearest non-zero before and after
        let before = null, after = null;
        for (let j = i - 1; j >= 0; j--) { if (pts[j].y > 1) { before = j; break; } }
        for (let j = i + 1; j < pts.length; j++) { if (pts[j].y > 1) { after = j; break; } }
        if (before !== null && after !== null) {
          const frac = (i - before) / (after - before);
          pts[i] = { ...pts[i], y: pts[before].y + frac * (pts[after].y - pts[before].y) };
        } else if (before !== null) {
          pts[i] = { ...pts[i], y: pts[before].y };
        } else if (after !== null) {
          pts[i] = { ...pts[i], y: pts[after].y };
        }
      }
      return { ...p, points: pts };
    });
    if (totalFixed === 0) { alert('Y=0のデータポイントはありませんでした'); return; }
    dispatch({ type: 'push', data: patched });
    setCcfResult(null);
    alert(`${totalFixed}点のY=0を線形補間しました`);
  }, [data]);

  const doFilterByK = useCallback(() => {
    const th = Number(kThresh);
    if (isNaN(th) || th <= 0) { alert('閾値は正の数を指定してください'); return; }
    let totalFixed = 0;
    const patched = data.map(p => {
      const pts = p.points.map(pt => ({ ...pt }));
      const badIdx = [];
      pts.forEach((pt, i) => { if (pt.k >= th) badIdx.push(i); });
      if (badIdx.length === 0) return p;
      if (badIdx.length === pts.length) return p;
      totalFixed += badIdx.length;
      for (const i of badIdx) {
        let before = null, after = null;
        for (let j = i - 1; j >= 0; j--) { if (pts[j].k < th && pts[j].y > 1) { before = j; break; } }
        for (let j = i + 1; j < pts.length; j++) { if (pts[j].k < th && pts[j].y > 1) { after = j; break; } }
        if (before !== null && after !== null) {
          const frac = (i - before) / (after - before);
          pts[i] = { ...pts[i], y: pts[before].y + frac * (pts[after].y - pts[before].y) };
        } else if (before !== null) {
          pts[i] = { ...pts[i], y: pts[before].y };
        } else if (after !== null) {
          pts[i] = { ...pts[i], y: pts[after].y };
        }
      }
      return { ...p, points: pts };
    });
    if (totalFixed === 0) { alert(`体動指数≥${th}のデータポイントはありませんでした`); return; }
    dispatch({ type: 'push', data: patched });
    setCcfResult(null);
    alert(`${totalFixed}点（体動指数≥${th}）のYを線形補間しました`);
  }, [data, kThresh]);

  const doCCF = useCallback(() => {
    const ml = Math.round(Number(lagMax));
    if (isNaN(ml) || ml < 1) { alert('最大ラグ≥1'); return; }
    setComputing(true);
    // Use setTimeout to allow UI to update with "computing" state
    setTimeout(() => {
      const allCCF = [], allACFx = [], allACFy = [], allPCCF = [];
      const tauStarsCCF = [], rStarsCCF = [], tauStarsPCCF = [], rStarsPCCF = [];
      const perPatient = [];

      for (const p of data) {
        const x = p.points.map(pt => pt.x), y = p.points.map(pt => pt.y);
        const effLag = Math.min(ml, Math.floor(x.length / 2) - 1);
        const entry = { id: p.id, nDays: p.points.length, tauCCF: null, rCCF: null, tauPCCF: null, rPCCF: null, phiX: null, phiY: null };

        if (effLag >= 1) {
          // CCF
          const c = ccfOne(x, y, effLag);
          if (c) { allCCF.push(c); const pk = findPeak(c); if (pk) { tauStarsCCF.push(pk.tau); rStarsCCF.push(pk.r); entry.tauCCF = pk.tau; entry.rCCF = pk.r; } }
          // ACF(X), ACF(Y)
          const ax = ccfOne(x, x, effLag); if (ax) allACFx.push(ax);
          const ay = ccfOne(y, y, effLag); if (ay) allACFy.push(ay);
          // Partial CCF
          const arX = fitAR1(x), arY = fitAR1(y);
          if (arX && arY) {
            entry.phiX = arX.phi; entry.phiY = arY.phi;
            const pLag = Math.min(effLag, Math.floor(arX.residuals.length / 2) - 1);
            if (pLag >= 1) {
              const pc = ccfOne(arX.residuals, arY.residuals, pLag);
              if (pc) { allPCCF.push(pc); const pk = findPeak(pc); if (pk) { tauStarsPCCF.push(pk.tau); rStarsPCCF.push(pk.r); entry.tauPCCF = pk.tau; entry.rPCCF = pk.r; } }
            }
          }
        }
        perPatient.push(entry);
      }

      setCcfResult({
        meanCCF: aggregateLag(allCCF, ml),
        meanACFx: aggregateLag(allACFx, ml),
        meanACFy: aggregateLag(allACFy, ml),
        meanPCCF: aggregateLag(allPCCF, ml),
        tauStarsCCF, rStarsCCF, tauStarsPCCF, rStarsPCCF,
        validN: allCCF.length, validNp: allPCCF.length,
        perPatient,
        fixedLagCCF: (() => {
          const out = {};
          for (let tau = -ml; tau <= ml; tau++) {
            out[tau] = allCCF.map(arr => arr.find(d => d.tau === tau)?.r).filter(v => v != null);
          }
          return out;
        })(),
        fixedLagPCCF: (() => {
          const out = {};
          for (let tau = -ml; tau <= ml; tau++) {
            out[tau] = allPCCF.map(arr => arr.find(d => d.tau === tau)?.r).filter(v => v != null);
          }
          return out;
        })(),
      });
      setComputing(false);
    }, 50);
  }, [data, lagMax]);

  const selectedPatient = useMemo(() => selected ? data.find(p => p.id === selected) : null, [data, selected]);
  const stats = useMemo(() => {
    if (data.length === 0) return null;
    const lens = data.map(p => p.points.length);
    return { n: data.length, minLen: Math.min(...lens), maxLen: Math.max(...lens), meanLen: (lens.reduce((a, b) => a + b, 0) / lens.length).toFixed(1) };
  }, [data]);

  const btn = (bg, fg) => ({ padding: '5px 12px', fontSize: 12, fontWeight: 600, borderRadius: 6, border: 'none', cursor: 'pointer', background: bg, color: fg, transition: 'opacity 0.15s' });
  const inputSt = { width: 52, padding: '4px 6px', fontSize: 12, border: '1px solid #cbd5e1', borderRadius: 4, textAlign: 'center' };

  /* === Upload screen === */
  if (data.length === 0) {
    return (
      <div style={{ minHeight: '100vh', background: '#f8fafc', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontFamily: 'Inter, -apple-system, sans-serif', padding: 24 }}>
        <div style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', marginBottom: 6 }}>CCF Dashboard</div>
        <div style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>離床時間 × 安静心拍 | CCF · ACF · 偏CCF</div>
        <div onDragOver={e => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files?.[0]); }}
          onClick={() => fileRef.current?.click()}
          style={{ width: 380, maxWidth: '90vw', padding: '48px 24px', border: dragOver ? '2px solid #3b82f6' : '2px dashed #cbd5e1', borderRadius: 12, background: dragOver ? '#eff6ff' : '#fff', textAlign: 'center', cursor: 'pointer', transition: 'all 0.2s' }}>
          <div style={{ fontSize: 32, marginBottom: 8 }}>📂</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#334155' }}>CSVをドロップ or クリック</div>
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8, lineHeight: 1.5 }}>A=患者ID B=経過日 C=離床(min) L=心拍(bpm)</div>
          <input ref={fileRef} type="file" accept=".csv,.tsv,.txt" onChange={e => handleFile(e.target.files?.[0])} style={{ display: 'none' }} />
        </div>
      </div>
    );
  }

  /* === Main view === */
  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', fontFamily: 'Inter, -apple-system, sans-serif', padding: '12px 16px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 17, fontWeight: 700, color: '#0f172a' }}>CCF Dashboard</div>
        {stats && <div style={{ fontSize: 11, color: '#64748b' }}>{stats.n}名 | 系列長 {stats.minLen}–{stats.maxLen} (平均 {stats.meanLen}) 日 | 履歴 {hist.idx + 1}/{hist.stack.length}</div>}
        <div style={{ marginLeft: 'auto' }}>
          <button style={{ ...btn('#e2e8f0', '#334155'), fontSize: 11, padding: '3px 8px' }} onClick={() => fileRef.current?.click()}>CSVを再読込</button>
          <input ref={fileRef} type="file" accept=".csv,.tsv,.txt" onChange={e => handleFile(e.target.files?.[0])} style={{ display: 'none' }} />
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 12, background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, padding: '8px 12px' }}>
        <button style={{ ...btn('#e2e8f0', '#334155'), opacity: canUndo ? 1 : 0.4 }} disabled={!canUndo} onClick={() => { dispatch({ type: 'undo' }); setCcfResult(null); }}>↩ Undo</button>
        <button style={{ ...btn('#e2e8f0', '#334155'), opacity: canRedo ? 1 : 0.4 }} disabled={!canRedo} onClick={() => { dispatch({ type: 'redo' }); setCcfResult(null); }}>Redo ↪</button>
        <div style={{ width: 1, height: 24, background: '#e2e8f0', margin: '0 4px' }} />
        <button style={btn('#f59e0b', '#fff')} onClick={doTrimEdges}>初日・最終日カット</button>
        <button style={btn('#f59e0b', '#fff')} onClick={doInterpY}>Y=0 線形補間</button>
        <span style={{ fontSize: 11, color: '#64748b', marginLeft: 4 }}>体動≥</span>
        <input type="number" min={1} value={kThresh} onChange={e => setKThresh(e.target.value)} style={inputSt} />
        <button style={btn('#f59e0b', '#fff')} onClick={doFilterByK}>Y補間</button>
        <div style={{ width: 1, height: 24, background: '#e2e8f0', margin: '0 4px' }} />
        <span style={{ fontSize: 11, color: '#64748b' }}>FFTローパス:</span>
        <input type="number" min={2} value={cutoff} onChange={e => setCutoff(e.target.value)} style={inputSt} />
        <span style={{ fontSize: 10, color: '#94a3b8' }}>day</span>
        <button style={btn('#3b82f6', '#fff')} onClick={doLowpass}>適用</button>
        <div style={{ width: 1, height: 24, background: '#e2e8f0', margin: '0 4px' }} />
        <span style={{ fontSize: 11, color: '#64748b' }}>メディアン:</span>
        <input type="number" min={2} value={medWin} onChange={e => setMedWin(e.target.value)} style={inputSt} />
        <span style={{ fontSize: 10, color: '#94a3b8' }}>窓</span>
        <button style={btn('#3b82f6', '#fff')} onClick={doMedian}>適用</button>
        <div style={{ width: 1, height: 24, background: '#e2e8f0', margin: '0 4px' }} />
        <span style={{ fontSize: 11, color: '#64748b' }}>最大ラグ:</span>
        <input type="number" min={1} value={lagMax} onChange={e => setLagMax(e.target.value)} style={inputSt} />
        <span style={{ fontSize: 10, color: '#94a3b8' }}>day</span>
        <button style={{ ...btn('#7c3aed', '#fff'), opacity: computing ? 0.5 : 1 }} disabled={computing} onClick={doCCF}>
          {computing ? '計算中…' : 'CCF計算'}
        </button>
      </div>

      {/* Legend */}
      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <span><span style={{ display: 'inline-block', width: 18, height: 3, background: X_CLR, verticalAlign: 'middle', marginRight: 4, borderRadius: 1 }} />X: 離床(min)</span>
        <span><span style={{ display: 'inline-block', width: 18, height: 3, background: Y_CLR, verticalAlign: 'middle', marginRight: 4, borderRadius: 1 }} />Y: 心拍(bpm)</span>
        <span><span style={{ display: 'inline-block', width: 18, height: 3, background: P_CLR, verticalAlign: 'middle', marginRight: 4, borderRadius: 1 }} />偏CCF(AR1残差)</span>
        <span style={{ color: '#94a3b8' }}>※ 各スパークラインは独立正規化</span>
      </div>

      {/* Selected detail */}
      {selectedPatient && <DetailPanel patient={selectedPatient} maxLag={Math.round(Number(lagMax)) || 7} />}

      {/* Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(176px, 1fr))', gap: 4, marginBottom: 16 }}>
        {data.map(p => <Sparkline key={p.id} patient={p} isSelected={selected === p.id} onClick={() => setSelected(s => s === p.id ? null : p.id)} />)}
      </div>

      {/* CCF Results */}
      {ccfResult && <CCFSummary results={ccfResult} maxLag={Math.round(Number(lagMax)) || 7} />}
    </div>
  );
}
