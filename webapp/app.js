// NFL PowerScore - runs the trained game model in the browser from data.js (written by src/nflmodel/export.py)
// the features mirror teamFeatures in src/nflmodel/train.py
const D = window.NFL_DATA;
const M = D.model;
const P = M.params;
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
const fix = (x, d = 3) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const signed = (x, d = 1) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const teams = Object.fromEntries(D.teams.map((t) => [t.team, t]));
const codes = D.teams.map((t) => t.team).sort();
const qbById = {};
Object.entries(D.qbs).forEach(([team, list]) => list.forEach((q) => { if (!qbById[q.id] || q.projected) qbById[q.id] = { ...q, team }; }));
const charts = {};
const seasonLabel = (s) => `${s}`;
const nick = (code) => (D.names[code] || code).split(" ").slice(-1)[0];

// ---------- the model ----------
const clipRest = (r) => Math.min(14, Math.max(4, r ?? 7));
function sideFeatures(t, qbRating, rest) {
  // one team's half of the feature row (same names as teamFeatures in train.py)
  const x = {};
  if (P.epa === "split") Object.assign(x, { offPass: t.offPass, offRush: t.offRush, defPass: t.defPass, defRush: t.defRush });
  else if (P.epa === "neutral") Object.assign(x, { off: t.offN, def: t.defN });
  else Object.assign(x, { off: t.off, def: t.def });
  if (P.points) Object.assign(x, { pointsFor: t.pf, pointsAgainst: t.pa });
  if (P.elo === "raw") x.elo = t.elo;
  if (P.qb === "delta" || P.qb === "both") x.qbDelta = qbRating - t.qbTypical;
  if (P.qb === "raw" || P.qb === "both") x.qb = qbRating;
  x.rest = clipRest(rest);
  return x;
}
const oppName = (n) => "opp" + n[0].toUpperCase() + n.slice(1);
function featureRow(own, opp, home, t) {
  const x = { ...own };
  for (const [k, v] of Object.entries(opp)) x[oppName(k)] = v;
  x.home = home;
  x.homeEdge = home * t.homeEdge;
  x.leaguePoints = t.leaguePoints;
  return x;
}
const linear = (part, x) => part.features.reduce((s, f, j) => s + part.coef[j] * (x[f] - part.mean[j]) / part.scale[j], part.intercept);
const meanOf = (part, f) => part.mean[part.features.indexOf(f)];
function normCdf(z) {
  // Abramowitz-Stegun approximation of the standard normal CDF
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z > 0 ? 1 - p : p;
}
const projectedQb = (code) => (D.qbs[code] || []).find((q) => q.projected) || (D.qbs[code] || [])[0];
function power(code, qb) {
  // PowerScore: expected margin against a league-average team on a neutral field with normal rest
  const t = teams[code];
  const own = sideFeatures(t, (qb || projectedQb(code))?.rating ?? t.qbTypical, 7);
  const avg = Object.fromEntries(Object.keys(own).map((k) => [k, meanOf(M.points, k)]));
  avg.rest = own.rest = 7;
  const versus = featureRow(own, avg, 0, t), against = featureRow(avg, own, 0, t);
  return linear(M.points, versus) - linear(M.points, against);
}
const TIERS = [["Toss-up", 0.5], ["Lean", 0.55], ["Solid", 0.65], ["Strong", 0.75]];
const tierOf = (p) => TIERS.filter(([, cut]) => Math.max(p, 1 - p) >= cut).pop()[0];
const tierIndex = (name) => TIERS.findIndex(([n]) => n === name);
const tierStats = (name) => (D.report.tiers || []).find((t) => t.tier === name);
function predict(o) {
  const h = teams[o.home], a = teams[o.away];
  const hq = qbById[o.homeQb] || projectedQb(o.home), aq = qbById[o.awayQb] || projectedQb(o.away);
  const hs = sideFeatures(h, hq?.rating ?? h.qbTypical, o.homeRest), as = sideFeatures(a, aq?.rating ?? a.qbTypical, o.awayRest);
  const homeFlag = o.neutral ? 0 : 1;
  const hx = featureRow(hs, as, homeFlag, h), ax = featureRow(as, hs, 0, a);
  const homePoints = linear(M.points, hx), awayPoints = linear(M.points, ax);
  const margin = homePoints - awayPoints;
  const pPoints = normCdf(margin / M.sigma);
  const pLogistic = 1 / (1 + Math.exp(-linear(M.win, hx)));
  const homeWin = P.blend * pLogistic + (1 - P.blend) * pPoints;
  // factor contributions vs an average game: normal rest, usual starters, no home field
  const base = Object.fromEntries(M.win.features.map((f, j) => [f, M.win.mean[j]]));
  Object.assign(base, { rest: 7, oppRest: 7, qbDelta: 0, oppQbDelta: 0, home: 0, homeEdge: 0 });
  const contrib = Object.fromEntries(M.win.features.map((f, j) => [f, M.win.coef[j] * (hx[f] - base[f]) / M.win.scale[j]]));
  const groups = {
    "Offense & defense (EPA)": ["off", "def", "offPass", "offRush", "defPass", "defRush"],
    "Scoring form": ["pointsFor", "pointsAgainst"], "Elo": ["elo"], "Quarterbacks": ["qb", "qbDelta"],
    "Rest": ["rest"], "Home field": ["home", "homeEdge"], "Scoring environment": ["leaguePoints"]
  };
  const factors = Object.entries(groups).map(([factor, cols]) => {
    const all = cols.flatMap((c) => [c, oppName(c)]).filter((c) => c in contrib && !["oppHome", "oppHomeEdge", "oppLeaguePoints"].includes(c));
    return { factor, value: all.reduce((s, c) => s + contrib[c], 0), present: all.length > 0 };
  }).filter((f) => f.present);
  return { homeWin, awayWin: 1 - homeWin, homePoints, awayPoints, margin, factors, hq, aq };
}
window.predictGame = predict;

// ---------- helpers ----------
function table(el, columns, rows) {
  el.innerHTML = "<thead><tr>" + columns.map((c) => `<th class="${c.num ? "num" : ""}">${c.label}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map((r, i) => "<tr>" + columns.map((c) => `<td class="${c.num ? "num" : ""}">${c.value(r, i)}</td>`).join("") + "</tr>").join("") + "</tbody>";
}
function chartTheme() {
  Chart.defaults.color = css("--ink-2");
  Chart.defaults.borderColor = css("--line");
  Chart.defaults.font.family = css("--body");
}
// a betting-style line for the home team: "KC −3.5" means KC favored by 3.5
const lineText = (home, away, margin) => (Math.abs(margin) < 0.25 ? "Pick'em" : margin > 0 ? `${home} −${Math.abs(margin).toFixed(1)}` : `${away} −${Math.abs(margin).toFixed(1)}`);
const qbPoints = (r) => (r - avgStarter) * D.dropbacksPerGame;
const avgStarter = (() => { const r = D.teams.map((t) => projectedQb(t.team)?.rating).filter((x) => x !== undefined); return r.reduce((s, x) => s + x, 0) / r.length; })();
const powers = Object.fromEntries(codes.map((c) => [c, power(c)]));
const rankBy = (key, desc = true) => { const s = [...codes].sort((a, b) => (desc ? teams[b][key] - teams[a][key] : teams[a][key] - teams[b][key])); return Object.fromEntries(s.map((c, i) => [c, i + 1])); };
const offRank = rankBy("off"), defRank = rankBy("def", false);
const powerRank = Object.fromEntries([...codes].sort((a, b) => powers[b] - powers[a]).map((c, i) => [c, i + 1]));
const ordinal = (n) => n + (["th", "st", "nd", "rd"][(n % 100 - 20) % 10] || ["th", "st", "nd", "rd"][n % 100] || "th");

// ---------- tabs ----------
const tabs = document.querySelectorAll("nav.tabs button");
function showTab(name) {
  if (![...tabs].some((t) => t.dataset.tab === name)) name = "week";
  tabs.forEach((t) => t.setAttribute("aria-selected", t.dataset.tab === name));
  document.querySelectorAll("section.panel").forEach((p) => (p.hidden = p.id !== name));
  if (name === "predict") runPrediction();
  if (name === "teams" && !charts.trend) renderTeam();
  if (name === "model" && !charts.calib) renderModel();
  if (name === "log" && !$("logTable").innerHTML) renderLog();
  try { history.replaceState(null, "", "#" + name); } catch (e) {}
}
tabs.forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));

// ---------- this week ----------
const weeks = [...new Set(D.schedule.map((g) => g.week))];
const weekLabel = (w, type) => (type && type !== "REG" ? { WC: "Wild card round", DIV: "Divisional round", CON: "Conference championships", SB: "Super Bowl" }[type] : `Week ${w}`);
function gameQb(code, id) {
  // the schedule's listed starter if we know him, otherwise the projected starter
  return id && qbById[id] ? qbById[id] : projectedQb(code);
}
function renderWeek() {
  const week = Number($("weekPick").value);
  const games = D.schedule.filter((g) => g.week === week);
  $("weekTitle").textContent = `${D.season} · ${weekLabel(week, games[0]?.type)}`;
  $("slate").innerHTML = games.map((g, i) => {
    const hq = gameQb(g.home, g.homeQb), aq = gameQb(g.away, g.awayQb);
    const r = predict({ home: g.home, away: g.away, homeQb: hq?.id, awayQb: aq?.id, homeRest: g.homeRest, awayRest: g.awayRest, neutral: g.neutral });
    const tier = tierOf(r.homeWin);
    const day = new Date(g.date + "T12:00:00").toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
    const vegas = g.spread === null ? "–" : lineText(g.home, g.away, g.spread);
    // how far the model's margin is from the betting line, in home-team points
    const edge = g.spread === null ? null : r.margin - g.spread;
    const edgeText = edge === null ? "" : Math.abs(edge) < 1.5 ? `<span class="edge" style="color:var(--muted)">agrees</span>` :
      `<span class="edge" style="color:var(--${edge > 0 ? "home" : "away"})">${edge > 0 ? g.home : g.away} +${Math.abs(edge).toFixed(1)} vs line</span>`;
    return `<button type="button" class="game" data-i="${i}">
      <div class="when"><span>${day}${g.time ? " · " + g.time : ""}${g.neutral ? " · neutral" : ""}</span><span class="tag small t${tierIndex(tier)}">${tier}</span></div>
      <div class="row"><span><span class="code away-ink">${g.away}</span><span class="qb">${esc(aq?.name ?? "")}</span></span><span class="pts">${r.awayPoints.toFixed(0)}</span><span class="pct away-ink">${pct(r.awayWin)}</span></div>
      <div class="row"><span><span class="code home-ink">${g.home}</span><span class="qb">${esc(hq?.name ?? "")}</span></span><span class="pts">${r.homePoints.toFixed(0)}</span><span class="pct home-ink">${pct(r.homeWin)}</span></div>
      <div class="bar" aria-hidden="true"><div style="width:${pct(r.awayWin, 2)};background:var(--away)"></div><div style="width:${pct(r.homeWin, 2)};background:var(--home)"></div></div>
      <div class="lines"><span>Model <b>${lineText(g.home, g.away, r.margin)}</b> · Vegas <b>${vegas}</b></span>${edgeText}</div>
    </button>`;
  }).join("") || `<p class="note">No upcoming games scheduled.</p>`;
  $("slate").querySelectorAll(".game").forEach((b) => b.addEventListener("click", () => openGame(games[Number(b.dataset.i)])));
}
function renderLastWeek() {
  const played = D.log.filter((g) => g[1] === D.dataSeason);
  if (!played.length) { $("lastWeekBox").hidden = true; return; }
  const week = Math.max(...played.map((g) => g[2]));
  const games = played.filter((g) => g[2] === week);
  const decided = games.filter((g) => g[5] !== g[6]);
  const right = decided.filter((g) => (g[9] >= 0.5) === (g[5] > g[6])).length;
  const seasonGames = played.filter((g) => g[5] !== g[6]);
  const seasonRight = seasonGames.filter((g) => (g[9] >= 0.5) === (g[5] > g[6])).length;
  $("lastWeekTitle").textContent = `${D.dataSeason} week ${week} results`;
  $("lastWeekNote").innerHTML = `Model picked <b>${right} of ${decided.length}</b> winners in week ${week} and <b>${seasonRight} of ${seasonGames.length}</b> (${pct(seasonRight / Math.max(seasonGames.length, 1), 1)}) this season. Pregame chances come from a model trained only on earlier seasons.`;
  table($("lastWeekTable"), logColumns(), games.slice().reverse());
}
function openGame(g) {
  const hq = gameQb(g.home, g.homeQb), aq = gameQb(g.away, g.awayQb);
  $("home").value = g.home; $("away").value = g.away;
  fillQbs("home", hq?.id); fillQbs("away", aq?.id);
  setRest("homeRest", g.homeRest); setRest("awayRest", g.awayRest);
  $("neutral").checked = !!g.neutral;
  state.game = g;
  showTab("predict");
}

// ---------- predict ----------
const state = { game: null };
const REST = [[4, "4 · Thursday game"], [5, "5"], [6, "6"], [7, "7 · normal week"], [8, "8"], [9, "9"], [10, "10 · after a Thursday"], [11, "11"], [12, "12"], [13, "13"], [14, "14+ · after a bye"]];
function setRest(id, days) { $(id).value = String(Math.round(clipRest(days))); }
function fillQbs(side, selected) {
  const code = $(side).value;
  const list = D.qbs[code] || [];
  const pick = selected && list.some((q) => q.id === selected) ? selected : projectedQb(code)?.id;
  $(side + "Qb").innerHTML = list.map((q) => `<option value="${esc(q.id)}" ${q.id === pick ? "selected" : ""}>${esc(q.name)}${q.projected ? " · starter" : ""}</option>`).join("");
}
function qbNote(q, code) {
  if (!q) return "";
  const t = teams[code], diff = (q.rating - t.qbTypical) * D.dropbacksPerGame;
  return `${esc(q.name)}: ${signed(q.rating, 3)} EPA/dropback · ${q.dropbacks < 60 ? "little NFL experience, rated near backup level" : `${signed(qbPoints(q.rating))} pts/game vs an average starter`}` +
    (Math.abs(diff) >= 0.5 ? ` · <b>${signed(diff)} pts vs ${code}'s usual starter</b>` : "");
}
function runPrediction() {
  const home = $("home").value, away = $("away").value;
  const o = { home, away, homeQb: $("homeQb").value, awayQb: $("awayQb").value, homeRest: Number($("homeRest").value), awayRest: Number($("awayRest").value), neutral: $("neutral").checked };
  const game = state.game && state.game.home === home && state.game.away === away ? state.game : null;
  const r = predict(o);
  $("homeName").textContent = D.names[home]; $("awayName").textContent = D.names[away];
  $("homePct").textContent = pct(r.homeWin); $("awayPct").textContent = pct(r.awayWin);
  $("homeBar").style.width = pct(r.homeWin, 2); $("awayBar").style.width = pct(r.awayWin, 2);
  $("probbar").setAttribute("aria-label", `${away} ${pct(r.awayWin)}, ${home} ${pct(r.homeWin)}`);
  $("xScore").textContent = `${away} ${r.awayPoints.toFixed(1)} – ${r.homePoints.toFixed(1)} ${home}`;
  $("xLine").textContent = lineText(home, away, r.margin);
  $("xLineVegas").textContent = game && game.spread !== null ? `Vegas: ${lineText(home, away, game.spread)}` : "Pick a game on This week to compare with Vegas";
  $("xTotal").textContent = (r.homePoints + r.awayPoints).toFixed(1);
  $("xTotalVegas").textContent = game && game.total !== null ? `Vegas: ${game.total}` : "";
  $("close").textContent = pct(normCdf((8.5 - r.margin) / M.sigma) - normCdf((-8.5 - r.margin) / M.sigma));
  $("homeQbNote").innerHTML = qbNote(r.hq, home); $("awayQbNote").innerHTML = qbNote(r.aq, away);
  const tier = tierOf(r.homeWin), stats = tierStats(tier), favorite = r.homeWin >= 0.5 ? home : away;
  $("tierBadge").innerHTML = `<span class="tag t${tierIndex(tier)}">${tier}</span> <span>Pick: <b>${D.names[favorite]}</b>${stats ?
    ` · in the backtest, ${tier.toLowerCase()} picks won <b>${pct(stats.all.accuracy, 1)}</b> of the time (${stats.all.games.toLocaleString()} games since ${D.report.bestBySeason[0].season})` : ""}</span>`;
  drawFactors(r, home, away);
  drawMargins(r, home, away);
}
function drawFactors(r, home, away) {
  chartTheme();
  const values = r.factors.map((f) => f.value);
  const colors = values.map((v) => (v >= 0 ? css("--home") : css("--away")));
  const axis = `← favors ${away}     favors ${home} →`;
  if (charts.factors) {
    Object.assign(charts.factors.data.datasets[0], { data: values, backgroundColor: colors });
    charts.factors.options.scales.x.title.text = axis;
    charts.factors.$teams = [home, away];
    return charts.factors.update();
  }
  charts.factors = new Chart($("factorChart"), {
    type: "bar",
    data: { labels: r.factors.map((f) => f.factor), datasets: [{ data: values, backgroundColor: colors, borderRadius: 4, barThickness: 14 }] },
    options: {
      indexAxis: "y", maintainAspectRatio: false, animation: { duration: 250 },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => { const [h, a] = charts.factors.$teams; return `Favors ${c.raw >= 0 ? h : a}: ${signed(c.raw, 3)} log-odds`; } } } },
      scales: { x: { title: { display: true, text: axis } }, y: { grid: { display: false } } }
    }
  });
  charts.factors.$teams = [home, away];
}
const BUCKETS = [[15, 99, "15+"], [8, 14, "8–14"], [4, 7, "4–7"], [1, 3, "1–3"]];
function drawMargins(r, home, away) {
  chartTheme();
  const prob = (lo, hi) => normCdf((hi + 0.5 - r.margin) / M.sigma) - normCdf((lo - 0.5 - r.margin) / M.sigma);
  const rows = [...BUCKETS.map(([lo, hi, label]) => ({ label: `${away} by ${label}`, p: prob(-hi, -lo), side: "away" })),
    ...BUCKETS.slice().reverse().map(([lo, hi, label]) => ({ label: `${home} by ${label}`, p: prob(lo, hi), side: "home" }))];
  const data = rows.map((x) => x.p), colors = rows.map((x) => css(x.side === "home" ? "--home" : "--away")), labels = rows.map((x) => x.label);
  if (charts.margin) {
    Object.assign(charts.margin.data, { labels });
    Object.assign(charts.margin.data.datasets[0], { data, backgroundColor: colors });
    return charts.margin.update();
  }
  charts.margin = new Chart($("marginChart"), {
    type: "bar",
    data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 4 }] },
    options: {
      maintainAspectRatio: false, animation: { duration: 250 },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => pct(c.raw, 1) } } },
      scales: { y: { beginAtZero: true, ticks: { callback: (v) => pct(v) } }, x: { grid: { display: false }, ticks: { autoSkip: false, maxRotation: 50 } } }
    }
  });
}

// ---------- rankings ----------
function sparkline(values) {
  const v = values.filter((x) => x !== null);
  if (v.length < 2) return "";
  const lo = -12, hi = 12, w = 110, h = 28, clamp = (x) => Math.min(hi, Math.max(lo, x));
  const y = (x) => h - 3 - ((clamp(x) - lo) / (hi - lo)) * (h - 6);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * (w - 4) + 2, y(x)]);
  const [ex, ey] = pts[pts.length - 1];
  return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true"><line x1="0" x2="${w}" y1="${y(0).toFixed(1)}" y2="${y(0).toFixed(1)}"/><polyline points="${pts.map((p) => p.map((n) => n.toFixed(1)).join(",")).join(" ")}"/><circle cx="${ex.toFixed(1)}" cy="${ey.toFixed(1)}" r="2.4"/></svg>`;
}
const scoreBar = (value, max) => {
  const w = Math.min(50, Math.abs(value) / max * 50);
  return `<div class="score"><strong>${signed(value)}</strong><span class="track"><i class="${value < 0 ? "neg" : ""}" style="${value < 0 ? `right:50%` : `left:50%`};width:${w.toFixed(1)}%"></i></span></div>`;
};
function renderRankings() {
  const rows = [...codes].sort((a, b) => powers[b] - powers[a]).map((c) => teams[c]);
  const max = Math.max(...Object.values(powers).map(Math.abs));
  const recordLabel = D.dataSeason === D.season ? "" : ` in ${D.dataSeason}`;
  table($("teamTable"), [
    { label: "Rank", value: (t, i) => `<span class="rkn">${i + 1}</span>` },
    { label: "Team", value: (t) => `<div class="who"><b>${esc(t.name)}</b><span>${t.record.join("-").replace(/-0$/, "")}${recordLabel}</span></div>` },
    { label: "PowerScore (pts)", value: (t) => scoreBar(powers[t.team], max) },
    { label: "Offense", num: true, value: (t) => `<span class="rankcell">${signed(t.off, 3)}<small>${ordinal(offRank[t.team])}</small></span>` },
    { label: "Defense", num: true, value: (t) => `<span class="rankcell">${signed(t.def, 3)}<small>${ordinal(defRank[t.team])}</small></span>` },
    { label: "Starting QB", value: (t) => { const q = projectedQb(t.team); return q ? `<div class="who"><span style="color:var(--ink);font-size:14px">${esc(q.name)}</span><span>${signed(qbPoints(q.rating))} pts vs avg starter</span></div>` : "–"; } },
    { label: "Elo", num: true, value: (t) => fix(t.elo, 0) },
    { label: "Trend", value: (t) => sparkline((D.trends[t.team] || []).map((x) => x[3])) }
  ], rows);
  $("teamTable").querySelectorAll("tbody tr").forEach((tr, i) => {
    tr.classList.add("link");
    tr.title = `Open ${rows[i].name}`;
    tr.addEventListener("click", () => { $("teamSelect").value = rows[i].team; showTab("teams"); renderTeam(); });
  });
  const qbs = Object.values(qbById).filter((q) => q.dropbacks >= 100 || q.projected).sort((a, b) => b.rating - a.rating);
  table($("qbTable"), [
    { label: "Rank", value: (q, i) => `<span class="rkn">${i + 1}</span>` },
    { label: "Quarterback", value: (q) => `<div class="who"><b>${esc(q.name)}</b><span>${q.team}${q.projected ? " · starter" : ""}</span></div>` },
    { label: "EPA / dropback", num: true, value: (q) => signed(q.rating, 3) },
    { label: "Pts / game vs avg starter", num: true, value: (q) => `<span class="pill ${qbPoints(q.rating) > 1 ? "good" : qbPoints(q.rating) < -1 ? "bad" : "flat"}">${signed(qbPoints(q.rating))}</span>` },
    { label: "CPOE", num: true, value: (q) => signed(q.cpoe, 1) },
    { label: "Starts (2 seasons)", num: true, value: (q) => q.starts }
  ], qbs);
  setRankView("teams");
}
function setRankView(view) {
  $("segTeams").setAttribute("aria-pressed", view === "teams");
  $("segQbs").setAttribute("aria-pressed", view === "qbs");
  $("teamRank").hidden = view !== "teams";
  $("qbRank").hidden = view !== "qbs";
  $("rankTitle").textContent = view === "teams" ? "Team power rankings" : "Quarterback ratings";
  $("rankNote").textContent = view === "teams"
    ? "PowerScore is how many points better than an average team each team would be on a neutral field with its projected starting QB. Offense and defense are EPA per play (lower is better on defense). Click a team for details."
    : "Recency-weighted EPA per dropback (sacks and scrambles included), shrunk toward backup level for QBs with few dropbacks. Points per game assume about 38 dropbacks.";
}
$("segTeams").addEventListener("click", () => setRankView("teams"));
$("segQbs").addEventListener("click", () => setRankView("qbs"));

// ---------- teams ----------
function renderTeam() {
  const code = $("teamSelect").value, t = teams[code];
  $("trendTitle").textContent = D.names[code];
  const tile = (k, v, s) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
  $("teamTiles").innerHTML = [
    tile("PowerScore", signed(powers[code]), `${ordinal(powerRank[code])} in the NFL`),
    tile("Offense EPA/play", signed(t.off, 3), `${ordinal(offRank[code])} · pass ${signed(t.offPass, 2)}, run ${signed(t.offRush, 2)}`),
    tile("Defense EPA/play", signed(t.def, 3), `${ordinal(defRank[code])} · pass ${signed(t.defPass, 2)}, run ${signed(t.defRush, 2)}`),
    tile("Scoring form", `${t.pf.toFixed(1)}–${t.pa.toFixed(1)}`, "Recency-weighted points for–against"),
    tile("Elo", fix(t.elo, 0), `${D.dataSeason} record ${t.record.join("-").replace(/-0$/, "")}`)
  ].join("");
  const trend = D.trends[code] || [];
  chartTheme();
  charts.trend?.destroy();
  charts.trend = new Chart($("trendChart"), {
    type: "line",
    data: { labels: trend.map((x) => `${x[1]} wk ${x[2]}`), datasets: [
      { label: "PowerScore", data: trend.map((x) => x[3]), borderColor: css("--turf"), backgroundColor: "transparent", borderWidth: 2, pointRadius: 2, pointHoverRadius: 5, tension: 0.25 }
    ] },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false }, tooltip: { callbacks: { title: (c) => `${trend[c[0].dataIndex][0]} · ${c[0].label}`, label: (c) => `PowerScore ${signed(c.raw)} pts · Elo ${fix(trend[c.dataIndex][4], 0)}` } } },
      scales: { x: { ticks: { maxTicksLimit: 8 }, grid: { display: false } }, y: { title: { display: true, text: "Points vs average team" } } }
    }
  });
  table($("qbTeamTable"), [
    { label: "QB", value: (q) => `${esc(q.name)}${q.projected ? ' <span class="pill good">starter</span>' : ""}` },
    { label: "Starts", num: true, value: (q) => q.starts },
    { label: "This season", num: true, value: (q) => q.seasonStarts },
    { label: "EPA/db", num: true, value: (q) => signed(q.rating, 3) },
    { label: "vs usual", num: true, value: (q) => signed((q.rating - t.qbTypical) * D.dropbacksPerGame) }
  ], D.qbs[code] || []);
  table($("recentTable"), [
    { label: "Date", value: (g) => g[0] },
    { label: "Opp", value: (g) => (g[4] ? "vs " : g[3] ? "vs " : "@ ") + g[2] + (g[4] ? " (n)" : "") },
    { label: "Result", value: (g) => { const r = g[5] > g[6] ? "W" : g[5] < g[6] ? "L" : "T"; return `<span class="pill ${r === "W" ? "good" : r === "L" ? "bad" : "flat"}">${r}</span> ${g[5]}-${g[6]}`; } },
    { label: "Line", num: true, value: (g) => (g[8] === null ? "–" : g[8] > 0 ? `−${g[8]}` : g[8] < 0 ? `+${-g[8]}` : "PK") },
    { label: "QB", value: (g) => esc(g[7] ?? "–") }
  ], D.recent[code] || []);
}
$("teamSelect").addEventListener("change", renderTeam);

// ---------- model report ----------
function describe(params) {
  const p = { ...D.report.defaults, ...params }, parts = [];
  if (p.halflife !== D.report.defaults.halflife) parts.push(`ratings half-life ${p.halflife} games`);
  if (p.epa === "neutral") parts.push("EPA without garbage time");
  if (p.epa === "split") parts.push("pass and run EPA split");
  if (p.qb === "none") parts.push("no QB features");
  if (p.qb === "raw") parts.push("QB rating instead of QB change");
  if (p.qb === "both") parts.push("QB rating + QB change");
  if (p.elo === "none") parts.push("no Elo");
  if (!p.points) parts.push("no scoring form");
  if (p.home === "fixed") parts.push("fixed home field");
  if (p.alpha !== 1) parts.push("stronger regularization");
  if (p.blend === 1) parts.push("logistic win model only");
  if (p.blend === 0) parts.push("points model win chance only");
  return parts.length ? parts.join(", ") : "Base model";
}
function renderModel() {
  const r = D.report, bestKey = JSON.stringify(r.best, Object.keys(r.best).sort());
  const key = (p) => JSON.stringify(p, Object.keys(p).sort());
  const best = r.candidates.find((c) => key(c.params) === bestKey);
  const vegas = r.baselines.find((b) => b.model.startsWith("Betting"));
  const elo = r.baselines.find((b) => b.model === "Elo only");
  const first = r.bestBySeason[0].season, last = r.bestBySeason[r.bestBySeason.length - 1].season;
  $("modelNote").textContent = `Walk-forward backtest: every season from ${first} to ${last} is predicted by a model trained only on earlier seasons (${best.games.toLocaleString()} games). Log loss scores the probabilities (lower is better).`;
  const tile = (k, v, s) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
  $("headline").innerHTML = [
    tile("Winner picked", pct(best.accuracy, 1), `Elo ${pct(elo.accuracy, 1)} · Vegas ${pct(vegas.accuracy, 1)}`),
    tile("Log loss", best.logLoss.toFixed(4), `Elo ${elo.logLoss.toFixed(4)} · Vegas ${vegas.logLoss.toFixed(4)}`),
    tile("Margin off by (avg)", best.marginMAE.toFixed(1) + " pts", `Vegas spread ${vegas.marginMAE.toFixed(1)} pts`),
    tile("Against the spread", pct(best.ats, 1), "Picks vs the Vegas line (50% = no edge)")
  ].join("");
  const rows = [...r.baselines.map((b) => ({ ...b, name: b.model, tag: "baseline" })),
    ...r.candidates.map((c) => ({ ...c, name: describe(c.params), tag: key(c.params) === bestKey ? "chosen" : "" }))];
  table($("compareTable"), [
    { label: "Version", value: (m) => `${esc(m.name)} ${m.tag === "chosen" ? '<span class="pill good">chosen</span>' : m.tag ? '<span class="pill flat">baseline</span>' : ""}` },
    { label: "Log loss", num: true, value: (m) => m.logLoss.toFixed(4) },
    { label: "Winner picked", num: true, value: (m) => pct(m.accuracy, 1) },
    { label: "Brier", num: true, value: (m) => m.brier.toFixed(4) },
    { label: "Margin off by", num: true, value: (m) => (m.marginMAE ? m.marginMAE.toFixed(2) : "–") },
    { label: "Better than Elo by", num: true, value: (m) => { const d = elo.logLoss - m.logLoss; return `<span class="pill ${d > 0.0001 ? "good" : d < -0.0001 ? "bad" : "flat"}">${signed(d, 4)}</span>`; } }
  ], rows);
  table($("tierTable"), [
    { label: "Tier", value: (t) => `<span class="tag t${tierIndex(t.tier)}">${t.tier}</span>` },
    { label: "Favorite's win chance", value: (t) => { const i = tierIndex(t.tier), next = TIERS[i + 1]; return next ? `${pct(t.minConfidence)} – ${pct(next[1])}` : `${pct(t.minConfidence)}+`; } },
    { label: `${first}–${last}: accuracy`, num: true, value: (t) => `<b>${pct(t.all.accuracy, 1)}</b>` },
    { label: "Share of games", num: true, value: (t) => pct(t.all.share) },
    { label: `${D.completeSeason}: accuracy`, num: true, value: (t) => (t.latest.accuracy === null ? "–" : pct(t.latest.accuracy, 1)) },
    { label: "Games", num: true, value: (t) => t.latest.games }
  ], [...(r.tiers || [])].reverse());
  table($("seasonTable"), [
    { label: "Season", value: (s) => s.season + (s.season === D.season && D.season !== D.completeSeason ? " (so far)" : "") },
    { label: "Games", num: true, value: (s) => s.games },
    { label: "Log loss", num: true, value: (s) => s.logLoss.toFixed(4) },
    { label: "Winner picked", num: true, value: (s) => pct(s.accuracy, 1) },
    { label: "Margin off by", num: true, value: (s) => s.marginMAE.toFixed(1) }
  ], r.bestBySeason.slice().reverse());
  const bins = r.calibration || [];
  $("calibNote").textContent = `Games from ${first}–${last} grouped by predicted home win chance. On the dashed line, "70%" means the home team really won 70% of the time.`;
  chartTheme();
  charts.calib = new Chart($("calibChart"), {
    type: "scatter",
    data: { datasets: [
      { label: "Perfect calibration", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], type: "line", borderColor: css("--muted"), borderDash: [4, 4], borderWidth: 1, pointRadius: 0 },
      { label: "Model", data: bins, borderColor: css("--turf"), backgroundColor: css("--turf"), pointRadius: 6, pointHoverRadius: 8, showLine: true, borderWidth: 2 }
    ] },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" }, tooltip: { filter: (c) => c.datasetIndex === 1, callbacks: { label: (c) => `Predicted ${pct(c.raw.x)} → won ${pct(c.raw.y)} (${c.raw.n} games)` } } },
      scales: { x: { min: 0, max: 1, title: { display: true, text: "Predicted home win chance" }, ticks: { callback: (v) => pct(v) } },
                y: { min: 0, max: 1, title: { display: true, text: "Home team actually won" }, ticks: { callback: (v) => pct(v) } } }
    }
  });
  const how = [
    ["Points", `each team's points are a ridge regression on both teams' pregame features. The margin is treated as normal with a standard deviation of ${M.sigma.toFixed(1)} points, which gives the win chance, the line and the margin chart.`],
    ["Win chance", `a logistic regression on the same features, averaged ${Math.round(P.blend * 100)}/${Math.round((1 - P.blend) * 100)} with the points model's win chance.`],
    ["Team strength", `offense and defense EPA per play from nflverse play-by-play, recency-weighted with a ${P.halflife}-game half-life and carried across seasons (half the evidence is kept over the offseason).${P.epa === "split" ? " Passing and rushing are rated separately." : P.epa === "neutral" ? " Garbage-time plays are left out." : ""}`],
    ["Quarterbacks", "each QB's recency-weighted EPA per dropback on any team, shrunk toward backup level. The model uses the starter's rating and how he compares with the team's usual starter, so a backup starting moves the line right away, as it does in Vegas."],
    ["Also", `a FiveThirtyEight-style Elo rating${P.points ? ", recent points for and against" : ""}, days of rest (Thursday games and byes), home field${P.home === "trend" ? " scaled by the league's recent home edge" : ""}, and the league scoring level.`],
    ["Not included", "injuries other than the QB, weather and coaching changes. The betting line prices all of these, which is why Vegas still beats the model."]
  ];
  $("howList").innerHTML = how.map(([k, v]) => `<li><b>${k}:</b> ${v}</li>`).join("");
}

// ---------- game log ----------
// log rows: [date, season, week, home, away, homeScore, awayScore, homePoints, awayPoints, pWin, spread, neutral, homeQb, awayQb]
function logColumns() {
  return [
    { label: "Wk", value: (g) => g[2] },
    { label: "Game", value: (g) => `${g[4]} ${g[11] ? "vs" : "@"} ${g[3]}` },
    { label: "Home win %", num: true, value: (g) => pct(g[9]) },
    { label: "Model", num: true, value: (g) => lineText(g[3], g[4], g[7] - g[8]) },
    { label: "Vegas", num: true, value: (g) => (g[10] === null ? "–" : lineText(g[3], g[4], g[10])) },
    { label: "Final", num: true, value: (g) => `${g[4]} ${g[6]}–${g[5]} ${g[3]}` },
    { label: "Tier", value: (g) => { const t = tierOf(g[9]); return `<span class="tag small t${tierIndex(t)}">${t}</span>`; } },
    { label: "Pick", value: (g) => (g[5] === g[6] ? '<span class="pill flat">tie</span>' : (g[9] >= 0.5) === (g[5] > g[6]) ? `<span class="pill good">✓ ${g[9] >= 0.5 ? g[3] : g[4]}</span>` : `<span class="pill bad">✗ ${g[9] >= 0.5 ? g[3] : g[4]}</span>`) },
    { label: "vs line", value: (g) => { if (g[10] === null) return "–"; const m = g[7] - g[8], a = g[5] - g[6]; if (a === g[10]) return '<span class="pill flat">push</span>';
      return (m > g[10]) === (a > g[10]) ? '<span class="pill good">✓</span>' : '<span class="pill bad">✗</span>'; } }
  ];
}
function renderLog() {
  const season = Number($("logSeason").value), team = $("logTeam").value, tierFilter = $("logTier").value;
  const seasonGames = D.log.filter((g) => g[1] === season && (!team || g[3] === team || g[4] === team));
  const games = seasonGames.filter((g) => !tierFilter || tierOf(g[9]) === tierFilter).slice().reverse();
  const record = (list) => { let n = 0, c = 0; list.forEach((g) => { if (g[5] !== g[6]) { n++; if ((g[9] >= 0.5) === (g[5] > g[6])) c++; } }); return [c, n]; };
  const [correct, decided] = record(games);
  const byTier = TIERS.slice().reverse().map(([name]) => { const [c, n] = record(seasonGames.filter((g) => tierOf(g[9]) === name)); return n ? `${name} ${pct(c / n, 1)} (${n})` : null; }).filter(Boolean).join(" · ");
  $("logSummary").innerHTML = `${games.length} games · winner picked in ${correct} of ${decided} (<b>${pct(correct / Math.max(decided, 1), 1)}</b>). By tier: ${byTier}. Every game was predicted by a model trained only on seasons before ${season}.`;
  table($("logTable"), logColumns(), games);
}
["logSeason", "logTeam", "logTier"].forEach((id) => $(id).addEventListener("change", renderLog));

// ---------- start ----------
$("asOf").textContent = `${D.season} season · through ${D.asOf}`;
const teamOptions = (selected) => codes.map((c) => `<option value="${c}" ${c === selected ? "selected" : ""}>${esc(D.names[c])}</option>`).join("");
const byPower = [...codes].sort((a, b) => powers[b] - powers[a]);
$("home").innerHTML = teamOptions(byPower[0]);
$("away").innerHTML = teamOptions(byPower[1]);
$("teamSelect").innerHTML = teamOptions(byPower[0]);
$("logTeam").innerHTML += teamOptions(null);
$("logSeason").innerHTML = [...new Set(D.log.map((g) => g[1]))].sort((a, b) => b - a).map((s) => `<option>${s}</option>`).join("");
for (const id of ["homeRest", "awayRest"]) $(id).innerHTML = REST.map(([v, l]) => `<option value="${v}" ${v === 7 ? "selected" : ""}>${l}</option>`).join("");
$("weekPick").innerHTML = weeks.map((w) => `<option value="${w}">${weekLabel(w, D.schedule.find((g) => g.week === w).type)}</option>`).join("");
$("weekPick").hidden = weeks.length < 2;
$("weekPick").addEventListener("change", renderWeek);
$("sigma").textContent = M.sigma.toFixed(1);
for (const side of ["home", "away"]) {
  fillQbs(side);
  $(side).addEventListener("change", () => { fillQbs(side); runPrediction(); });
  for (const id of [side + "Qb", side + "Rest"]) $(id).addEventListener("change", runPrediction);
}
$("neutral").addEventListener("change", runPrediction);
renderWeek();
renderLastWeek();
renderRankings();
showTab((location.hash || "#week").slice(1));
