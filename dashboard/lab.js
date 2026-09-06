import { applyMode, bindModeToggle } from './app.js';

const $ = id => document.getElementById(id);
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => Number.isFinite(value) ? Math.round(value).toLocaleString() : '—';
const percent = value => Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : '—';
const checkpoints = ['untrained', 'after_a', 'after_b', 'after_return_a'];
const checkpointLabel = value => ({untrained:'Before learning',after_a:'After A',after_b:'After B',after_return_a:'Return to A',posthoc_reference:'Diagnostic reference'}[value] || value);
const conditionLabel = value => ({continued:'Continuing learner',frozen_after_a:'Frozen after A',untrained:'Untrained',shortest_path:'Shortest-path reference',random_actions:'Random actions'}[value] || value);
const colors = {continued:'#a6e5c1', frozen_after_a:'#e7b985', untrained:'#828b93',raw:'#98b8d1',unique:'#a6e5c1',decay:'#e7b985',never:'#626d69'};
let adaptation = null, provenance = null, diagnostics = null, competence = null, currentTrajectory = null, frame = 0, timer = null;
let competenceCondition = 'fixed_1', competenceTrajectory = null, competenceFrame = 0, competenceTimer = null;
const supportColors = {fixed_1:'#a6e5c1',fixed_16:'#98b8d1',stream:'#e7b985'};
const policyLabel = value => ({learner:'Learned policy',random_actions:'Random actions',shortest_path:'Shortest path'}[value] || value);
const panelLabel = value => value === 'heldout' ? 'Fresh maps' : 'Probe maps';
const decimal = value => Number.isFinite(value) ? value.toFixed(3) : '—';
const stepLabel = value => value >= 1000 ? `${value / 1000}k` : number(value);
const competenceMode = () => $('competence-epsilon').value;
const supportLabel = id => competence?.protocol?.conditions?.find(c=>c.id===id)?.label || id;
const latestCompetenceStep = () => Math.max(0,...(competence?.aggregate || []).map(r=>r.checkpoint).filter(Number.isFinite));
let loadInProgress = false;
const replayRows = () => [...(adaptation?.trajectories || []),...(diagnostics?.trajectories || [])];

applyMode(); bindModeToggle();
function stopPlayback() { if (timer) clearInterval(timer); timer = null; $('replay-play').textContent = '▶ Play'; $('replay-play').setAttribute('aria-label','Play trajectory'); }
function selectOptions(id, options, preferred) {
  const el = $(id), previous = el.value;
  el.innerHTML = options.map(([value,label]) => `<option value="${escape(value)}">${escape(label)}</option>`).join('');
  const values = options.map(([value]) => String(value));
  el.value = values.includes(previous) ? previous : values.includes(preferred) ? preferred : values[0] || '';
}
function listItems(id, values) { $(id).innerHTML = (values || []).map(v => `<li>${escape(v)}</li>`).join(''); }
const tracks = ['competence','adaptation','provenance'];
function activateTab(name) {
  for (const track of tracks) {
    const active = track === name;
    $(`tab-${track}`).setAttribute('aria-selected',String(active));
    $(`tab-${track}`).tabIndex = active ? 0 : -1;
    $(`${track}-panel`).hidden = !active;
  }
  stopPlayback();
  stopCompetencePlayback();
  history.replaceState(null,'',`#${name}`);
}
for (const [index,name] of tracks.entries()) {
  $(`tab-${name}`).addEventListener('click',() => activateTab(name));
  $(`tab-${name}`).addEventListener('keydown',event => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === 'ArrowLeft' ? -1 : 1;
    const other = event.key === 'Home' ? tracks[0] : event.key === 'End' ? tracks.at(-1) : tracks[(index + direction + tracks.length) % tracks.length];
    activateTab(other); $(`tab-${other}`).focus();
  });
}
if (tracks.includes(location.hash.slice(1))) activateTab(location.hash.slice(1));

function renderCompetence() {
  stopCompetencePlayback();
  const ready = Boolean(competence?.aggregate?.length);
  $('competence-empty').hidden = ready;
  $('competence-content').hidden = !ready;
  if (!ready) return;
  const protocol = competence.protocol || {}, run = competence.run || {}, conditions = protocol.conditions || [];
  if (!conditions.some(c=>c.id===competenceCondition)) competenceCondition = conditions[0]?.id || '';
  $('competence-study-label').textContent = `${run.id || protocol.id || 'Saved study'} · ${(run.status || 'exploratory').replaceAll('_',' ')}`;
  const interpretations = {
    fresh_competence_gate_met:`Fresh-map competence gate met. At least one condition reaches ${percent(protocol.gates?.heldout)} greedy success in every learner seed and exceeds the random reference. This warrants independent replication before interpreting learning across worlds.`,
    repeated_support_only:`Repeated tasks learned; the fresh-map gate remains unmet. At least one fixed-support condition reaches ${percent(protocol.gates?.memorization)} greedy probe success in every seed. This establishes learning the selected tasks, not useful transfer to fresh maps.`,
    known_task_competence_not_established:`Known-task competence was not established. Neither fixed-support condition reached ${percent(protocol.gates?.memorization)} greedy probe success in every seed. The next diagnosis remains learning known tasks.`,
    smoke_or_protocol_deviation_not_gate_evidence:'Smoke or protocol-deviation run. These measurements check execution; they cannot pass the declared research gates. Inspect the protocol and scope before interpreting scores.',
    incomplete_not_gate_evidence:'Incomplete study. The admission budget ended before the declared comparison finished. Saved measurements show partial coverage and cannot pass a competence gate.',
  };
  $('competence-gate-note').textContent = interpretations[run.interpretation] || 'Gate interpretation is unavailable. Read the saved protocol and per-seed results before drawing a learning conclusion.';
  $('competence-gate-note').classList.toggle('passed',run.interpretation==='fresh_competence_gate_met');
  const cards = [
    [number(protocol.seeds?.length),'Training seeds','Independent initializations per condition'],
    [number(latestCompetenceStep()),'Latest saved checkpoint','Training transitions within a run'],
    [number(protocol.heldout_seeds?.length),'Fresh evaluation maps','Shared across conditions and checkpoints'],
    [number(run.train_steps),'Total training transitions',`${number(run.wall_seconds)} seconds elapsed`],
  ];
  $('competence-stats').innerHTML = cards.map(([value,label,detail])=>`<div class="stat-card"><span>${escape(label)}</span><strong>${escape(value)}</strong><small>${escape(detail)}</small></div>`).join('');
  const hypotheses = Array.isArray(protocol.hypothesis) ? protocol.hypothesis : [protocol.hypothesis];
  $('competence-hypotheses').innerHTML = `<p>${escape(protocol.question || '')}</p><ul>${hypotheses.filter(Boolean).map(h=>`<li>${escape(h)}</li>`).join('')}</ul>`;
  $('competence-gates').textContent = `Greedy competence targets, required in each of the ${number(protocol.seeds?.length)} learner seeds: at least ${percent(protocol.gates?.memorization)} on fixed training tasks; at least ${percent(protocol.gates?.heldout)} on fresh maps and above the random reference. Streaming probes are scheduled once initially, so their score is not a fixed-support memorization test. The ε = 0.1 view is a separate diagnostic.`;
  const gateStep=protocol.budget?.steps_per_condition_seed ?? Math.max(...(protocol.checkpoints || []));
  const gateStatus=(panel,condition)=>competence.gates?.eligible!==true?'Not eligible':competence.gates[panel]?.[condition]===true?'Met':competence.gates[panel]?.[condition]===false?'Not met':'Not measured';
  $('competence-gate-results').innerHTML=conditions.map(c=>{
    const scores=panel=>(protocol.seeds || []).map(seed=>{
      const row=competence.seed_results?.find(r=>r.condition===c.id && r.seed===seed && r.panel===panel && r.mode==='greedy' && r.checkpoint===gateStep);
      return `seed ${seed}: ${percent(row?.success_rate)}`;
    }).join(' · ');
    return `<div class="gate-row"><strong>${escape(c.label)}</strong><p>Final greedy probes · ${escape(scores('probe'))}</p><p>Final greedy fresh maps · ${escape(scores('heldout'))}</p><small>${c.id==='stream'?'Repeated-support gate: not applicable':`Repeated-support gate: ${gateStatus('repeated_support',c.id)}`} · Fresh-map gate: ${gateStatus('heldout',c.id)}</small></div>`;
  }).join('');
  $('competence-method').textContent = `One movement rule, ${number(conditions.length)} training-support conditions, ${number(protocol.seeds?.length)} training seeds each. Evaluation never updates the learner. Curves use saved checkpoints; replays show recorded actions, not a new simulation.`;
  listItems('competence-limitations',run.limitations);
  const artifacts = competence.artifacts || {};
  const links = [['protocol_document','Study design'],['protocol','Saved machine protocol'],['report','Measured findings'],['evaluations','Raw evaluations'],['training','Training log'],['manifest','Artifact checksums']].flatMap(([key,label])=>{
    const path=artifacts[key];
    // Exports link to repository-relative evidence only.
    return typeof path==='string' && /^(docs|experiments)\/[a-zA-Z0-9_./-]+$/.test(path) && !path.split('/').includes('..')
      ? [`<a href="../${escape(path)}">${label} ↗</a>`] : [];
  });
  $('competence-evidence-links').innerHTML = '<a href="data/competence.json" download>Download displayed data ↓</a>'+links.join('');
  renderCompetenceComparison();
}

function renderCompetenceComparison() {
  if (!competence?.aggregate?.length) return;
  const mode=competenceMode(), latest=latestCompetenceStep(), conditions=competence.protocol.conditions || [];
  const result=(condition,panel)=>competence.aggregate.find(r=>r.condition===condition && r.panel===panel && r.mode===mode && r.checkpoint===latest);
  $('competence-support-cards').innerHTML=conditions.map(c=>`<button class="competence-support-card" data-support="${escape(c.id)}" aria-pressed="${c.id===competenceCondition}" style="--support-color:${supportColors[c.id] || '#a6e5c1'}"><strong>${escape(c.label)}</strong><small>${escape(c.description)}</small><div class="support-values"><span>${c.id==='stream'?'Initial probes':'Training tasks'}<b>${percent(result(c.id,'probe')?.success_rate)}</b></span><span>Fresh maps<b>${percent(result(c.id,'heldout')?.success_rate)}</b></span></div><small>At ${number(latest)} training steps · select to inspect</small></button>`).join('');
  for (const button of $('competence-support-cards').querySelectorAll('button')) button.addEventListener('click',()=>{
    competenceCondition=button.dataset.support;
    renderCompetenceComparison();
    // Keep keyboard focus after rebuilding the selection cards.
    $('competence-support-cards').querySelector(`[data-support="${competenceCondition}"]`)?.focus();
  });
  renderCompetenceChart('probe','competence-support-chart');
  renderCompetenceChart('heldout','competence-fresh-chart');
  $('competence-support-title').textContent='Can it solve familiar tasks?';
  $('competence-support-caption').textContent='Fixed 1 and fixed 16 are the complete training task sets: walls, start, and pellet. Streaming schedules the same 16 probes once initially, then samples with replacement from a larger task range. Its probe score is not a repeated-support memorization test. Early checkpoints may precede some initial probe encounters; at step zero, none have been encountered.';
  $('competence-fresh-caption').textContent=`The same ${number(competence.protocol.heldout_seeds?.length)} fresh maps are evaluated at each checkpoint. Shading in both charts is the minimum–maximum across training seeds, not a confidence interval. Dashed line: ${percent(competence.protocol.gates?.heldout)} greedy target.`;
  $('competence-curve-legend').innerHTML=conditions.map(c=>`<span><i style="background:${supportColors[c.id] || '#a6e5c1'}"></i>${escape(c.label)}</span>`).join('');
  $('competence-mode-note').textContent = mode==='greedy' ? 'Greedy evaluation chooses the highest-valued action. Every chart, support card, learner reference bar, and learned replay uses this action mode.' : 'Exploring evaluation takes a uniformly random action with probability 0.1. This checks whether occasional random moves help escape a poor greedy policy; it is not the greedy competence gate. Random and planner references keep their own unchanged policies.';
  const rows=competence.aggregate.filter(r=>r.mode===mode).slice().sort((a,b)=>a.checkpoint-b.checkpoint || a.condition.localeCompare(b.condition) || a.panel.localeCompare(b.panel));
  $('competence-table').innerHTML='<thead><tr><th>Training steps</th><th>Support</th><th>Map panel</th><th>Success</th><th>Seed range</th><th>Episodes</th><th>Mean steps</th><th>No-op rate</th><th>Optimal actions</th><th>Winnable states</th><th>Mean action regret</th><th>Mean |Q error|</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${number(r.checkpoint)}</td><td>${escape(supportLabel(r.condition))}</td><td>${panelLabel(r.panel)}</td><td>${percent(r.success_rate)}</td><td>${percent(r.seed_success_min)}–${percent(r.seed_success_max)}</td><td>${number(r.episodes)}</td><td>${Number.isFinite(r.mean_steps)?r.mean_steps.toFixed(1):'—'}</td><td>${percent(r.noop_rate)}</td><td>${percent(r.optimal_action_rate)}</td><td>${number(r.winnable_steps)}</td><td>${decimal(r.mean_action_regret)}</td><td>${decimal(r.mean_abs_q_error)}</td></tr>`).join('')+'</tbody>';
  selectCompetenceTrajectory();
}

function renderCompetenceChart(panel,id) {
  const w=500,h=275,pad={l:43,r:17,t:18,b:51};
  const checkpoints=(competence.protocol.checkpoints || []).filter(Number.isFinite).slice().sort((a,b)=>a-b);
  const maximum=Math.max(1,...checkpoints);
  const x=step=>pad.l+step/maximum*(w-pad.l-pad.r),y=value=>pad.t+(1-value)*(h-pad.t-pad.b);
  let svg=`<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg"><title>${panelLabel(panel)} success by training step; ${escape(competenceMode())}; mean and per-seed range</title>`;
  for(let i=0;i<=4;i++) { const value=i/4;svg+=`<line x1="${pad.l}" y1="${y(value)}" x2="${w-pad.r}" y2="${y(value)}" stroke="#26302a"/><text x="${pad.l-9}" y="${y(value)+4}" text-anchor="end" fill="#8a978f" font-size="10">${100*value}%</text>`; }
  for(let i=0;i<=3;i++) { const value=maximum*i/3;svg+=`<text x="${x(value)}" y="${h-28}" text-anchor="${i===0?'start':i===3?'end':'middle'}" fill="#8a978f" font-size="10">${escape(stepLabel(value))}</text>`; }
  svg+=`<text x="${w/2}" y="${h-8}" text-anchor="middle" fill="#8a978f" font-size="10">Training transitions per run · linear scale</text>`;
  const gate=panel==='heldout'?competence.protocol.gates?.heldout:null;
  if(Number.isFinite(gate)) svg+=`<line x1="${pad.l}" y1="${y(gate)}" x2="${w-pad.r}" y2="${y(gate)}" stroke="#727e75" stroke-dasharray="3 5"/><text x="${w-pad.r-3}" y="${y(gate)-5}" text-anchor="end" fill="#8a978f" font-size="9">${percent(gate)} greedy gate</text>`;
  for(const condition of competence.protocol.conditions || []) {
    const color=supportColors[condition.id] || '#a6e5c1';
    const values=checkpoints.map(step=>competence.aggregate.find(r=>r.condition===condition.id && r.panel===panel && r.mode===competenceMode() && r.checkpoint===step));
    // Draw each contiguous measured segment separately: missing observations are gaps.
    const segments=[];let segment=[];
    values.forEach((r,i)=>{if(Number.isFinite(r?.success_rate))segment.push([checkpoints[i],r]);else if(segment.length){segments.push(segment);segment=[];}});
    if(segment.length)segments.push(segment);
    for(const points of segments) {
      const hasRange=points.every(([,r])=>Number.isFinite(r.seed_success_min)&&Number.isFinite(r.seed_success_max));
      if(hasRange) svg+=`<polygon points="${[...points.map(([step,r])=>`${x(step)},${y(r.seed_success_max)}`),...points.slice().reverse().map(([step,r])=>`${x(step)},${y(r.seed_success_min)}`)].join(' ')}" fill="${color}" opacity=".07"/>`;
      svg+=`<path d="${points.map(([step,r],i)=>`${i?'L':'M'}${x(step)},${y(r.success_rate)}`).join(' ')}" stroke="${color}" stroke-width="${condition.id===competenceCondition?2.5:1.6}" fill="none"/>`;
      for(const [step,r] of points) svg+=`<circle cx="${x(step)}" cy="${y(r.success_rate)}" r="3.5" fill="${color}"><title>${escape(condition.label)}, ${number(step)} steps: ${percent(r.success_rate)}; seed range ${percent(r.seed_success_min)}–${percent(r.seed_success_max)}</title></circle>`;
    }
  }
  $(id).innerHTML=svg+'</svg>';
}

function stopCompetencePlayback() {
  if(competenceTimer)clearInterval(competenceTimer);
  competenceTimer=null;
  $('competence-replay-play').textContent='▶ Play';
  $('competence-replay-play').setAttribute('aria-label','Play saved one-world trajectory');
}

function selectCompetenceTrajectory() {
  stopCompetencePlayback();
  const rows=(competence?.trajectories || []).filter(r=>r.condition===competenceCondition && (r.policy!=='learner' || r.mode===competenceMode()));
  selectOptions('competence-replay-controller',[...new Set(rows.map(r=>r.policy))].map(p=>[p,policyLabel(p)]),'learner');
  const policy=$('competence-replay-controller').value;
  const available=rows.filter(r=>r.policy===policy);
  const checkpoints=[...new Set(available.map(r=>r.checkpoint))].sort((a,b)=>a-b);
  selectOptions('competence-replay-checkpoint',checkpoints.map(c=>[String(c),policy==='learner'?number(c):'Reference']),String(latestCompetenceStep()));
  $('competence-replay-checkpoint').disabled=policy!=='learner' || checkpoints.length<2;
  const checkpoint=Number($('competence-replay-checkpoint').value);
  const atStep=available.filter(r=>r.checkpoint===checkpoint);
  selectOptions('competence-replay-panel',[...new Set(atStep.map(r=>r.panel))].map(p=>[p,panelLabel(p)]),'heldout');
  const panel=$('competence-replay-panel').value, atPanel=atStep.filter(r=>r.panel===panel);
  selectOptions('competence-replay-seed',[...new Set(atPanel.map(r=>r.seed))].sort((a,b)=>a-b).map(s=>[String(s),String(s)]),'0');
  competenceTrajectory=atPanel.find(r=>r.seed===Number($('competence-replay-seed').value)) || null;
  competenceFrame=0;
  $('competence-replay-scrub').max=competenceTrajectory?.steps.length || 0;
  $('competence-replay-scrub').value='0';
  for(const id of ['competence-replay-play','competence-replay-reset','competence-replay-scrub']) $(id).disabled=!competenceTrajectory;
  drawCompetenceWorld();renderCompetenceReferences();
}

function renderCompetenceReferences() {
  if(!competence)return;
  const panel=$('competence-replay-panel').value || 'heldout';
  const step=$('competence-replay-controller').value==='learner'?Number($('competence-replay-checkpoint').value):latestCompetenceStep();
  const learner=competence.aggregate.find(r=>r.condition===competenceCondition && r.checkpoint===step && r.panel===panel && r.mode===competenceMode());
  const reference=policy=>competence.references?.find(r=>r.condition===competenceCondition && r.policy===policy && r.panel===panel);
  $('competence-reference-title').textContent=`${supportLabel(competenceCondition)} · ${panelLabel(panel).toLowerCase()}`;
  $('competence-reference-bars').innerHTML=[[learner,`Learner at ${stepLabel(step)}`,'#a6e5c1'],[reference('random_actions'),'Random actions','#e7b985'],[reference('shortest_path'),'Shortest path','#98b8d1']].map(([r,label,color])=>`<div class="diagnostic-row"><span>${escape(label)}</span><div class="diagnostic-track">${Number.isFinite(r?.success_rate)?`<div class="diagnostic-fill" style="width:${100*r.success_rate}%;background:${color}"></div>`:''}</div><strong>${percent(r?.success_rate)}</strong></div>`).join('');
  $('competence-reference-caption').textContent=`Success on the same map panel. The learner uses ${competenceMode()==='greedy'?'greedy actions':'ε = 0.1'} at ${number(step)} training steps. Random samples actions; the planner uses the known map without learning. These reference policies do not change with the evaluation toggle. Learner: ${number(learner?.episodes)} episodes; random: ${number(reference('random_actions')?.episodes)}; planner: ${number(reference('shortest_path')?.episodes)}.`;
  $('competence-policy-diagnostics').innerHTML=[
    [percent(learner?.noop_rate),'No-op actions','Learner stays in place'],
    [percent(learner?.optimal_action_rate),'Optimal actions',`${number(learner?.winnable_steps)} still-winnable states`],
    [decimal(learner?.mean_abs_q_error),'Mean |Q error|','Learner vs exact values'],
  ].map(([value,label,detail])=>`<div><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join('');
}

function drawCompetenceWorld() {
  const canvas=$('competence-world-canvas'),ctx=canvas.getContext('2d'),size=canvas.width,t=competenceTrajectory;
  ctx.clearRect(0,0,size,size);ctx.fillStyle='#0a100c';ctx.fillRect(0,0,size,size);
  $('competence-step-diagnostics').innerHTML='';
  if(!t) {
    ctx.fillStyle='#89988e';ctx.font='17px sans-serif';ctx.textAlign='center';ctx.fillText('No saved replay for this selection',size/2,size/2);
    $('competence-world-caption').textContent='No episode is substituted for a missing recording.';
    $('competence-replay-step').textContent='—';$('competence-replay-outcome').textContent='No trajectory';$('competence-replay-explanation').textContent='';return;
  }
  const n=t.grid_size,cell=size/n,step=competenceFrame?t.steps[competenceFrame-1]:null;
  const pos=step?.position || t.start,pellets=step?.pellets || t.pellets_initial;
  for(let r=0;r<n;r++)for(let c=0;c<n;c++){ctx.strokeStyle='#1b2a20';ctx.lineWidth=1;ctx.strokeRect(c*cell+.5,r*cell+.5,cell-1,cell-1);}
  for(const [r,c] of t.walls){ctx.fillStyle='#2c3b31';ctx.fillRect(c*cell+2,r*cell+2,cell-4,cell-4);}
  ctx.strokeStyle='#6e9c7c';ctx.lineWidth=2;ctx.setLineDash([3,5]);ctx.beginPath();
  [t.start,...t.steps.slice(0,competenceFrame).map(s=>s.position)].forEach(([r,c],i)=>{if(i===0)ctx.moveTo((c+.5)*cell,(r+.5)*cell);else ctx.lineTo((c+.5)*cell,(r+.5)*cell);});
  ctx.stroke();ctx.setLineDash([]);
  for(const [r,c] of pellets){ctx.beginPath();ctx.arc((c+.5)*cell,(r+.5)*cell,cell*.105,0,Math.PI*2);ctx.fillStyle='#e7b985';ctx.fill();}
  ctx.beginPath();ctx.arc((pos[1]+.5)*cell,(pos[0]+.5)*cell,cell*.23,0,Math.PI*2);ctx.fillStyle='#a6e5c1';ctx.fill();ctx.strokeStyle='#d8fce7';ctx.lineWidth=2;ctx.stroke();
  const arrows=['↑','↓','←','→'];
  $('competence-world-caption').textContent=`Rule A · key→move: ${(t.action_mapping || [0,1,2,3]).map((move,action)=>`${arrows[action]}→${arrows[move]}`).join('  ')}`;
  $('competence-replay-step').textContent=`${competenceFrame} / ${t.steps.length}`;
  $('competence-replay-outcome').textContent=competenceFrame===t.steps.length?(t.success?'Collected all pellets':'Episode ended without success'):`${pellets.length} pellet${pellets.length===1?'':'s'} left`;
  const mode=t.policy==='learner'?`${t.mode==='greedy'?'greedy':'ε = 0.1'} · ${number(t.checkpoint)} training steps`:'reference policy';
  $('competence-replay-explanation').textContent=`${supportLabel(t.condition)} · ${policyLabel(t.policy)} · ${mode}. ${panelLabel(t.panel)} · seed ${t.seed} · map ${t.map_seed}. This saved episode illustrates behavior; the panel-wide evaluation measures success.`;
  const decision=step || t.steps[0], decisionNumber=competenceFrame || 1;
  if(decision) {
    const metric=[];
    if(Number.isFinite(decision.action))metric.push(`Chosen action ${arrows[decision.action] || decision.action}`);
    if(step && Number.isFinite(step.reward))metric.push(`Reward ${decimal(step.reward)}`);
    if(Number.isFinite(decision.action_regret))metric.push(`Action regret ${decimal(decision.action_regret)}`);
    $('competence-step-diagnostics').innerHTML=`<p><strong>Decision for step ${number(decisionNumber)}</strong> · ${metric.map(escape).join(' · ')}</p>`;
    if(decision.q_values?.length || decision.optimal_q_values?.length) {
      $('competence-step-diagnostics').innerHTML+=`<div class="table-scroll"><table class="evidence-table"><thead><tr><th>Pre-action values</th>${arrows.map((a,i)=>`<th${i===decision.action?' class="chosen-action"':''}>${a}${i===decision.action?' · chosen':''}</th>`).join('')}</tr></thead><tbody>${[['Learned Q',decision.q_values],['Optimal Q',decision.optimal_q_values]].filter(([,values])=>values?.length).map(([label,values])=>`<tr><td>${label}</td>${values.map((v,i)=>`<td${i===decision.action?' class="chosen-action"':''}>${decimal(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p>Values are from before step ${number(decisionNumber)}; the grid shows ${competenceFrame?'the position after that step':'the starting position'}. Q is discounted return. Regret compares the chosen action with the optimal choice for that state.</p>`;
    }
  }
  canvas.setAttribute('aria-label',`${supportLabel(t.condition)}, ${policyLabel(t.policy)}, step ${competenceFrame} of ${t.steps.length}, agent at row ${pos[0]}, column ${pos[1]}, ${pellets.length} pellets remaining.`);
}

$('competence-epsilon').addEventListener('change',renderCompetenceComparison);
for(const id of ['competence-replay-controller','competence-replay-checkpoint','competence-replay-panel','competence-replay-seed']) $(id).addEventListener('change',selectCompetenceTrajectory);
$('competence-replay-play').addEventListener('click',()=>{
  if(competenceTimer){stopCompetencePlayback();return;}
  if(!competenceTrajectory)return;
  if(competenceFrame>=competenceTrajectory.steps.length)competenceFrame=0;
  $('competence-replay-play').textContent='Ⅱ Pause';$('competence-replay-play').setAttribute('aria-label','Pause saved one-world trajectory');
  competenceTimer=setInterval(()=>{competenceFrame=Math.min(competenceFrame+1,competenceTrajectory.steps.length);$('competence-replay-scrub').value=String(competenceFrame);drawCompetenceWorld();if(competenceFrame>=competenceTrajectory.steps.length)stopCompetencePlayback();},160);
});
$('competence-replay-reset').addEventListener('click',()=>{stopCompetencePlayback();competenceFrame=0;$('competence-replay-scrub').value='0';drawCompetenceWorld();});
$('competence-replay-scrub').addEventListener('input',event=>{stopCompetencePlayback();competenceFrame=Number(event.target.value);drawCompetenceWorld();});

function renderAdaptation() {
  const ready = Boolean(adaptation?.aggregate?.length);
  $('adaptation-empty').hidden = ready;
  $('adaptation-content').hidden = !ready;
  if (!ready) return;
  const find = (checkpoint, world) => adaptation.aggregate.find(r => r.condition === 'continued' && r.checkpoint === checkpoint && r.world === world);
  const a = find('after_a','A'), retained = find('after_b','A'), b = find('after_b','B');
  const threshold = adaptation.protocol?.competence_gate?.minimum ?? 0.7;
  const passed = adaptation.run?.interpretation_status !== 'baseline_underlearned' && Number.isFinite(a?.success_rate) && a.success_rate >= threshold;
  $('study-label').textContent = `${adaptation.run?.id || 'Saved pilot'} · ${adaptation.run?.status || 'exploratory'}`;
  $('interpretation-note').classList.toggle('passed',passed);
  $('interpretation-note').textContent = passed
    ? `Initial competence gate met: ${percent(a.success_rate)} on A after learning A (target ${percent(threshold)}). Inspect adaptation and retention separately; this remains a small exploratory study.`
    : `Baseline underlearned: ${percent(a?.success_rate)} on A after learning A, below the ${percent(threshold)} feasibility gate. Later changes do not establish forgetting. The full result is kept here to guide the next experiment.`;
  const version = adaptation.protocol?.id === 'adaptation-v2' ? 'v2' : 'v1';
  $('adaptation-protocol-link').href=`../docs/experiments/adaptation_protocol_${version}.md`;
  $('adaptation-data-link').href=$('study-run').value==='pilot_v1' ? '../experiments/adaptation/pilot_v1/results.json' : 'data/adaptation.json';
  const delta = a && retained ? 100 * (retained.success_rate - a.success_rate) : null;
  const cards = [
    [number(adaptation.protocol.seeds?.length),'Training seeds','Independent initializations'],
    [percent(a?.success_rate),'A competence after learning A','Fixed held-out map panel'],
    [percent(retained?.success_rate),'A performance after learning B',delta === null ? 'Awaiting B evaluation' : `${delta >= 0 ? '+' : ''}${delta.toFixed(1)} percentage points vs after A`],
    [percent(b?.success_rate),'B performance after learning B','Same task, changed control rule'],
  ];
  $('adaptation-stats').innerHTML = cards.map(([value,label,detail]) => `<div class="stat-card"><span>${escape(label)}</span><strong>${escape(value)}</strong><small>${escape(detail)}</small></div>`).join('');
  const rows = replayRows();
  selectOptions('replay-checkpoint',[...checkpoints,'posthoc_reference'].filter(c => rows.some(r => r.checkpoint === c)).map(c => [c,checkpointLabel(c)]),'after_a');
  selectOptions('replay-world',[...new Set(rows.map(r => r.world))].map(w => [w,`World ${w}`]),'A');
  selectOptions('replay-seed',[...new Set(rows.map(r => r.seed))].sort((a,b)=>a-b).map(seed => [String(seed),String(seed)]),'0');
  $('adaptation-method').textContent = `${number(adaptation.run?.training_steps)} training transitions · ${number(adaptation.protocol.eval_panel?.length)} fixed evaluation maps per world and checkpoint · ${number(adaptation.run?.wall_seconds)} seconds elapsed. Evaluation uses greedy actions and does not update the agent.`;
  $('competence-note').textContent = adaptation.run?.competence_note || 'Inspect A competence before interpreting any drop as forgetting. A weak initial policy cannot establish meaningful retention.';
  listItems('adaptation-limitations',adaptation.run?.limitations || adaptation.limitations);
  selectTrajectory(); renderSuccessChart();
}

function selectTrajectory() {
  stopPlayback();
  if (!adaptation) return;
  const rows = replayRows(), checkpoint = $('replay-checkpoint').value;
  const conditions = [...new Set(rows.filter(r=>r.checkpoint===checkpoint).map(r=>r.condition))];
  selectOptions('replay-condition',conditions.map(c=>[c,conditionLabel(c)]),'continued');
  const condition = $('replay-condition').value;
  const available = rows.filter(r=>r.checkpoint===checkpoint && r.condition===condition);
  selectOptions('replay-world',[...new Set(available.map(r=>r.world))].map(w=>[w,`World ${w}`]),'A');
  const world = $('replay-world').value;
  selectOptions('replay-seed',[...new Set(available.filter(r=>r.world===world).map(r=>r.seed))].sort((a,b)=>a-b).map(s=>[String(s),String(s)]),'0');
  const seed = Number($('replay-seed').value);
  currentTrajectory = available.find(r => r.world === world && r.seed === seed) || null;
  frame = 0;
  $('replay-scrub').max = currentTrajectory?.steps.length || 0;
  $('replay-scrub').value = '0';
  $('replay-play').disabled = !currentTrajectory;
  for (const button of document.querySelectorAll('.phase-card')) button.classList.toggle('active',button.dataset.phase === checkpoint);
  drawWorld();
}
for (const id of ['replay-condition','replay-checkpoint','replay-world','replay-seed']) $(id).addEventListener('change',selectTrajectory);
for (const button of document.querySelectorAll('.phase-card')) button.addEventListener('click',() => {
  const value = button.dataset.phase;
  if ([...$('replay-checkpoint').options].some(o=>o.value === value)) { $('replay-checkpoint').value = value; selectTrajectory(); }
});
for (const button of document.querySelectorAll('[data-reference]')) button.addEventListener('click',() => {
  $('replay-checkpoint').value='posthoc_reference';
  selectTrajectory();
  $('replay-condition').value=button.dataset.reference;
  $('replay-world').value=$('evaluation-world').value;
  selectTrajectory();
  $('replay-play').focus();
});
$('replay-play').addEventListener('click',() => {
  if (timer) { stopPlayback(); return; }
  if (!currentTrajectory) return;
  if (frame >= currentTrajectory.steps.length) frame = 0;
  $('replay-play').textContent = 'Ⅱ Pause'; $('replay-play').setAttribute('aria-label','Pause trajectory');
  timer = setInterval(() => {
    frame = Math.min(frame+1,currentTrajectory.steps.length);
    $('replay-scrub').value = String(frame); drawWorld();
    if (frame >= currentTrajectory.steps.length) stopPlayback();
  },160);
});
$('replay-reset').addEventListener('click',() => { stopPlayback(); frame=0; $('replay-scrub').value='0'; drawWorld(); });
$('replay-scrub').addEventListener('input',event => { stopPlayback(); frame=Number(event.target.value); drawWorld(); });

function drawWorld() {
  const canvas = $('world-canvas'), ctx = canvas.getContext('2d');
  const size = canvas.width;
  ctx.clearRect(0,0,size,size);
  ctx.fillStyle = '#0a100c'; ctx.fillRect(0,0,size,size);
  if (!currentTrajectory) {
    ctx.fillStyle='#89988e'; ctx.font='17px sans-serif'; ctx.textAlign='center';
    ctx.fillText('No replay for this combination',size/2,size/2);
    $('world-caption').textContent='Select an available agent and checkpoint.';
    $('replay-step').textContent='—'; $('replay-outcome').textContent='No trajectory'; return;
  }
  const t=currentTrajectory, n=t.grid_size, cell=size/n;
  const step = frame ? t.steps[frame-1] : null;
  const pos = step?.position || t.start;
  const pellets = step?.pellets || t.pellets_initial;
  for(let r=0;r<n;r++) for(let c=0;c<n;c++) {
    ctx.strokeStyle='#1b2a20'; ctx.lineWidth=1;
    ctx.strokeRect(c*cell+.5,r*cell+.5,cell-1,cell-1);
  }
  for(const [r,c] of t.walls) { ctx.fillStyle='#2c3b31';ctx.fillRect(c*cell+2,r*cell+2,cell-4,cell-4); }
  ctx.strokeStyle='#6e9c7c';ctx.lineWidth=2;ctx.setLineDash([3,5]);ctx.beginPath();
  [t.start,...t.steps.slice(0,frame).map(s=>s.position)].forEach(([r,c],i)=>{if(i===0)ctx.moveTo((c+.5)*cell,(r+.5)*cell);else ctx.lineTo((c+.5)*cell,(r+.5)*cell);});
  ctx.stroke();ctx.setLineDash([]);
  for(const [r,c] of pellets) { ctx.beginPath();ctx.arc((c+.5)*cell,(r+.5)*cell,cell*.105,0,Math.PI*2);ctx.fillStyle='#e7b985';ctx.fill(); }
  ctx.beginPath();ctx.arc((pos[1]+.5)*cell,(pos[0]+.5)*cell,cell*.23,0,Math.PI*2);ctx.fillStyle='#a6e5c1';ctx.fill();
  ctx.strokeStyle='#d8fce7';ctx.lineWidth=2;ctx.stroke();
  const arrows=['↑','↓','←','→'],mapping=t.action_mapping || [0,1,2,3];
  const controls = mapping.map((move,action)=>`${arrows[action]}→${arrows[move]}`).join('  ');
  $('world-caption').textContent=`Visible rule ${t.world} · key→move: ${controls}`;
  $('replay-step').textContent=`${frame} / ${t.steps.length}`;
  $('replay-outcome').textContent=frame===t.steps.length ? (t.success?'Collected all pellets':'Time limit reached') : `${pellets.length} pellet${pellets.length===1?'':'s'} left`;
  const seedLabel = t.checkpoint === 'posthoc_reference' ? 'Reference seed' : 'Training seed';
  $('replay-explanation').textContent=`${seedLabel} ${t.seed} · held-out map ${t.map_seed} · ${conditionLabel(t.condition)} · ${checkpointLabel(t.checkpoint)}. This is the first map in the fixed panel, selected independently of its outcome.`;
  canvas.setAttribute('aria-label',`World ${t.world}, ${checkpointLabel(t.checkpoint)}, step ${frame} of ${t.steps.length}, agent at row ${pos[0]}, column ${pos[1]}, ${pellets.length} pellets remaining.`);
}

function renderSuccessChart() {
  if (!adaptation) return;
  const world = $('evaluation-world').value;
  $('chart-title').textContent=world === 'A' ? 'Can it still solve A?' : 'Can it adapt to B?';
  const data=adaptation.aggregate.filter(r=>r.world===world), w=500,h=258,pad={l:42,r:14,t:17,b:40};
  const x=i=>pad.l+i*(w-pad.l-pad.r)/(checkpoints.length-1),y=v=>pad.t+(1-v)*(h-pad.t-pad.b);
  let svg=`<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg"><title>World ${world} evaluation success; mean across training seeds with per-seed range</title>`;
  for(let i=0;i<=4;i++) { const v=i/4;svg+=`<line x1="${pad.l}" y1="${y(v)}" x2="${w-pad.r}" y2="${y(v)}" stroke="#26302a"/><text x="${pad.l-9}" y="${y(v)+4}" text-anchor="end" fill="#8a978f" font-size="10">${100*v}%</text>`; }
  checkpoints.forEach((c,i)=>{svg+=`<text x="${x(i)}" y="${h-15}" text-anchor="${i===0?'start':i===3?'end':'middle'}" fill="#8a978f" font-size="10">${escape(checkpointLabel(c))}</text>`;});
  for(const condition of ['frozen_after_a','continued']) {
    const values=checkpoints.map(c=>data.find(r=>r.checkpoint===c && r.condition===condition) || (c==='untrained'?data.find(r=>r.checkpoint===c && r.condition==='untrained'):null));
    const points=values.flatMap((r,i)=>r?[[i,r]]:[]);if(!points.length)continue;
    const range=[...points.map(([i,r])=>`${x(i)},${y(r.seed_success_max ?? r.success_rate)}`),...points.slice().reverse().map(([i,r])=>`${x(i)},${y(r.seed_success_min ?? r.success_rate)}`)].join(' ');
    svg+=`<polygon points="${range}" fill="${colors[condition]}" opacity=".07"/>`;
    // Missing checkpoints break a line instead of inventing intermediate observations.
    let path='',previous=-2;
    points.forEach(([i,r])=>{path+=`${i===previous+1?'L':'M'}${x(i)},${y(r.success_rate)} `;previous=i;});
    svg+=`<path d="${path}" stroke="${colors[condition]}" stroke-width="2.2" fill="none" ${condition==='frozen_after_a'?'stroke-dasharray="6 5"':''}/>`;
    points.forEach(([i,r])=>{svg+=`<circle cx="${x(i)}" cy="${y(r.success_rate)}" r="4" fill="${colors[condition]}"><title>${escape(conditionLabel(condition))}, ${escape(checkpointLabel(r.checkpoint))}: ${percent(r.success_rate)}; seed range ${percent(r.seed_success_min)}–${percent(r.seed_success_max)}</title></circle>`;});
  }
  svg+='</svg>';$('success-chart').innerHTML=svg;
  $('curve-legend').innerHTML=['continued','frozen_after_a'].map(c=>`<span><i style="background:${colors[c]}"></i>${conditionLabel(c)}</span>`).join('');
  $('chart-caption').textContent='Lines show mean success over the same test maps. Shading shows the minimum–maximum across training seeds, not a confidence interval. The initial untrained point is shared.';
  $('evaluation-table').innerHTML='<thead><tr><th>Checkpoint</th><th>Agent</th><th>Success</th><th>Seed range</th></tr></thead><tbody>'+data.slice().sort((a,b)=>checkpoints.indexOf(a.checkpoint)-checkpoints.indexOf(b.checkpoint)).map(r=>`<tr><td>${escape(checkpointLabel(r.checkpoint))}</td><td>${escape(conditionLabel(r.condition))}</td><td>${percent(r.success_rate)}</td><td>${percent(r.seed_success_min)}–${percent(r.seed_success_max)}</td></tr>`).join('')+'</tbody>';
  renderDiagnostics(world);
}
$('evaluation-world').addEventListener('change',renderSuccessChart);

function renderDiagnostics(world) {
  $('diagnostic-panel').hidden=!diagnostics?.aggregate?.length;
  if(!diagnostics?.aggregate?.length)return;
  const checkpoint=world==='A'?'after_a':'after_b';
  const learned=adaptation.aggregate.find(r=>r.world===world && r.checkpoint===checkpoint && r.condition==='continued');
  const random=diagnostics.aggregate.find(r=>r.world===world && r.condition==='random_actions');
  const planner=diagnostics.aggregate.find(r=>r.world===world && r.condition==='shortest_path');
  $('diagnostic-title').textContent=`Can world ${world} be solved?`;
  $('diagnostic-bars').innerHTML=[
    [learned,`Learner ${world==='A'?'after A':'after B'}`,'#a6e5c1'],
    [random,'Random actions','#e7b985'],[planner,'Shortest path','#98b8d1']
  ].map(([r,label,color])=>`<div class="diagnostic-row"><span>${label}</span><div class="diagnostic-track"><div class="diagnostic-fill" style="width:${100*(r?.success_rate||0)}%;background:${color}"></div></div><strong>${percent(r?.success_rate)}</strong></div>`).join('');
  $('diagnostic-note').textContent=`Same visible state, maps and time limit. These checks were added after the training pilots; they are separate diagnostics. The planner computes routes without learning. Random: ${number(random?.episodes)} episodes; planner: ${number(planner?.episodes)}. Next milestone: establish reliable learning before interpreting retention.`;
}

function renderProvenance() {
  const ready=Boolean(provenance?.scenarios?.length);
  $('provenance-empty').hidden=ready;$('provenance-content').hidden=!ready;if(!ready)return;
  selectOptions('provenance-scenario',provenance.scenarios.map(s=>[s.id,s.label]),'default');
  listItems('provenance-limitations',provenance.limitations);
  $('selected-settings').innerHTML=provenance.policies.map(p=>`<div class="settings-item"><strong>${escape(p.label)}</strong><br>${escape(Object.entries(p.selected_parameters || {}).map(([k,v])=>`${k}: ${v}`).join(' · ') || 'Untuned baseline')}</div>`).join('');
  $('provenance-method').textContent='Settings are selected by calibration shortfall. Held-out seeds never enter that choice. Error intervals and paired comparisons resample complete seeds.';
  renderProvenanceScenario();
}
function renderProvenanceScenario() {
  if(!provenance)return;
  const scenario=provenance.scenarios.find(s=>s.id===$('provenance-scenario').value);if(!scenario)return;
  const rows=scenario.policy_results, max=Math.max(...rows.map(r=>r.ci95_shortfall?.[1] ?? r.mean_shortfall))*1.04;
  const policy=id=>provenance.policies.find(p=>p.id===id);
  $('provenance-bars').innerHTML=rows.map(r=>{
    const p=policy(r.policy_id), low=r.ci95_shortfall?.[0], high=r.ci95_shortfall?.[1];
    const interval=Number.isFinite(low)&&Number.isFinite(high)?`<span style="position:absolute;top:9px;left:${100*low/max}%;width:${100*(high-low)/max}%;height:2px;background:#f2eee5"><span style="position:absolute;left:0;top:-3px;height:8px;border-left:1px solid #f2eee5"></span><span style="position:absolute;right:0;top:-3px;height:8px;border-left:1px solid #f2eee5"></span></span>`:'';
    return `<div class="bar-row"><div>${escape(p?.label||r.policy_id)}<small>${p?.selected_parameters?.threshold!=null?`threshold ${escape(p.selected_parameters.threshold)}`:'untuned baseline'}</small></div><div class="bar-track" aria-label="${escape(p?.label)}: ${number(r.mean_shortfall)}; interval ${number(low)} to ${number(high)}"><div class="bar-fill" style="width:${100*r.mean_shortfall/max}%;background:${colors[r.policy_id]||'#98b8d1'}"></div>${interval}</div><div class="bar-value">${number(r.mean_shortfall)}</div></div>`;
  }).join('');
  $('provenance-caption').textContent=`Mean need-shortfall agent-ticks per 10,000 world ticks, normalized from 1,000-tick runs · ${rows[0]?.n || '—'} paired evaluation seeds · whiskers: 95% seed-bootstrap intervals. One agent can contribute both food and medicine shortfall in a tick.`;
  $('provenance-table').innerHTML='<thead><tr><th>Controller</th><th>Observed false alarms</th><th>Observed missed crises</th><th>Response delay</th><th>Loss vs unique</th></tr></thead><tbody>'+rows.map(r=>{
    const d=r.paired_difference_vs_unique,ci=d?.ci95;
    return `<tr><td>${escape(policy(r.policy_id)?.label||r.policy_id)}</td><td>${percent(r.false_alarm_rate)}</td><td>${percent(r.missed_crisis_rate)}</td><td>${Number.isFinite(r.mean_response_delay_ticks)?r.mean_response_delay_ticks.toFixed(1):'—'} ticks</td><td>${d?`${d.mean>=0?'+':''}${number(d.mean)}<br><small>${number(ci?.[0])} to ${number(ci?.[1])}</small>`:'—'}</td></tr>`;
  }).join('')+'</tbody>';
}
$('provenance-scenario').addEventListener('change',renderProvenanceScenario);

async function getData(path) {
  const response=await fetch(path,{cache:'no-store'});
  if(response.status===404)return null;
  if(!response.ok)throw new Error(`Could not load ${path} (${response.status})`);
  const data=await response.json();
  if(data.schema_version!==1)throw new Error(`Unsupported schema in ${path}`);
  return data;
}
async function loadData() {
  if(loadInProgress)return;loadInProgress=true;$('refresh').disabled=true;$('study-run').disabled=true;
  stopPlayback();stopCompetencePlayback();
  const adaptationPath=$('study-run').value==='pilot_v1'?'../experiments/adaptation/pilot_v1/results.json':'data/adaptation.json';
  const results=await Promise.allSettled([getData(adaptationPath),getData('data/provenance.json'),getData('data/competence.json')]);
  const errors=[];
  diagnostics=null;
  if(results[0].status==='fulfilled') {
    adaptation=results[0].value;
    const directory=adaptation?.artifacts?.directory;
    if(directory?.startsWith('experiments/adaptation/') && !directory.split('/').includes('..')) {
      const diagnosticPath=`../${directory}/diagnostics.json`;
      try { diagnostics=await getData(diagnosticPath);$('diagnostics-link').href=diagnosticPath; }
      catch(error) { errors.push(error.message); }
    }
    renderAdaptation();
  } else { adaptation=null;renderAdaptation();errors.push(results[0].reason.message); }
  if(results[1].status==='fulfilled') { provenance=results[1].value;renderProvenance(); } else errors.push(results[1].reason.message);
  if(results[2].status==='fulfilled') competence=results[2].value; else {competence=null;errors.push(results[2].reason.message);}
  renderCompetence();
  if(errors.length) $('load-status').textContent=errors.join(' · ');
  else {
    const loaded=[competence&&'Competence',adaptation&&'Adaptation',provenance&&'Provenance'].filter(Boolean);
    $('load-status').textContent=`${loaded.join(' + ') || 'No'} saved ${loaded.length===1?'study':'studies'} loaded · ${new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}`;
  }
  $('refresh').disabled=false;$('study-run').disabled=false;loadInProgress=false;
}
$('refresh').addEventListener('click',loadData);
$('study-run').addEventListener('change',loadData);
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopPlayback();stopCompetencePlayback();}});
await loadData();
