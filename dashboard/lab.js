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
let supervised=null, supervisedTrajectory=null, supervisedFrame=0, supervisedTimer=null;
const supervisedMode=()=>$('supervised-epsilon').value;
const supervisedPanelLabel=panel=>panel==='train'?'Training layouts':'Fresh layouts';
const supervisedPolicyLabel=policy=>({learner:'Supervised network',prior_stream:'Historical frozen RL',random_actions:'Random actions',shortest_path:'Shortest path'}[policy] || policy);
const latestSupervisedUpdate=()=>Math.max(0,...(supervised?.aggregate || []).map(r=>r.checkpoint).filter(Number.isFinite));
let fixed=null, fixedTrajectory=null, fixedFrame=0, fixedTimer=null;
const fixedConditions=['exact_q','double_dqn'];
const fixedColors={exact_q:'#a6e5c1',double_dqn:'#98b8d1'};
const fixedLabel=condition=>({exact_q:'Exact targets',double_dqn:'Double DQN targets',shared:'Shared references'}[condition] || condition);
const fixedMode=()=>$('fixed-epsilon').value;
const latestFixedUpdate=()=>Math.max(0,...(fixed?.aggregate || []).map(r=>r.checkpoint).filter(Number.isFinite));
let coverage=null, coverageTrajectory=null, coverageFrame=0, coverageTimer=null;
let coverageConditions=['collected_unique','uniform_subset'];
const coverageStudies={coverage:null,equal_support:null};
const coverageIsEqual=()=>$('coverage-study').value==='equal_support';
const coverageDirection=()=>coverageIsEqual()?'uniform subset minus collected unique states':'collected unique states minus exhaustive states';
const coverageColors={exhaustive:'#a6e5c1',collected_unique:'#98b8d1',uniform_subset:'#e7b985'};
const coverageLabel=condition=>({exhaustive:'Exhaustive states',collected_unique:'Collected unique states',uniform_subset:'Uniform subset',shared:'Shared references'}[condition] || condition);
const coverageMode=()=>$('coverage-epsilon').value;
const latestCoverageUpdate=()=>Math.max(0,...(coverage?.aggregate || []).map(r=>r.checkpoint).filter(Number.isFinite));
const signed=(value,scale=1,suffix='')=>Number.isFinite(value)?`${value>0?'+':''}${(value*scale).toFixed(1)}${suffix}`:'—';

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
const tracks = ['coverage','fixed','supervised','competence','adaptation','provenance'];
function activateTab(name) {
  for (const track of tracks) {
    const active = track === name;
    $(`tab-${track}`).setAttribute('aria-selected',String(active));
    $(`tab-${track}`).tabIndex = active ? 0 : -1;
    $(`${track}-panel`).hidden = !active;
  }
  stopPlayback();
  stopCompetencePlayback();
  stopSupervisedPlayback();
  stopFixedPlayback();
  stopCoveragePlayback();
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

function selectCoverageStudy(){
  stopCoveragePlayback();coverage=coverageStudies[$('coverage-study').value] || null;
  coverageConditions=coverageIsEqual()?['collected_unique','uniform_subset']:['exhaustive','collected_unique'];
  for(const id of ['coverage-replay-condition','coverage-replay-controller','coverage-replay-checkpoint','coverage-replay-panel','coverage-replay-seed','coverage-support-condition'])$(id).value='';
  renderCoverage();
}
$('coverage-study').addEventListener('change',selectCoverageStudy);
function renderCoverage(){
  stopCoveragePlayback();
  const equal=coverageIsEqual();
  $('coverage-study-context').textContent=equal?'Same number of states; different bank composition. Fresh panel belongs only to this study.':'Earlier measured coverage comparison, with its own fresh panel and larger exhaustive bank.';
  $('coverage-intro-copy').textContent=equal?'Both conditions use the same Double DQN learner and the same number of training states. One reuses the archived bank of unique states reached by random collection; the other uses an equally sized uniform subset of the exhaustive state universe. This tests which states are included while holding bank size fixed. Both remain offline and receive all four action transitions.':'Both conditions use the same Double DQN target procedure, network, all four actions, and update budget. One samples every training state. The other samples unique states encountered during random collection on the same layouts. The support and sampling distribution change together; this is offline learning, with the movement rule visible initially.';
  $('coverage-condition-preview').innerHTML=coverageConditions.map(c=>`<span>${coverageLabel(c)}</span>`).join('');
  $('coverage-flow').innerHTML=(equal?[
    ['Same state count','Archived collected bank / uniform subset'],['Different included states','Same exhaustive training universe'],['Same Double DQN','All actions, network, loss, updates'],['Separate fresh panel','Own rollouts; no oracle actions'],
  ]:[['Same training layouts','Every possible state / random trajectories'],['Two fixed state banks','Exhaustive / unique collected states'],['Same Double DQN','All actions, network, loss, update budget'],['New fresh layouts','Own rollouts; no oracle actions']]).map(([title,detail])=>`<li><strong>${title}</strong><small>${detail}</small></li>`).join('');
  $('coverage-flow-note').textContent=equal?'No new collection is performed. The collected bank is reused unchanged, and the uniform bank is sampled before training. Both banks are shared across learner seeds. Detached successor queries may reach states outside a bank without adding them to its direct training support.':'Random collection happens before learning and never adapts to either network. Collected states are sampled uniformly after duplicates are removed. Both conditions still receive all four action transitions at each training state, beyond ordinary trajectory-only experience.';
  $('coverage-paired-caption').textContent=`Every difference is ${coverageDirection()}. Success, efficient-success, and no-op-rate differences are percentage points; a negative step difference means fewer steps. Failed episodes still count. These are descriptive paired measurements from this study's own final fresh panel.`;
  $('coverage-layout-caption').textContent=`Per-layout success and efficient-success differences are −1, 0, or +1. No-op-rate differences are percentage points. Every difference is ${coverageDirection()}.`;

  const {run={},protocol={},gates={}}=coverage || {},dataset=protocol.dataset || {},latest=latestCoverageUpdate();
  const inconsistent=run.status==='inconsistent_not_gate_evidence',stopReason=coverage?.provenance?.stop_reason;
  const consistencyNotice=`Consistency check failed. This run is ineligible for the declared gates.${typeof stopReason==='string' && stopReason?` Reason: ${stopReason}`:''}`;
  const ready=Boolean(coverage?.aggregate?.length);$('coverage-empty').hidden=ready;$('coverage-content').hidden=!ready;
  if(!ready){
    $('coverage-empty-title').textContent=inconsistent?'Consistency check failed.':coverage?'No completed comparison checkpoint.':'No saved experience-coverage comparison yet.';
    $('coverage-empty-message').textContent=inconsistent?consistencyNotice:coverage?`The saved run has no complete rollout aggregates to display. Status: ${(run.status || 'unavailable').replaceAll('_',' ')}.${typeof stopReason==='string' && stopReason?` Reason: ${stopReason}`:''} No gate can be established.`:'Refresh after the declared study finishes. Earlier measured experiments remain available in the other tabs.';
    for(const id of ['coverage-train-chart','coverage-fresh-chart','coverage-train-agreement','coverage-fresh-agreement','coverage-loss-chart','coverage-outcome-success','coverage-outcome-efficient','coverage-outcome-steps'])$(id).innerHTML='';$('coverage-outcome-caption').textContent='';return;
  }
  $('coverage-study-label').textContent=`${run.id || protocol.id || 'Saved comparison'} · ${(run.status || 'exploratory').replaceAll('_',' ')}`;
  const eligible=gates.eligible===true && !inconsistent,conditionGates=gates.per_condition || [];
  const status=value=>!eligible?'Not eligible':value===true?'Met':value===false?'Not met':'Not measured';
  let interpretation;
  if(inconsistent)interpretation=consistencyNotice;
  else if(run.interpretation==='incomplete_not_gate_evidence' || (run.status && run.status!=='complete')) interpretation='Incomplete comparison. Partial measurements cannot pass the declared gates or establish a difference between coverage conditions.';
  else if(!eligible)interpretation='Smoke or protocol-deviation run. These measurements check execution and cannot pass the declared research gates.';
  else {
    const exact=conditionGates.find(r=>r.condition===coverageConditions[0]),boot=conditionGates.find(r=>r.condition===coverageConditions[1]);
    if(equal){
      if(exact?.fresh && boot?.fresh)interpretation='Both equal-size banks pass the fresh-success gate. Compare the paired gap and efficiency before deciding which state composition helps under this fixed learner and budget.';
      else if(exact?.fresh && boot?.fresh===false)interpretation='Collected unique states pass the fresh-success gate; the uniform subset does not. State composition matters under this equal-size, offline comparison. This does not establish that every trajectory-derived bank is better.';
      else if(exact?.fresh===false && boot?.fresh)interpretation='The uniform subset passes the fresh-success gate; collected unique states do not. Bank composition remains a limitation at the same state count under this learner and budget. Inspect the coverage and own-bank fit diagnostics before attributing the gap to any particular missing states.';
      else interpretation='Neither equal-size bank establishes the declared fresh-success gate. Inspect own-bank fitting, full-universe diagnostics, and coverage before interpreting which states matter.';
    }
    else if(exact?.fresh && boot?.fresh)interpretation='Both coverage conditions pass the fresh-success gate. Inspect the paired gap and efficiency before deciding how much offline state coverage was sufficient under this collector and budget.';
    else if(exact?.fresh && boot?.fresh===false)interpretation='Exhaustive states pass the fresh-success gate; collected unique states do not. The collected support and its uniform sampling distribution limit this fixed offline learner. This does not isolate the effect of any one missing state or establish that every collector fails.';
    else if(exact?.fresh===false && boot?.fresh)interpretation='Collected unique states pass the fresh-success gate; exhaustive states do not. Inspect fitting, paired outcomes, and sampling distributions before drawing a broader conclusion.';
    else interpretation='Neither coverage condition establishes the declared fresh-success gate. Inspect the exhaustive control and fitting diagnostics before interpreting the collected bank.';
    const unresolved=conditionGates.filter(r=>r.training_fit===false).map(r=>coverageLabel(r.condition));
    if(unresolved.length)interpretation+=` Training fit remains unresolved for ${unresolved.join(' and ')}.`;
    const inefficient=conditionGates.filter(r=>r.efficient===false).map(r=>coverageLabel(r.condition));
    if(inefficient.length)interpretation+=` The separate efficiency gate is unmet for ${inefficient.join(' and ')}.`;
    interpretation+=' Both conditions retain privileged all-action transitions. This is offline learning, not online RL competence.';
  }
  $('coverage-gate-note').textContent=interpretation;$('coverage-gate-note').classList.toggle('passed',eligible && conditionGates.length===2 && conditionGates.every(r=>r.fresh===true && r.training_fit===true && r.efficient===true));
  $('coverage-stats').innerHTML=[
    [number(protocol.seeds?.length),'Paired initialization seeds','Same initial weights within each pair'],
    [number(latest),'Latest optimizer checkpoint','Matched updates per condition and seed'],
    [number(dataset.train_states),'Exhaustive reference states','Collected current-state coverage is measured below'],
    [number(dataset.heldout_map_seeds?.length),'New fresh layouts',`First map ${number(dataset.heldout_map_seeds?.[0])} · shared by both conditions`],
  ].map(([value,label,detail])=>`<div class="stat-card"><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join('');
  $('coverage-gates').textContent=`Training fit requires ${percent(protocol.gates?.training_success)} greedy training success and ${percent(protocol.gates?.training_state_optimal)} optimal-action agreement over the exhaustive still-winnable training-state panel, including uncollected states, in every seed. Fresh success requires ${percent(protocol.gates?.heldout_success)} in every seed and a score above random. Efficient success means collecting within twice the shortest-path length, with all episodes in the denominator; its separate threshold is ${percent(protocol.gates?.efficient_success ?? protocol.gates?.heldout_efficient_success)} per seed. The exploration toggle does not change these greedy gates.`;
  $('coverage-gate-results').innerHTML=conditionGates.map(c=>`<div class="gate-row"><strong>${coverageLabel(c.condition)}</strong><p>All-seed gates · Training fit: ${status(c.training_fit)} · Fresh: ${status(c.fresh)} · Efficient: ${status(c.efficient)}</p>${(c.per_seed || []).map(r=>`<p>Seed ${number(r.seed)} · Train success ${percent(r.train_success)} / agreement ${percent(r.train_state_optimal)} · Fresh success ${percent(r.fresh_success)} / efficient ${percent(r.fresh_efficient_success)}</p><small>Training fit: ${status(r.training_fit)} · Fresh: ${status(r.fresh)} · Efficient: ${status(r.efficient)}</small>`).join('')}</div>`).join('');
  const paired=coverage.paired_differences || {};
  $('coverage-paired-table').innerHTML='<thead><tr><th>Panel</th><th>Paired seed</th><th>Success Δ</th><th>Efficient success Δ</th><th>Mean steps Δ</th><th>No-op rate Δ</th></tr></thead><tbody>'+(paired.per_seed || []).map(r=>`<tr><td>${supervisedPanelLabel(r.panel)}</td><td>${number(r.seed)}</td><td>${signed(r.success_rate_delta,100,' pp')}</td><td>${signed(r.efficient_success_rate_delta,100,' pp')}</td><td>${signed(r.mean_steps_delta)}</td><td>${signed(r.noop_rate_delta,100,' pp')}</td></tr>`).join('')+(paired.aggregate || []).map(r=>`<tr class="paired-total"><td>${supervisedPanelLabel(r.panel)}</td><td>Mean across seeds</td><td>${signed(r.mean_seed_success_rate_delta,100,' pp')}</td><td>${signed(r.mean_seed_efficient_success_rate_delta,100,' pp')}</td><td>${signed(r.mean_seed_mean_steps_delta)}</td><td>${signed(r.mean_seed_noop_rate_delta,100,' pp')}</td></tr>`).join('')+'</tbody>';
  $('coverage-layout-table').innerHTML='<thead><tr><th>Panel</th><th>Seed</th><th>Map</th><th>Success Δ</th><th>Efficient success Δ</th><th>Steps Δ</th><th>No-op steps Δ</th><th>No-op rate Δ</th></tr></thead><tbody>'+(paired.per_layout || []).map(r=>`<tr><td>${supervisedPanelLabel(r.panel)}</td><td>${number(r.seed)}</td><td>${number(r.map_seed)}</td><td>${signed(r.success_delta)}</td><td>${signed(r.efficient_success_delta)}</td><td>${signed(r.steps_delta)}</td><td>${signed(r.noop_steps_delta)}</td><td>${signed(r.noop_rate_delta,100,' pp')}</td></tr>`).join('')+'</tbody>';
  $('coverage-method').textContent=`${number(run.train_updates)} optimizer updates and ${number(run.training_examples)} state presentations across both conditions and all seeds · batch size ${number(protocol.budget?.batch_size)} · ${number(run.wall_seconds)} seconds elapsed. Both conditions share the network, Double DQN targets, all four actions, loss reduction, optimizer, and update budget. Their fixed current-state supports and uniform sampling distributions intentionally differ. Random collection happens before optimization. The collected bank contains unique pre-action states; detached successor queries do not expand it.`;
  if(equal)$('coverage-method').textContent=$('coverage-method').textContent.replace('Random collection happens before optimization.','No new collection is performed; the archived collected bank is reused and a uniform subset with the same state count is selected before optimization.');
  const trainingCosts=coverageConditions.map(condition=>{const rows=Object.entries(run.per_condition_seed_timing || {}).filter(([key])=>key.startsWith(condition+':')).map(([,value])=>value);return rows.length?`${coverageLabel(condition)} optimization: ${rows.reduce((sum,r)=>sum+(r.training_wall_seconds || 0),0).toFixed(1)} s`:null;}).filter(Boolean);
  if(trainingCosts.length)$('coverage-method').textContent+=` ${trainingCosts.join(' · ')}. Equal update counts do not mean equal compute.`;
  if(Number.isFinite(run.peak_rss_bytes))$('coverage-method').textContent+=` Peak process memory: ${(run.peak_rss_bytes/1024**3).toFixed(2)} GiB; one CPU thread.`;
  listItems('coverage-limitations',run.limitations);
  $('coverage-evidence-links').innerHTML=`<a href="data/${equal?'equal_support':'coverage'}.json" download>Download displayed data ↓</a>`+[['protocol_document','Study design'],['protocol','Saved protocol'],['report','Measured findings'],['training','Optimization loss'],['evaluations','Raw rollouts'],['state_metrics','Raw state measurements'],['dataset_metadata','Dataset provenance'],['sampling','Sampling provenance'],['paired_differences','Paired outcomes'],['coverage','Coverage counts'],['support_diagnostics','Own-bank fit diagnostics'],['collection_steps','Raw collection steps'],['collection_episodes','Raw collection episodes'],['manifest','Artifact checksums']].flatMap(([key,label])=>{const path=coverage.artifacts?.[key];return typeof path==='string' && /^(docs|experiments)\/[a-zA-Z0-9_./-]+$/.test(path) && !path.split('/').includes('..')?[`<a href="../${escape(path)}">${label} ↗</a>`]:[];}).join('');
  renderCoverageSupport();
  renderCoverageSupportFit();
  renderCoverageComparison();
}
function renderCoverageSupport(){
  const equal=coverageIsEqual(),description=coverage?.coverage || {},records=equal?(description.per_condition || []):[{condition:'collected_unique',...description}];
  $('coverage-bank-comparison').hidden=!equal;$('coverage-support-selector-label').hidden=!equal;
  selectOptions('coverage-support-condition',records.map(r=>[r.condition,coverageLabel(r.condition)]),'collected_unique');
  const support=records.find(r=>r.condition===$('coverage-support-condition').value) || {},maps=support.by_map || [],buckets=support.by_time_bucket || [],queries=support.successor_queries || {};

  const overall=support.overall || {},precisePercent=value=>Number.isFinite(value)?`${(100*value).toFixed(2)}%`:'—';
  if(equal){
    const intersection=description.intersection || {};
    $('coverage-intersection-caption').textContent=`Each bank contains ${number(description.support_size)} states out of ${number(description.total_training_states)}. Their intersection contains ${number(intersection.states)} states (${precisePercent(intersection.overlap_fraction)} of either bank); union ${number(intersection.union_states)}; Jaccard overlap ${precisePercent(intersection.jaccard)}. No new collection is performed in this study.`;
    $('coverage-bank-table').innerHTML='<thead><tr><th>Bank</th><th>Included states</th><th>Layouts represented</th><th>Current-state coverage</th><th>Winnable coverage</th><th>Goal-near coverage</th><th>Nonterminal edges outside support</th></tr></thead><tbody>'+records.map(r=>`<tr><td>${coverageLabel(r.condition)}</td><td>${number(r.unique_current_states)}</td><td>${number((r.by_map || []).filter(m=>m.visited_states>0).length)} / ${number(r.by_map?.length)}</td><td>${precisePercent(r.current_state_fraction)}</td><td>${precisePercent(r.winnable_current_state_fraction)}</td><td>${precisePercent(r.goal_near_current_state_fraction)}</td><td>${precisePercent(r.successor_queries?.outside_support_fraction)}</td></tr>`).join('')+'</tbody>';
  }else{$('coverage-intersection-caption').textContent='';$('coverage-bank-table').innerHTML='';}
  $('coverage-support-caption').textContent=`Before learning, a uniform-random collector produced ${number(support.collection_episodes)} episodes and ${number(support.collection_steps)} action steps on the training layouts. Its fixed bank contains ${number(support.unique_current_states)} unique pre-action states out of ${number(support.total_training_states)} exhaustive states. The bank is shared by all collected-condition learner seeds.`;
  if(equal)$('coverage-support-caption').textContent=`Inspecting ${coverageLabel(support.condition)}: ${number(support.unique_current_states)} included current states out of ${number(support.total_training_states)}. These are bank-membership counts, not new visits. ${description.synthetic_smoke_support?'This smoke uses synthetic support to validate execution; it does not use the archived collected bank.':'The collected bank is reused from the archived collector; the uniform bank is sampled from the exhaustive universe.'} Each bank is shared across learner seeds.`;
  $('coverage-support-stats').innerHTML=[
    [percent(support.current_state_fraction),equal?'Included current-state coverage':'Unique current-state coverage',`${number(support.unique_current_states)} / ${number(support.total_training_states)} states`],
    ...(equal?[[number(support.unique_current_states),'Included current states','Membership in this fixed bank'],['None','New collection','Archived collected bank reused; no new episodes']]:[[number(support.collection_episodes),'Collection episodes','Uniform random; completed before optimization'],[number(support.collection_steps),'Collection action steps','Repeated visits are deduplicated for training']]),
    [precisePercent(support.winnable_current_state_fraction),'Winnable-state coverage',`${number(overall.visited_winnable_states)} / ${number(overall.winnable_states)} still-winnable states`],
    [precisePercent(support.goal_near_current_state_fraction),'Goal-near state coverage',`${number(overall.visited_goal_near_states)} / ${number(overall.goal_near_states)} states · physical shortest path ≤ 2 moves; all remaining clocks`],
    [precisePercent(queries.outside_support_fraction),'Successor edges outside support',`${number(queries.outside_support_nonterminal_transitions)} / ${number(queries.nonterminal_transitions)} nonterminal action edges`],
  ].map(([value,label,detail])=>`<div><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join('');
  if(maps.length){
    const cols=Math.min(16,maps.length),cell=24,gap=3,w=cols*cell,h=Math.ceil(maps.length/cols)*cell;
    $('coverage-map-heatmap').innerHTML=`<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg"><title>Included current-state coverage across ${number(maps.length)} training layouts</title>${maps.map((row,i)=>{const x=(i%cols)*cell,y=Math.floor(i/cols)*cell,ratio=Number.isFinite(row.coverage_rate)?Math.max(0,Math.min(1,row.coverage_rate)):0;return `<rect x="${x}" y="${y}" width="${cell-gap}" height="${cell-gap}" rx="2" fill="#98b8d1" fill-opacity="${.08+.92*ratio}"><title>Map ${number(row.map_seed)}: ${number(row.visited_states)} / ${number(row.states)} unique current states (${percent(row.coverage_rate)})</title></rect>`;}).join('')}</svg>`;
  }else $('coverage-map-heatmap').innerHTML='<p class="help-text">No saved per-layout coverage measurements.</p>';
  $('coverage-time-bars').innerHTML=buckets.map(row=>`<div class="diagnostic-row"><span>${row.time_bucket==='all'?'All remaining times':`${escape(row.time_bucket)} steps left`}<small>${number(row.visited_states)} / ${number(row.states)} states</small></span><div class="diagnostic-track">${Number.isFinite(row.coverage_rate)?`<div class="diagnostic-fill" style="width:${100*Math.max(0,Math.min(1,row.coverage_rate))}%;background:#98b8d1"></div>`:''}</div><strong>${percent(row.coverage_rate)}</strong></div>`).join('') || '<p class="help-text">No saved remaining-time coverage measurements.</p>';
  $('coverage-successor-caption').textContent=`Across all four transitions from each included current state, ${number(queries.outside_support_nonterminal_transitions)} of ${number(queries.nonterminal_transitions)} nonterminal successors (${percent(queries.outside_support_fraction)}) lie outside its current-state support. These are static bank counts, not optimizer samples or newly collected experience. Terminal transitions use their reward directly.`;
  $('coverage-support-table').innerHTML='<thead><tr><th>Scope</th><th>Layout / remaining time</th><th>Included states</th><th>Exhaustive states</th><th>Coverage</th><th>Included winnable states</th><th>Exhaustive winnable states</th><th>Winnable coverage</th></tr></thead><tbody>'+[[maps,'Layout','map_seed'],[buckets,'Time bucket','time_bucket']].flatMap(([rows,label,key])=>rows.map(row=>`<tr><td>${label}</td><td>${escape(row[key])}</td><td>${number(row.visited_states)}</td><td>${number(row.states)}</td><td>${percent(row.coverage_rate)}</td><td>${number(row.visited_winnable_states)}</td><td>${number(row.winnable_states)}</td><td>${percent(row.winnable_coverage_rate)}</td></tr>`)).join('')+'</tbody>';
}

$('coverage-support-condition').addEventListener('change',renderCoverageSupport);
function renderCoverageSupportFit(){
  const diagnostic=coverage?.support_diagnostics || {},rows=[...(diagnostic.per_seed || []),...(diagnostic.pooled || [])];
  $('coverage-support-fit-panel').hidden=!coverageIsEqual() || !rows.length;
  $('coverage-support-fit-table').innerHTML=rows.length?'<thead><tr><th>Condition</th><th>Seed scope</th><th>Own membership</th><th>State evaluations</th><th>Unique states</th><th>Winnable evaluations</th><th>Unique winnable states</th><th>Optimal actions</th><th>Q MAE</th><th>Q RMSE</th><th>95th pct. state MAE</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${coverageLabel(r.condition)}</td><td>${Number.isFinite(r.seed)?`Seed ${number(r.seed)}`:`Pooled ${number(r.seeds)} seeds`}</td><td>${r.subset==='support'?'Own bank':'Outside own bank'}</td><td>${number(r.states)}</td><td>${number(r.unique_states ?? r.states)}</td><td>${number(r.winnable_states)}</td><td>${number(r.unique_winnable_states ?? r.winnable_states)}</td><td>${percent(r.optimal_action_rate)}</td><td>${decimal(r.mean_abs_q_error)}</td><td>${decimal(r.rmse_q_error)}</td><td>${decimal(r.q95_abs_q_error)}</td></tr>`).join('')+'</tbody>':'';
}

function renderCoverageComparison(){
  if(!coverage?.aggregate?.length)return;
  const latest=latestCoverageUpdate(),states=coverage.state_aggregate || [],stateAll=states.filter(r=>r.time_bucket==='all'),rollouts=coverage.aggregate.filter(r=>r.mode===coverageMode());
  const finalUpdate=coverage.protocol?.budget?.updates_per_seed ?? Math.max(0,...(coverage.protocol?.checkpoints || []).filter(Number.isFinite));
  const finalRows=condition=>rollouts.find(r=>r.condition===condition && r.panel==='heldout' && r.checkpoint===finalUpdate);
  const outcomes=[
    ['success','Primary · Success','success_rate',percent,'All fresh episodes in the denominator'],
    ['efficient','Secondary · Efficient success','efficient_success_rate',percent,`Success within ${number(coverage.protocol?.gates?.planner_step_multiplier)} × the shortest-path length`],
    ['steps','Diagnostic · Mean steps','mean_steps',value=>Number.isFinite(value)?value.toFixed(1):'—','Includes failed episodes; fewer steps is better'],
  ];
  for(const [id,label,key,format,detail] of outcomes)$(`coverage-outcome-${id}`).innerHTML=`<span>${label}</span><div class="fixed-outcome-pair">${coverageConditions.map(condition=>`<div><small>${coverageLabel(condition)}</small><strong style="color:${coverageColors[condition]}">${format(finalRows(condition)?.[key])}</strong></div>`).join('')}</div><small>${detail}</small>`;
  $('coverage-outcome-caption').textContent=`${number(finalUpdate)} updates per condition and seed · ${coverageMode()==='greedy'?'greedy actions':'ε = 0.1 exploration'}. Outcomes follow the rollout mode; declared gates stay greedy. Missing final measurements appear as —.`;
  const series=(rows,panel,key,low,high)=>coverageConditions.map(condition=>({label:coverageLabel(condition),color:coverageColors[condition],rows:rows.filter(r=>r.condition===condition && r.panel===panel).map(r=>({checkpoint:r.checkpoint,value:r[key],low:r[low],high:r[high]}))}));
  for(const [panel,id] of [['train','train'],['heldout','fresh']]){
    renderUpdateChart(`coverage-${id}-chart`,series(rollouts,panel,'success_rate','seed_success_min','seed_success_max'),coverage.protocol?.checkpoints,true,`${supervisedPanelLabel(panel)} episode success`);
    renderUpdateChart(`coverage-${id}-agreement`,series(stateAll,panel,'optimal_action_rate','seed_optimal_action_min','seed_optimal_action_max'),coverage.protocol?.checkpoints,true,`${supervisedPanelLabel(panel)} optimal-action agreement`);
  }
  const losses=coverage.loss_aggregate || [];
  renderUpdateChart('coverage-loss-chart',coverageConditions.map(condition=>({label:coverageLabel(condition),color:coverageColors[condition],rows:losses.filter(r=>r.condition===condition).map(r=>({checkpoint:r.checkpoint,value:r.mean_loss,low:r.seed_loss_min,high:r.seed_loss_max}))})),losses.map(r=>r.checkpoint),false,'Within-condition minibatch loss');
  $('coverage-curve-legend').innerHTML=coverageConditions.map(c=>`<span><i style="background:${coverageColors[c]}"></i>${coverageLabel(c)}</span>`).join('');
  $('coverage-mode-note').textContent=`Rollouts and network replays use ${coverageMode()==='greedy'?'greedy actions':'ε = 0.1 exploration'}. Curves use optimizer updates; shading is the seed range. Exhaustive greedy agreement, loss, paired final differences, and declared gates stay unchanged when this mode changes.`;
  $('coverage-fit-metrics').innerHTML=coverageConditions.flatMap(condition=>{const row=stateAll.find(r=>r.condition===condition && r.panel==='heldout' && r.checkpoint===latest);return [['MAE',row?.mean_abs_q_error],['RMSE',row?.rmse_q_error],['95th pct. state MAE',row?.q95_abs_q_error]].map(([label,value])=>`<div><span>${coverageLabel(condition)} · ${label}</span><strong>${decimal(value)}</strong><small>Fresh states at ${number(latest)} updates</small></div>`);}).join('');
  $('coverage-rollout-table').innerHTML='<thead><tr><th>Condition</th><th>Updates</th><th>Panel</th><th>Success</th><th>Seed range</th><th>Efficient success</th><th>Episodes</th><th>Mean steps</th><th>No-op rate</th></tr></thead><tbody>'+rollouts.map(r=>`<tr><td>${coverageLabel(r.condition)}</td><td>${number(r.checkpoint)}</td><td>${supervisedPanelLabel(r.panel)}</td><td>${percent(r.success_rate)}</td><td>${percent(r.seed_success_min)}–${percent(r.seed_success_max)}</td><td>${percent(r.efficient_success_rate)}</td><td>${number(r.episodes)}</td><td>${decimal(r.mean_steps)}</td><td>${percent(r.noop_rate)}</td></tr>`).join('')+'</tbody>';
  $('coverage-state-table').innerHTML='<thead><tr><th>Condition</th><th>Updates</th><th>Panel</th><th>Remaining time</th><th>States</th><th>Winnable states</th><th>Q MAE</th><th>Q RMSE</th><th>95th pct. state MAE</th><th>Signed Q bias</th><th>Optimal actions</th></tr></thead><tbody>'+states.map(r=>`<tr><td>${coverageLabel(r.condition)}</td><td>${number(r.checkpoint)}</td><td>${supervisedPanelLabel(r.panel)}</td><td>${escape(r.time_bucket)}</td><td>${number(r.states)}</td><td>${number(r.winnable_states)}</td><td>${decimal(r.mean_abs_q_error)}</td><td>${decimal(r.rmse_q_error)}</td><td>${decimal(r.q95_abs_q_error)}</td><td>${decimal(r.mean_signed_q_bias)}</td><td>${percent(r.optimal_action_rate)}</td></tr>`).join('')+'</tbody>';
  selectCoverageTrajectory();
}
function stopCoveragePlayback(){if(coverageTimer)clearInterval(coverageTimer);coverageTimer=null;$('coverage-replay-play').textContent='▶ Play';$('coverage-replay-play').setAttribute('aria-label','Play saved experience-coverage trajectory');}
function selectCoverageTrajectory(){
  stopCoveragePlayback();
  const rows=(coverage?.trajectories || []).filter(r=>r.policy!=='learner' || r.mode===coverageMode());
  selectOptions('coverage-replay-condition',[...new Set(rows.map(r=>r.condition))].map(c=>[c,coverageLabel(c)]),coverageConditions[0]);
  const atCondition=rows.filter(r=>r.condition===$('coverage-replay-condition').value);
  selectOptions('coverage-replay-controller',[...new Set(atCondition.map(r=>r.policy))].map(p=>[p,p==='learner'?'Trained network':policyLabel(p)]),'learner');
  const policy=$('coverage-replay-controller').value,available=atCondition.filter(r=>r.policy===policy),updates=[...new Set(available.map(r=>r.checkpoint))].sort((a,b)=>a-b);
  selectOptions('coverage-replay-checkpoint',updates.map(c=>[String(c),policy==='learner'?number(c):'Reference']),String(latestCoverageUpdate()));$('coverage-replay-checkpoint').disabled=policy!=='learner' || updates.length<2;
  const atStep=available.filter(r=>r.checkpoint===Number($('coverage-replay-checkpoint').value));
  selectOptions('coverage-replay-panel',[...new Set(atStep.map(r=>r.panel))].map(p=>[p,supervisedPanelLabel(p)]),'heldout');
  const atPanel=atStep.filter(r=>r.panel===$('coverage-replay-panel').value);
  selectOptions('coverage-replay-seed',[...new Set(atPanel.map(r=>r.seed))].sort((a,b)=>a-b).map(s=>[String(s),String(s)]),'0');
  coverageTrajectory=atPanel.find(r=>r.seed===Number($('coverage-replay-seed').value)) || null;coverageFrame=0;
  $('coverage-replay-scrub').max=coverageTrajectory?.steps.length || 0;$('coverage-replay-scrub').value='0';
  for(const id of ['coverage-replay-play','coverage-replay-reset','coverage-replay-scrub'])$(id).disabled=!coverageTrajectory;
  drawCoverageWorld();renderCoverageReferences();
}
function drawCoverageWorld(){
  const t=coverageTrajectory,description=t?`${coverageLabel(t.condition)} · ${t.policy==='learner'?`${number(t.checkpoint)} offline optimizer updates · ${t.mode==='greedy'?'greedy':'ε = 0.1'}`:policyLabel(t.policy)}. ${supervisedPanelLabel(t.panel)} · seed ${t.seed} · map ${t.map_seed}. Recorded on the first panel layout, selected before outcomes. No new learning occurs during replay.`:'';
  drawRecordedWorld('coverage',t,coverageFrame,description,t?`${coverageLabel(t.condition)}, ${policyLabel(t.policy)}`:'');
}
function renderCoverageReferences(){
  if(!coverage)return;
  const panel=$('coverage-replay-panel').value || 'heldout',update=$('coverage-replay-controller').value==='learner'?Number($('coverage-replay-checkpoint').value):latestCoverageUpdate();
  const rows=coverageConditions.map(condition=>[coverage.aggregate.find(r=>r.condition===condition && r.panel===panel && r.checkpoint===update && r.mode===coverageMode()),`${coverageLabel(condition)} at ${stepLabel(update)}`,coverageColors[condition]]);
  for(const [policy,color] of [['random_actions','#e7b985'],['shortest_path','#b5a7d2']])rows.push([coverage.references?.find(r=>r.policy===policy && r.panel===panel),policyLabel(policy),color]);
  $('coverage-reference-title').textContent=supervisedPanelLabel(panel);
  $('coverage-reference-bars').innerHTML=rows.filter(([r])=>r).map(([r,label,color])=>`<div class="diagnostic-row"><span>${escape(label)}</span><div class="diagnostic-track">${Number.isFinite(r.success_rate)?`<div class="diagnostic-fill" style="width:${100*r.success_rate}%;background:${color}"></div>`:''}</div><strong>${percent(r.success_rate)}</strong></div>`).join('');
  $('coverage-reference-caption').textContent=`Networks use ${coverageMode()==='greedy'?'greedy actions':'ε = 0.1'} on the same panel. Random and planner use their own policies. Episodes: ${rows.filter(([r])=>r).map(([r,label])=>`${label}: ${number(r.episodes)}`).join(' · ')}.`;
}
$('coverage-epsilon').addEventListener('change',renderCoverageComparison);
for(const id of ['coverage-replay-condition','coverage-replay-controller','coverage-replay-checkpoint','coverage-replay-panel','coverage-replay-seed'])$(id).addEventListener('change',selectCoverageTrajectory);
$('coverage-replay-play').addEventListener('click',()=>{if(coverageTimer){stopCoveragePlayback();return;}if(!coverageTrajectory)return;if(coverageFrame>=coverageTrajectory.steps.length)coverageFrame=0;$('coverage-replay-play').textContent='Ⅱ Pause';$('coverage-replay-play').setAttribute('aria-label','Pause saved experience-coverage trajectory');coverageTimer=setInterval(()=>{coverageFrame=Math.min(coverageFrame+1,coverageTrajectory.steps.length);$('coverage-replay-scrub').value=String(coverageFrame);drawCoverageWorld();if(coverageFrame>=coverageTrajectory.steps.length)stopCoveragePlayback();},160);});
$('coverage-replay-reset').addEventListener('click',()=>{stopCoveragePlayback();coverageFrame=0;$('coverage-replay-scrub').value='0';drawCoverageWorld();});
$('coverage-replay-scrub').addEventListener('input',event=>{stopCoveragePlayback();coverageFrame=Number(event.target.value);drawCoverageWorld();});

function renderFixed(){
  stopFixedPlayback();
  const {run={},protocol={},gates={}}=fixed || {},dataset=protocol.dataset || {},latest=latestFixedUpdate();
  const inconsistent=run.status==='inconsistent_not_gate_evidence',stopReason=fixed?.provenance?.stop_reason;
  const consistencyNotice=`Consistency check failed. This run is ineligible for the declared gates.${typeof stopReason==='string' && stopReason?` Reason: ${stopReason}`:''}`;
  const ready=Boolean(fixed?.aggregate?.length);$('fixed-empty').hidden=ready;$('fixed-content').hidden=!ready;
  if(!ready){
    $('fixed-empty-title').textContent=inconsistent?'Consistency check failed.':fixed?'No completed comparison checkpoint.':'No saved fixed-data comparison yet.';
    $('fixed-empty-message').textContent=inconsistent?consistencyNotice:fixed?`The saved run has no complete rollout aggregates to display. Status: ${(run.status || 'unavailable').replaceAll('_',' ')}.${typeof stopReason==='string' && stopReason?` Reason: ${stopReason}`:''} No gate can be established.`:'Refresh after the declared study finishes. Earlier measured experiments remain available in the other tabs.';
    for(const id of ['fixed-train-chart','fixed-fresh-chart','fixed-train-agreement','fixed-fresh-agreement','fixed-loss-chart','fixed-outcome-success','fixed-outcome-efficient','fixed-outcome-steps'])$(id).innerHTML='';$('fixed-outcome-caption').textContent='';return;
  }
  $('fixed-study-label').textContent=`${run.id || protocol.id || 'Saved comparison'} · ${(run.status || 'exploratory').replaceAll('_',' ')}`;
  const eligible=gates.eligible===true && !inconsistent,conditionGates=gates.per_condition || [];
  const status=value=>!eligible?'Not eligible':value===true?'Met':value===false?'Not met':'Not measured';
  let interpretation;
  if(inconsistent)interpretation=consistencyNotice;
  else if(run.interpretation==='incomplete_not_gate_evidence' || (run.status && run.status!=='complete')) interpretation='Incomplete comparison. Partial measurements cannot pass the declared gates or establish a difference between target procedures.';
  else if(!eligible)interpretation='Smoke or protocol-deviation run. These measurements check execution and cannot pass the declared research gates.';
  else {
    const exact=conditionGates.find(r=>r.condition==='exact_q'),boot=conditionGates.find(r=>r.condition==='double_dqn');
    if(exact?.fresh && boot?.fresh)interpretation='Both target procedures pass the fresh-success gate under fixed, broad state coverage. Inspect the paired gap, training fit, and route efficiency before deciding what transfers to online learning.';
    else if(exact?.fresh && boot?.fresh===false)interpretation='Exact targets pass the fresh-success gate; Double DQN targets do not. Under this matched dataset and budget, the bootstrapped target procedure remains a learning limitation. This does not show that every bootstrapped method fails.';
    else if(exact?.fresh===false && boot?.fresh)interpretation='Double DQN targets pass the fresh-success gate; exact targets do not. Inspect paired results and fitting diagnostics before drawing a broader conclusion.';
    else interpretation='Neither target procedure establishes the declared fresh-success gate. Inspect whether the exact-target control fits its training states before interpreting the bootstrapped comparison.';
    const unresolved=conditionGates.filter(r=>r.training_fit===false).map(r=>fixedLabel(r.condition));
    if(unresolved.length)interpretation+=` Training fit remains unresolved for ${unresolved.join(' and ')}.`;
    const inefficient=conditionGates.filter(r=>r.efficient===false).map(r=>fixedLabel(r.condition));
    if(inefficient.length)interpretation+=` The separate efficiency gate is unmet for ${inefficient.join(' and ')}.`;
    interpretation+=' This remains an offline experiment with privileged coverage, not online RL competence.';
  }
  $('fixed-gate-note').textContent=interpretation;$('fixed-gate-note').classList.toggle('passed',eligible && conditionGates.length===2 && conditionGates.every(r=>r.fresh===true && r.training_fit===true && r.efficient===true));
  $('fixed-stats').innerHTML=[
    [number(protocol.seeds?.length),'Paired initialization seeds','Same initial weights within each pair'],
    [number(latest),'Latest optimizer checkpoint','Matched updates per condition and seed'],
    [number(dataset.train_states),'Fixed training states','All four actions in each sampled row'],
    [number(dataset.heldout_map_seeds?.length),'New fresh layouts',`First map ${number(dataset.heldout_map_seeds?.[0])} · shared by both conditions`],
  ].map(([value,label,detail])=>`<div class="stat-card"><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join('');
  $('fixed-gates').textContent=`Training fit requires ${percent(protocol.gates?.training_success)} greedy training success and ${percent(protocol.gates?.training_state_optimal)} optimal-action agreement on still-winnable training states in every seed. Fresh success requires ${percent(protocol.gates?.heldout_success)} in every seed and a score above random. Efficient success means collecting within twice the shortest-path length, with all episodes in the denominator; its separate threshold is ${percent(protocol.gates?.efficient_success ?? protocol.gates?.heldout_efficient_success)} per seed. The exploration toggle does not change these greedy gates.`;
  $('fixed-gate-results').innerHTML=conditionGates.map(c=>`<div class="gate-row"><strong>${fixedLabel(c.condition)}</strong><p>All-seed gates · Training fit: ${status(c.training_fit)} · Fresh: ${status(c.fresh)} · Efficient: ${status(c.efficient)}</p>${(c.per_seed || []).map(r=>`<p>Seed ${number(r.seed)} · Train success ${percent(r.train_success)} / agreement ${percent(r.train_state_optimal)} · Fresh success ${percent(r.fresh_success)} / efficient ${percent(r.fresh_efficient_success)}</p><small>Training fit: ${status(r.training_fit)} · Fresh: ${status(r.fresh)} · Efficient: ${status(r.efficient)}</small>`).join('')}</div>`).join('');
  const paired=fixed.paired_differences || {};
  $('fixed-paired-table').innerHTML='<thead><tr><th>Panel</th><th>Paired seed</th><th>Success Δ</th><th>Efficient success Δ</th><th>Mean steps Δ</th><th>No-op rate Δ</th></tr></thead><tbody>'+(paired.per_seed || []).map(r=>`<tr><td>${supervisedPanelLabel(r.panel)}</td><td>${number(r.seed)}</td><td>${signed(r.success_rate_delta,100,' pp')}</td><td>${signed(r.efficient_success_rate_delta,100,' pp')}</td><td>${signed(r.mean_steps_delta)}</td><td>${signed(r.noop_rate_delta,100,' pp')}</td></tr>`).join('')+(paired.aggregate || []).map(r=>`<tr class="paired-total"><td>${supervisedPanelLabel(r.panel)}</td><td>Mean across seeds</td><td>${signed(r.mean_seed_success_rate_delta,100,' pp')}</td><td>${signed(r.mean_seed_efficient_success_rate_delta,100,' pp')}</td><td>${signed(r.mean_seed_mean_steps_delta)}</td><td>${signed(r.mean_seed_noop_rate_delta,100,' pp')}</td></tr>`).join('')+'</tbody>';
  $('fixed-layout-table').innerHTML='<thead><tr><th>Panel</th><th>Seed</th><th>Map</th><th>Success Δ</th><th>Steps Δ</th><th>No-op steps Δ</th></tr></thead><tbody>'+(paired.per_layout || []).map(r=>`<tr><td>${supervisedPanelLabel(r.panel)}</td><td>${number(r.seed)}</td><td>${number(r.map_seed)}</td><td>${signed(r.success_delta)}</td><td>${signed(r.steps_delta)}</td><td>${signed(r.noop_steps_delta)}</td></tr>`).join('')+'</tbody>';
  $('fixed-method').textContent=`${number(run.train_updates)} optimizer updates and ${number(run.training_examples)} state presentations across both conditions and all seeds · batch size ${number(protocol.budget?.batch_size)} · ${number(run.wall_seconds)} seconds elapsed. Both conditions reuse the same state batches, all four action labels, network, loss reduction, optimizer, and update budget. No live environment collection is used for training. Read the protocol for target-network refresh and terminal handling.`;
  const trainingCosts=fixedConditions.map(condition=>{const rows=Object.entries(run.per_condition_seed_timing || {}).filter(([key])=>key.startsWith(condition+':')).map(([,value])=>value);return rows.length?`${fixedLabel(condition)} optimization: ${rows.reduce((sum,r)=>sum+(r.training_wall_seconds || 0),0).toFixed(1)} s`:null;}).filter(Boolean);
  if(trainingCosts.length)$('fixed-method').textContent+=` ${trainingCosts.join(' · ')}. Equal update counts do not mean equal compute.`;
  if(Number.isFinite(run.peak_rss_bytes))$('fixed-method').textContent+=` Peak process memory: ${(run.peak_rss_bytes/1024**3).toFixed(2)} GiB; one CPU thread.`;
  listItems('fixed-limitations',run.limitations);
  $('fixed-evidence-links').innerHTML='<a href="data/fixed_targets.json" download>Download displayed data ↓</a>'+[['protocol_document','Study design'],['protocol','Saved protocol'],['report','Measured findings'],['training','Optimization loss'],['evaluations','Raw rollouts'],['state_metrics','Raw state measurements'],['dataset_metadata','Dataset provenance'],['sampling','Paired sampling summary'],['paired_differences','Paired outcomes'],['manifest','Artifact checksums']].flatMap(([key,label])=>{const path=fixed.artifacts?.[key];return typeof path==='string' && /^(docs|experiments)\/[a-zA-Z0-9_./-]+$/.test(path) && !path.split('/').includes('..')?[`<a href="../${escape(path)}">${label} ↗</a>`]:[];}).join('');
  renderFixedComparison();
}
function renderFixedComparison(){
  if(!fixed?.aggregate?.length)return;
  const latest=latestFixedUpdate(),states=fixed.state_aggregate || [],stateAll=states.filter(r=>r.time_bucket==='all'),rollouts=fixed.aggregate.filter(r=>r.mode===fixedMode());
  const finalUpdate=fixed.protocol?.budget?.updates_per_seed ?? Math.max(0,...(fixed.protocol?.checkpoints || []).filter(Number.isFinite));
  const finalRows=condition=>rollouts.find(r=>r.condition===condition && r.panel==='heldout' && r.checkpoint===finalUpdate);
  const outcomes=[
    ['success','Primary · Success','success_rate',percent,'All fresh episodes in the denominator'],
    ['efficient','Secondary · Efficient success','efficient_success_rate',percent,`Success within ${number(fixed.protocol?.gates?.planner_step_multiplier)} × the shortest-path length`],
    ['steps','Diagnostic · Mean steps','mean_steps',value=>Number.isFinite(value)?value.toFixed(1):'—','Includes failed episodes; fewer steps is better'],
  ];
  for(const [id,label,key,format,detail] of outcomes)$(`fixed-outcome-${id}`).innerHTML=`<span>${label}</span><div class="fixed-outcome-pair">${fixedConditions.map(condition=>`<div><small>${fixedLabel(condition)}</small><strong style="color:${fixedColors[condition]}">${format(finalRows(condition)?.[key])}</strong></div>`).join('')}</div><small>${detail}</small>`;
  $('fixed-outcome-caption').textContent=`${number(finalUpdate)} updates per condition and seed · ${fixedMode()==='greedy'?'greedy actions':'ε = 0.1 exploration'}. Outcomes follow the rollout mode; declared gates stay greedy. Missing final measurements appear as —.`;
  const series=(rows,panel,key,low,high)=>fixedConditions.map(condition=>({label:fixedLabel(condition),color:fixedColors[condition],rows:rows.filter(r=>r.condition===condition && r.panel===panel).map(r=>({checkpoint:r.checkpoint,value:r[key],low:r[low],high:r[high]}))}));
  for(const [panel,id] of [['train','train'],['heldout','fresh']]){
    renderUpdateChart(`fixed-${id}-chart`,series(rollouts,panel,'success_rate','seed_success_min','seed_success_max'),fixed.protocol?.checkpoints,true,`${supervisedPanelLabel(panel)} episode success`);
    renderUpdateChart(`fixed-${id}-agreement`,series(stateAll,panel,'optimal_action_rate','seed_optimal_action_min','seed_optimal_action_max'),fixed.protocol?.checkpoints,true,`${supervisedPanelLabel(panel)} optimal-action agreement`);
  }
  const losses=fixed.loss_aggregate || [];
  renderUpdateChart('fixed-loss-chart',fixedConditions.map(condition=>({label:fixedLabel(condition),color:fixedColors[condition],rows:losses.filter(r=>r.condition===condition).map(r=>({checkpoint:r.checkpoint,value:r.mean_loss,low:r.seed_loss_min,high:r.seed_loss_max}))})),losses.map(r=>r.checkpoint),false,'Within-condition minibatch loss');
  $('fixed-curve-legend').innerHTML=fixedConditions.map(c=>`<span><i style="background:${fixedColors[c]}"></i>${fixedLabel(c)}</span>`).join('');
  $('fixed-mode-note').textContent=`Rollouts and network replays use ${fixedMode()==='greedy'?'greedy actions':'ε = 0.1 exploration'}. Curves use optimizer updates; shading is the seed range. Exhaustive greedy agreement, loss, paired final differences, and declared gates stay unchanged when this mode changes.`;
  $('fixed-fit-metrics').innerHTML=fixedConditions.flatMap(condition=>{const row=stateAll.find(r=>r.condition===condition && r.panel==='heldout' && r.checkpoint===latest);return [['MAE',row?.mean_abs_q_error],['RMSE',row?.rmse_q_error],['95th pct. state MAE',row?.q95_abs_q_error]].map(([label,value])=>`<div><span>${fixedLabel(condition)} · ${label}</span><strong>${decimal(value)}</strong><small>Fresh states at ${number(latest)} updates</small></div>`);}).join('');
  $('fixed-rollout-table').innerHTML='<thead><tr><th>Condition</th><th>Updates</th><th>Panel</th><th>Success</th><th>Seed range</th><th>Efficient success</th><th>Episodes</th><th>Mean steps</th><th>No-op rate</th></tr></thead><tbody>'+rollouts.map(r=>`<tr><td>${fixedLabel(r.condition)}</td><td>${number(r.checkpoint)}</td><td>${supervisedPanelLabel(r.panel)}</td><td>${percent(r.success_rate)}</td><td>${percent(r.seed_success_min)}–${percent(r.seed_success_max)}</td><td>${percent(r.efficient_success_rate)}</td><td>${number(r.episodes)}</td><td>${decimal(r.mean_steps)}</td><td>${percent(r.noop_rate)}</td></tr>`).join('')+'</tbody>';
  $('fixed-state-table').innerHTML='<thead><tr><th>Condition</th><th>Updates</th><th>Panel</th><th>Remaining time</th><th>States</th><th>Winnable states</th><th>Q MAE</th><th>Q RMSE</th><th>95th pct. state MAE</th><th>Signed Q bias</th><th>Optimal actions</th></tr></thead><tbody>'+states.map(r=>`<tr><td>${fixedLabel(r.condition)}</td><td>${number(r.checkpoint)}</td><td>${supervisedPanelLabel(r.panel)}</td><td>${escape(r.time_bucket)}</td><td>${number(r.states)}</td><td>${number(r.winnable_states)}</td><td>${decimal(r.mean_abs_q_error)}</td><td>${decimal(r.rmse_q_error)}</td><td>${decimal(r.q95_abs_q_error)}</td><td>${decimal(r.mean_signed_q_bias)}</td><td>${percent(r.optimal_action_rate)}</td></tr>`).join('')+'</tbody>';
  selectFixedTrajectory();
}
function stopFixedPlayback(){if(fixedTimer)clearInterval(fixedTimer);fixedTimer=null;$('fixed-replay-play').textContent='▶ Play';$('fixed-replay-play').setAttribute('aria-label','Play saved fixed-data trajectory');}
function selectFixedTrajectory(){
  stopFixedPlayback();
  const rows=(fixed?.trajectories || []).filter(r=>r.policy!=='learner' || r.mode===fixedMode());
  selectOptions('fixed-replay-condition',[...new Set(rows.map(r=>r.condition))].map(c=>[c,fixedLabel(c)]),'exact_q');
  const atCondition=rows.filter(r=>r.condition===$('fixed-replay-condition').value);
  selectOptions('fixed-replay-controller',[...new Set(atCondition.map(r=>r.policy))].map(p=>[p,p==='learner'?'Trained network':policyLabel(p)]),'learner');
  const policy=$('fixed-replay-controller').value,available=atCondition.filter(r=>r.policy===policy),updates=[...new Set(available.map(r=>r.checkpoint))].sort((a,b)=>a-b);
  selectOptions('fixed-replay-checkpoint',updates.map(c=>[String(c),policy==='learner'?number(c):'Reference']),String(latestFixedUpdate()));$('fixed-replay-checkpoint').disabled=policy!=='learner' || updates.length<2;
  const atStep=available.filter(r=>r.checkpoint===Number($('fixed-replay-checkpoint').value));
  selectOptions('fixed-replay-panel',[...new Set(atStep.map(r=>r.panel))].map(p=>[p,supervisedPanelLabel(p)]),'heldout');
  const atPanel=atStep.filter(r=>r.panel===$('fixed-replay-panel').value);
  selectOptions('fixed-replay-seed',[...new Set(atPanel.map(r=>r.seed))].sort((a,b)=>a-b).map(s=>[String(s),String(s)]),'0');
  fixedTrajectory=atPanel.find(r=>r.seed===Number($('fixed-replay-seed').value)) || null;fixedFrame=0;
  $('fixed-replay-scrub').max=fixedTrajectory?.steps.length || 0;$('fixed-replay-scrub').value='0';
  for(const id of ['fixed-replay-play','fixed-replay-reset','fixed-replay-scrub'])$(id).disabled=!fixedTrajectory;
  drawFixedWorld();renderFixedReferences();
}
function drawFixedWorld(){
  const t=fixedTrajectory,description=t?`${fixedLabel(t.condition)} · ${t.policy==='learner'?`${number(t.checkpoint)} offline optimizer updates · ${t.mode==='greedy'?'greedy':'ε = 0.1'}`:policyLabel(t.policy)}. ${supervisedPanelLabel(t.panel)} · seed ${t.seed} · map ${t.map_seed}. Recorded on the first panel layout, selected before outcomes. No new learning occurs during replay.`:'';
  drawRecordedWorld('fixed',t,fixedFrame,description,t?`${fixedLabel(t.condition)}, ${policyLabel(t.policy)}`:'');
}
function renderFixedReferences(){
  if(!fixed)return;
  const panel=$('fixed-replay-panel').value || 'heldout',update=$('fixed-replay-controller').value==='learner'?Number($('fixed-replay-checkpoint').value):latestFixedUpdate();
  const rows=fixedConditions.map(condition=>[fixed.aggregate.find(r=>r.condition===condition && r.panel===panel && r.checkpoint===update && r.mode===fixedMode()),`${fixedLabel(condition)} at ${stepLabel(update)}`,fixedColors[condition]]);
  for(const [policy,color] of [['random_actions','#e7b985'],['shortest_path','#b5a7d2']])rows.push([fixed.references?.find(r=>r.policy===policy && r.panel===panel),policyLabel(policy),color]);
  $('fixed-reference-title').textContent=supervisedPanelLabel(panel);
  $('fixed-reference-bars').innerHTML=rows.filter(([r])=>r).map(([r,label,color])=>`<div class="diagnostic-row"><span>${escape(label)}</span><div class="diagnostic-track">${Number.isFinite(r.success_rate)?`<div class="diagnostic-fill" style="width:${100*r.success_rate}%;background:${color}"></div>`:''}</div><strong>${percent(r.success_rate)}</strong></div>`).join('');
  $('fixed-reference-caption').textContent=`Networks use ${fixedMode()==='greedy'?'greedy actions':'ε = 0.1'} on the same panel. Random and planner use their own policies. Episodes: ${rows.filter(([r])=>r).map(([r,label])=>`${label}: ${number(r.episodes)}`).join(' · ')}.`;
}
$('fixed-epsilon').addEventListener('change',renderFixedComparison);
for(const id of ['fixed-replay-condition','fixed-replay-controller','fixed-replay-checkpoint','fixed-replay-panel','fixed-replay-seed'])$(id).addEventListener('change',selectFixedTrajectory);
$('fixed-replay-play').addEventListener('click',()=>{if(fixedTimer){stopFixedPlayback();return;}if(!fixedTrajectory)return;if(fixedFrame>=fixedTrajectory.steps.length)fixedFrame=0;$('fixed-replay-play').textContent='Ⅱ Pause';$('fixed-replay-play').setAttribute('aria-label','Pause saved fixed-data trajectory');fixedTimer=setInterval(()=>{fixedFrame=Math.min(fixedFrame+1,fixedTrajectory.steps.length);$('fixed-replay-scrub').value=String(fixedFrame);drawFixedWorld();if(fixedFrame>=fixedTrajectory.steps.length)stopFixedPlayback();},160);});
$('fixed-replay-reset').addEventListener('click',()=>{stopFixedPlayback();fixedFrame=0;$('fixed-replay-scrub').value='0';drawFixedWorld();});
$('fixed-replay-scrub').addEventListener('input',event=>{stopFixedPlayback();fixedFrame=Number(event.target.value);drawFixedWorld();});

function renderSupervised() {
  stopSupervisedPlayback();
  const ready=Boolean(supervised?.aggregate?.length);
  $('supervised-empty').hidden=ready;$('supervised-content').hidden=!ready;if(!ready)return;
  const {protocol={},run={},gates={}}=supervised, dataset=protocol.dataset || {}, latest=latestSupervisedUpdate();
  $('supervised-study-label').textContent=`${run.id || protocol.id || 'Saved study'} · ${(run.status || 'exploratory').replaceAll('_',' ')}`;
  const interpretations={
    fresh_competence_gate_met:'Fresh-layout supervised gate met. Every seed reaches the declared fresh rollout threshold and exceeds random. Exact privileged supervision produced useful behavior on this panel; this does not establish RL competence or identify the cause of the earlier RL gap.',
    fresh_pass_fit_unresolved:'Fresh-layout rollout gate met; training fit remains unresolved. Every seed passes the fresh success gate, but the two-part training-fit gate is not met in every seed. Inspect rollout and full-state measurements separately. This remains supervised evidence, not RL competence.',
    training_fit_only:'Training fit established; the fresh-layout gate remains unmet. The network learns the supplied training answers well enough to pass both fit criteria. Useful transfer remains limited under this design.',
    training_fit_not_established:'Training fit was not established. The declared training rollout and state-action agreement criteria were not both met in every seed. Fitting this supervised dataset needs diagnosis before interpreting transfer.',
    smoke_or_protocol_deviation_not_gate_evidence:'Smoke or protocol-deviation run. These measurements check execution and cannot pass the declared supervised research gates.',
    incomplete_not_gate_evidence:'Incomplete study. The declared comparison did not finish within its admission budget. Partial measurements cannot pass a supervised gate.',
  };
  $('supervised-gate-note').textContent=interpretations[run.interpretation] || 'Gate interpretation is unavailable. Inspect the saved protocol and per-seed results before drawing a learning conclusion.';
  $('supervised-gate-note').classList.toggle('passed',run.interpretation==='fresh_competence_gate_met');
  $('supervised-stats').innerHTML=[
    [number(protocol.seeds?.length),'Training seeds','Same compact network architecture'],
    [number(latest),'Latest optimizer checkpoint','Updates, not environment transitions'],
    [number(dataset.train_states),'Training states','Four exact action targets per state'],
    [number(dataset.heldout_map_seeds?.length),'Fresh wall-and-goal layouts','Unseen layout keys, fixed evaluation panel'],
  ].map(([value,label,detail])=>`<div class="stat-card"><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join('');
  $('supervised-hypothesis').textContent=protocol.hypothesis || '';
  $('supervised-gates').textContent=`Each seed must reach ${percent(protocol.gates?.training_success)} greedy training-layout rollout success and ${percent(protocol.gates?.training_state_optimal)} optimal-action agreement on still-winnable training states for the training-fit gate. The fresh gate requires ${percent(protocol.gates?.heldout_success)} greedy fresh-layout success in each seed and a score above random. These are supervised feasibility gates, not RL competence.`;
  const status=value=>gates.eligible!==true?'Not eligible':value===true?'Met':value===false?'Not met':'Not measured';
  $('supervised-gate-results').innerHTML=(gates.per_seed || []).map(r=>`<div class="gate-row"><strong>Seed ${number(r.seed)}</strong><p>Training rollout: ${percent(r.train_success)} · Training action agreement: ${percent(r.train_state_optimal)}</p><p>Fresh rollout: ${percent(r.fresh_success)}</p><small>Training fit: ${status(r.training_fit)} · Fresh gate: ${status(r.fresh)}</small></div>`).join('')+`<div class="gate-row"><strong>All-seed gates</strong><p>Training fit: ${status(gates.training_fit)} · Fresh: ${status(gates.fresh)}</p></div>`;
  $('supervised-method').textContent=`Offline optimizer updates: ${number(run.train_updates)} across all seeds · batch examples consumed: ${number(run.training_examples)} · batch size: ${number(protocol.budget?.batch_size)} · ${number(run.wall_seconds)} seconds elapsed. Exact targets cover all four actions. Dataset construction, optimization and evaluation are separate costs; consult the saved accounting. No RL replay or bootstrapped targets are used for this supervised fit.`;
  listItems('supervised-limitations',run.limitations);
  $('supervised-evidence-links').innerHTML='<a href="data/supervised.json" download>Download displayed data ↓</a>'+[['protocol_document','Study design'],['protocol','Saved protocol'],['report','Measured findings'],['training','Recorded optimization loss'],['evaluations','Raw rollout evaluations'],['state_metrics','Raw state measurements'],['dataset_metadata','Dataset provenance'],['sampling','Sampling summary'],['prior_reference_metadata','Historical reference provenance'],['manifest','Artifact checksums']].flatMap(([key,label])=>{
    const path=supervised.artifacts?.[key];return typeof path==='string' && /^(docs|experiments)\/[a-zA-Z0-9_./-]+$/.test(path) && !path.split('/').includes('..')?[`<a href="../${escape(path)}">${label} ↗</a>`]:[];
  }).join('');
  renderSupervisedComparison();
}

function renderSupervisedComparison() {
  if(!supervised?.aggregate?.length)return;
  const latest=latestSupervisedUpdate(),protocol=supervised.protocol || {},stateRows=supervised.state_aggregate || [];
  const panels=['train','heldout'],panelColors={train:'#a6e5c1',heldout:'#98b8d1'};
  const rolloutRows=supervised.aggregate.filter(r=>r.mode===supervisedMode());
  const stateAll=stateRows.filter(r=>r.time_bucket==='all');
  const series=(rows,value,low,high)=>panels.map(panel=>({label:supervisedPanelLabel(panel),color:panelColors[panel],rows:rows.filter(r=>r.panel===panel).map(r=>({checkpoint:r.checkpoint,value:r[value],low:r[low],high:r[high]}))}));
  renderUpdateChart('supervised-success-chart',series(rolloutRows,'success_rate','seed_success_min','seed_success_max'),protocol.checkpoints,true,'Episode success');
  renderUpdateChart('supervised-agreement-chart',series(stateAll,'optimal_action_rate','seed_optimal_action_min','seed_optimal_action_max'),protocol.checkpoints,true,'Winnable-state optimal-action agreement');
  const losses=supervised.loss_aggregate || [];
  renderUpdateChart('supervised-loss-chart',[{label:'Mean minibatch loss',color:'#e7b985',rows:losses.map(r=>({checkpoint:r.checkpoint,value:r.mean_loss,low:r.seed_loss_min,high:r.seed_loss_max}))}],losses.map(r=>r.checkpoint),false,'All-action SmoothL1 minibatch loss');
  $('supervised-loss-caption').textContent='Recorded pre-update minibatch SmoothL1 loss, averaged over each preceding logged window (normally 100 updates; the last window may be shorter). This is not exhaustive dataset loss. Shading is the range across seed means.';
  const metric=panel=>stateAll.find(r=>r.panel===panel && r.checkpoint===latest);
  $('supervised-fit-metrics').innerHTML=panels.flatMap(panel=>[[decimal(metric(panel)?.mean_abs_q_error),`${supervisedPanelLabel(panel)} MAE`],[decimal(metric(panel)?.rmse_q_error),`${supervisedPanelLabel(panel)} RMSE`]]).map(([value,label])=>`<div><span>${label}</span><strong>${value}</strong><small>At ${number(latest)} updates</small></div>`).join('');
  $('supervised-success-caption').textContent=`${supervisedMode()==='greedy'?'Greedy':'ε = 0.1'} episodes from each layout’s declared start and full time horizon. Curves use optimizer updates on a linear axis. Shading is the seed minimum–maximum, not a confidence interval.`;
  $('supervised-curve-legend').innerHTML=panels.map(panel=>`<span><i style="background:${panelColors[panel]}"></i>${supervisedPanelLabel(panel)}</span>`).join('');
  $('supervised-mode-note').textContent=supervisedMode()==='greedy'?'Rollouts and learned-policy replays use greedy actions. Exhaustive state agreement always tests greedy ranking; optimization loss and state-value errors do not change with this rollout toggle.':'Rollouts and neural-policy replays use ε = 0.1: occasional uniform random actions. Exhaustive state agreement remains greedy; the declared gates also remain greedy. Random and planner references keep their own policies.';
  $('supervised-rollout-table').innerHTML='<thead><tr><th>Updates</th><th>Layout panel</th><th>Success</th><th>Seed range</th><th>Episodes</th><th>No-op rate</th><th>Optimal actions</th><th>Winnable visited states</th><th>Visited-state |Q error|</th></tr></thead><tbody>'+rolloutRows.map(r=>`<tr><td>${number(r.checkpoint)}</td><td>${supervisedPanelLabel(r.panel)}</td><td>${percent(r.success_rate)}</td><td>${percent(r.seed_success_min)}–${percent(r.seed_success_max)}</td><td>${number(r.episodes)}</td><td>${percent(r.noop_rate)}</td><td>${percent(r.optimal_action_rate)}</td><td>${number(r.winnable_steps)}</td><td>${decimal(r.mean_abs_q_error)}</td></tr>`).join('')+'</tbody>';
  $('supervised-state-table').innerHTML='<thead><tr><th>Updates</th><th>Layout panel</th><th>Remaining time</th><th>States</th><th>Winnable states</th><th>Q MAE</th><th>Q RMSE</th><th>95th percentile |error|</th><th>Signed Q bias</th><th>Optimal actions</th><th>Winnable regret</th></tr></thead><tbody>'+stateRows.map(r=>`<tr><td>${number(r.checkpoint)}</td><td>${supervisedPanelLabel(r.panel)}</td><td>${escape(r.time_bucket)}</td><td>${number(r.states)}</td><td>${number(r.winnable_states)}</td><td>${decimal(r.mean_abs_q_error)}</td><td>${decimal(r.rmse_q_error)}</td><td>${decimal(r.q95_abs_q_error)}</td><td>${decimal(r.mean_signed_q_bias)}</td><td>${percent(r.optimal_action_rate)}</td><td>${decimal(r.winnable_mean_action_regret)}</td></tr>`).join('')+'</tbody>';
  selectSupervisedTrajectory();
}

function renderUpdateChart(id,series,checkpoints,rates,title) {
  const steps=[...new Set(checkpoints || [])].filter(Number.isFinite).sort((a,b)=>a-b);
  const maximum=Math.max(1,...steps),values=series.flatMap(s=>s.rows.flatMap(r=>[r.value,r.high])).filter(Number.isFinite);
  const ceiling=rates?1:Math.max(.001,...values)*1.05,w=500,h=265,p={l:48,r:17,t:18,b:45};
  const x=step=>p.l+step/maximum*(w-p.l-p.r),y=value=>p.t+(1-value/ceiling)*(h-p.t-p.b),fmt=rates?percent:decimal;
  let svg=`<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg"><title>${escape(title)} by optimizer update</title>`;
  for(let i=0;i<=4;i++){const value=i*ceiling/4;svg+=`<line x1="${p.l}" y1="${y(value)}" x2="${w-p.r}" y2="${y(value)}" stroke="#26302a"/><text x="${p.l-8}" y="${y(value)+4}" text-anchor="end" fill="#8a978f" font-size="10">${fmt(value)}</text>`;}
  for(let i=0;i<=3;i++){const step=i*maximum/3;svg+=`<text x="${x(step)}" y="${h-25}" text-anchor="${i===0?'start':i===3?'end':'middle'}" fill="#8a978f" font-size="10">${escape(stepLabel(Math.round(step)))}</text>`;}
  svg+=`<text x="${w/2}" y="${h-6}" text-anchor="middle" fill="#8a978f" font-size="10">Optimizer updates per seed · linear scale</text>`;
  for(const s of series){
    const segments=[];let segment=[];
    for(const step of steps){const row=s.rows.find(r=>r.checkpoint===step);if(Number.isFinite(row?.value))segment.push(row);else if(segment.length){segments.push(segment);segment=[];}}if(segment.length)segments.push(segment);
    for(const points of segments){
      if(points.every(r=>Number.isFinite(r.low)&&Number.isFinite(r.high)))svg+=`<polygon points="${[...points.map(r=>`${x(r.checkpoint)},${y(r.high)}`),...points.slice().reverse().map(r=>`${x(r.checkpoint)},${y(r.low)}`)].join(' ')}" fill="${s.color}" opacity=".08"/>`;
      svg+=`<path d="${points.map((r,i)=>`${i?'L':'M'}${x(r.checkpoint)},${y(r.value)}`).join(' ')}" stroke="${s.color}" stroke-width="2" fill="none"/>`;
      for(const r of points)svg+=`<circle cx="${x(r.checkpoint)}" cy="${y(r.value)}" r="${steps.length>30?1.5:3.5}" fill="${s.color}"><title>${escape(s.label)}, ${number(r.checkpoint)} updates: ${fmt(r.value)}</title></circle>`;
    }
  }
  if(!values.length)svg+=`<text x="${w/2}" y="${h/2}" text-anchor="middle" fill="#8a978f" font-size="12">No saved measurements</text>`;
  $(id).innerHTML=svg+'</svg>';
}

function stopSupervisedPlayback(){if(supervisedTimer)clearInterval(supervisedTimer);supervisedTimer=null;$('supervised-replay-play').textContent='▶ Play';$('supervised-replay-play').setAttribute('aria-label','Play saved supervised trajectory');}
function selectSupervisedTrajectory(){
  stopSupervisedPlayback();
  const rows=(supervised?.trajectories || []).filter(r=>!['learner','prior_stream'].includes(r.policy) || r.mode===supervisedMode());
  selectOptions('supervised-replay-controller',[...new Set(rows.map(r=>r.policy))].map(p=>[p,supervisedPolicyLabel(p)]),'learner');
  const policy=$('supervised-replay-controller').value,available=rows.filter(r=>r.policy===policy);
  const checkpoints=[...new Set(available.map(r=>r.checkpoint))].sort((a,b)=>a-b);
  selectOptions('supervised-replay-checkpoint',checkpoints.map(c=>[String(c),policy==='learner'?number(c):'Frozen reference']),String(latestSupervisedUpdate()));
  $('supervised-replay-checkpoint').disabled=policy!=='learner' || checkpoints.length<2;
  const atStep=available.filter(r=>r.checkpoint===Number($('supervised-replay-checkpoint').value));
  selectOptions('supervised-replay-panel',[...new Set(atStep.map(r=>r.panel))].map(p=>[p,supervisedPanelLabel(p)]),'heldout');
  const atPanel=atStep.filter(r=>r.panel===$('supervised-replay-panel').value);
  selectOptions('supervised-replay-seed',[...new Set(atPanel.map(r=>r.seed))].sort((a,b)=>a-b).map(s=>[String(s),String(s)]),'0');
  supervisedTrajectory=atPanel.find(r=>r.seed===Number($('supervised-replay-seed').value)) || null;supervisedFrame=0;
  $('supervised-replay-scrub').max=supervisedTrajectory?.steps.length || 0;$('supervised-replay-scrub').value='0';
  for(const id of ['supervised-replay-play','supervised-replay-reset','supervised-replay-scrub'])$(id).disabled=!supervisedTrajectory;
  drawSupervisedWorld();renderSupervisedReferences();
}
function drawSupervisedWorld(){
  const t=supervisedTrajectory;
  const context=t?.policy==='learner'?`${number(t.checkpoint)} offline optimizer updates`:t?.policy==='prior_stream'?'Historical frozen RL checkpoint; different training budget and data':'Reference policy; no learning';
  const mode=['learner','prior_stream'].includes(t?.policy)?(t.mode==='greedy'?'greedy':'ε = 0.1'):'own reference policy';
  const description=t?`${supervisedPolicyLabel(t.policy)} · ${context} · ${mode}. ${supervisedPanelLabel(t.panel)} · seed ${t.seed} · map ${t.map_seed}. Saved episode selected by the protocol, not its outcome.`:'';
  drawRecordedWorld('supervised',t,supervisedFrame,description,t?supervisedPolicyLabel(t.policy):'');
}
function renderSupervisedReferences(){
  if(!supervised)return;
  const panel=$('supervised-replay-panel').value || 'heldout',update=$('supervised-replay-controller').value==='learner'?Number($('supervised-replay-checkpoint').value):latestSupervisedUpdate();
  const learner=supervised.aggregate.find(r=>r.panel===panel && r.checkpoint===update && r.mode===supervisedMode());
  const reference=policy=>supervised.references?.find(r=>r.panel===panel && r.policy===policy && (policy!=='prior_stream' || r.mode===supervisedMode()));
  const rows=[[learner,`Supervised at ${stepLabel(update)}`,'#a6e5c1'],[reference('prior_stream'),'Historical frozen RL','#b5a7d2'],[reference('random_actions'),'Random actions','#e7b985'],[reference('shortest_path'),'Shortest path','#98b8d1']];
  $('supervised-reference-title').textContent=supervisedPanelLabel(panel);
  $('supervised-reference-bars').innerHTML=rows.filter(([r,,])=>r).map(([r,label,color])=>`<div class="diagnostic-row"><span>${escape(label)}</span><div class="diagnostic-track">${Number.isFinite(r.success_rate)?`<div class="diagnostic-fill" style="width:${100*r.success_rate}%;background:${color}"></div>`:''}</div><strong>${percent(r.success_rate)}</strong></div>`).join('');
  $('supervised-reference-caption').textContent=`Same rollout panel and visible state. Neural policies use ${supervisedMode()==='greedy'?'greedy actions':'ε = 0.1'}; random and planner retain their own policies. ${panel==='train'?'The historical RL reference is only evaluated on fresh layouts. ':''}Episode counts: ${rows.filter(([r])=>r).map(([r,label])=>`${label}: ${number(r.episodes)}`).join(' · ')}.`;
}
$('supervised-epsilon').addEventListener('change',renderSupervisedComparison);
for(const id of ['supervised-replay-controller','supervised-replay-checkpoint','supervised-replay-panel','supervised-replay-seed'])$(id).addEventListener('change',selectSupervisedTrajectory);
$('supervised-replay-play').addEventListener('click',()=>{if(supervisedTimer){stopSupervisedPlayback();return;}if(!supervisedTrajectory)return;if(supervisedFrame>=supervisedTrajectory.steps.length)supervisedFrame=0;$('supervised-replay-play').textContent='Ⅱ Pause';$('supervised-replay-play').setAttribute('aria-label','Pause saved supervised trajectory');supervisedTimer=setInterval(()=>{supervisedFrame=Math.min(supervisedFrame+1,supervisedTrajectory.steps.length);$('supervised-replay-scrub').value=String(supervisedFrame);drawSupervisedWorld();if(supervisedFrame>=supervisedTrajectory.steps.length)stopSupervisedPlayback();},160);});
$('supervised-replay-reset').addEventListener('click',()=>{stopSupervisedPlayback();supervisedFrame=0;$('supervised-replay-scrub').value='0';drawSupervisedWorld();});
$('supervised-replay-scrub').addEventListener('input',event=>{stopSupervisedPlayback();supervisedFrame=Number(event.target.value);drawSupervisedWorld();});

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
  const t=competenceTrajectory;
  const mode=t?.policy==='learner'?`${t.mode==='greedy'?'greedy':'ε = 0.1'} · ${number(t.checkpoint)} training steps`:'reference policy';
  const description=t?`${supportLabel(t.condition)} · ${policyLabel(t.policy)} · ${mode}. ${panelLabel(t.panel)} · seed ${t.seed} · map ${t.map_seed}. This saved episode illustrates behavior; the panel-wide evaluation measures success.`:'';
  drawRecordedWorld('competence',t,competenceFrame,description,t?`${supportLabel(t.condition)}, ${policyLabel(t.policy)}`:'');
}

function drawRecordedWorld(prefix,t,replayFrame,description,label) {
  const canvas=$(`${prefix}-world-canvas`),ctx=canvas.getContext('2d'),size=canvas.width;
  ctx.clearRect(0,0,size,size);ctx.fillStyle='#0a100c';ctx.fillRect(0,0,size,size);
  $(`${prefix}-step-diagnostics`).innerHTML='';
  if(!t) {
    ctx.fillStyle='#89988e';ctx.font='17px sans-serif';ctx.textAlign='center';ctx.fillText('No saved replay for this selection',size/2,size/2);
    $(`${prefix}-world-caption`).textContent='No episode is substituted for a missing recording.';
    $(`${prefix}-replay-step`).textContent='—';$(`${prefix}-replay-outcome`).textContent='No trajectory';$(`${prefix}-replay-explanation`).textContent='';return;
  }
  const n=t.grid_size,cell=size/n,step=replayFrame?t.steps[replayFrame-1]:null;
  const pos=step?.position || t.start,pellets=step?.pellets || t.pellets_initial;
  for(let r=0;r<n;r++)for(let c=0;c<n;c++){ctx.strokeStyle='#1b2a20';ctx.lineWidth=1;ctx.strokeRect(c*cell+.5,r*cell+.5,cell-1,cell-1);}
  for(const [r,c] of t.walls){ctx.fillStyle='#2c3b31';ctx.fillRect(c*cell+2,r*cell+2,cell-4,cell-4);}
  ctx.strokeStyle='#6e9c7c';ctx.lineWidth=2;ctx.setLineDash([3,5]);ctx.beginPath();
  [t.start,...t.steps.slice(0,replayFrame).map(s=>s.position)].forEach(([r,c],i)=>{if(i===0)ctx.moveTo((c+.5)*cell,(r+.5)*cell);else ctx.lineTo((c+.5)*cell,(r+.5)*cell);});
  ctx.stroke();ctx.setLineDash([]);
  for(const [r,c] of pellets){ctx.beginPath();ctx.arc((c+.5)*cell,(r+.5)*cell,cell*.105,0,Math.PI*2);ctx.fillStyle='#e7b985';ctx.fill();}
  ctx.beginPath();ctx.arc((pos[1]+.5)*cell,(pos[0]+.5)*cell,cell*.23,0,Math.PI*2);ctx.fillStyle='#a6e5c1';ctx.fill();ctx.strokeStyle='#d8fce7';ctx.lineWidth=2;ctx.stroke();
  const arrows=['↑','↓','←','→'];
  $(`${prefix}-world-caption`).textContent=`Rule A · key→move: ${(t.action_mapping || [0,1,2,3]).map((move,action)=>`${arrows[action]}→${arrows[move]}`).join('  ')}`;
  $(`${prefix}-replay-step`).textContent=`${replayFrame} / ${t.steps.length}`;
  $(`${prefix}-replay-outcome`).textContent=replayFrame===t.steps.length?(t.success?'Collected all pellets':'Episode ended without success'):`${pellets.length} pellet${pellets.length===1?'':'s'} left`;
  $(`${prefix}-replay-explanation`).textContent=description;
  const decision=step || t.steps[0], decisionNumber=replayFrame || 1;
  if(decision) {
    const metric=[];
    if(Number.isFinite(decision.action))metric.push(`Chosen action ${arrows[decision.action] || decision.action}`);
    if(step && Number.isFinite(step.reward))metric.push(`Reward ${decimal(step.reward)}`);
    if(Number.isFinite(decision.action_regret))metric.push(`Action regret ${decimal(decision.action_regret)}`);
    $(`${prefix}-step-diagnostics`).innerHTML=`<p><strong>Decision for step ${number(decisionNumber)}</strong> · ${metric.map(escape).join(' · ')}</p>`;
    if(decision.q_values?.length || decision.optimal_q_values?.length) {
      $(`${prefix}-step-diagnostics`).innerHTML+=`<div class="table-scroll"><table class="evidence-table"><thead><tr><th>Pre-action values</th>${arrows.map((a,i)=>`<th${i===decision.action?' class="chosen-action"':''}>${a}${i===decision.action?' · chosen':''}</th>`).join('')}</tr></thead><tbody>${[['Learned Q',decision.q_values],['Optimal Q',decision.optimal_q_values]].filter(([,values])=>values?.length).map(([label,values])=>`<tr><td>${label}</td>${values.map((v,i)=>`<td${i===decision.action?' class="chosen-action"':''}>${decimal(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p>Values are from before step ${number(decisionNumber)}; the grid shows ${replayFrame?'the position after that step':'the starting position'}. Q is discounted return. Regret compares the chosen action with the optimal choice for that state.</p>`;
    }
  }
  canvas.setAttribute('aria-label',`${label}, step ${replayFrame} of ${t.steps.length}, agent at row ${pos[0]}, column ${pos[1]}, ${pellets.length} pellets remaining.`);
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
  stopPlayback();stopCompetencePlayback();stopSupervisedPlayback();stopFixedPlayback();stopCoveragePlayback();
  const adaptationPath=$('study-run').value==='pilot_v1'?'../experiments/adaptation/pilot_v1/results.json':'data/adaptation.json';
  const results=await Promise.allSettled([getData(adaptationPath),getData('data/provenance.json'),getData('data/competence.json'),getData('data/supervised.json'),getData('data/fixed_targets.json'),getData('data/coverage.json'),getData('data/equal_support.json')]);
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
  if(results[3].status==='fulfilled')supervised=results[3].value;else {supervised=null;errors.push(results[3].reason.message);}
  renderSupervised();
  if(results[4].status==='fulfilled')fixed=results[4].value;else {fixed=null;errors.push(results[4].reason.message);}
  renderFixed();
  if(results[5].status==='fulfilled')coverageStudies.coverage=results[5].value;else {coverageStudies.coverage=null;errors.push(results[5].reason.message);}
  if(results[6].status==='fulfilled')coverageStudies.equal_support=results[6].value;else {coverageStudies.equal_support=null;errors.push(results[6].reason.message);}
  coverage=coverageStudies[$('coverage-study').value] || null;coverageConditions=coverageIsEqual()?['collected_unique','uniform_subset']:['exhaustive','collected_unique'];
  renderCoverage();
  if(errors.length) $('load-status').textContent=errors.join(' · ');
  else {
    const loaded=[coverageStudies.equal_support&&'Equal-size banks',coverageStudies.coverage&&'Experience coverage',fixed&&'Fixed-data targets',supervised&&'Exact targets',competence&&'Competence',adaptation&&'Adaptation',provenance&&'Provenance'].filter(Boolean);
    $('load-status').textContent=`${loaded.join(' + ') || 'No'} saved ${loaded.length===1?'study':'studies'} loaded · ${new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}`;
  }
  $('refresh').disabled=false;$('study-run').disabled=false;loadInProgress=false;
}
$('refresh').addEventListener('click',loadData);
$('study-run').addEventListener('change',loadData);
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopPlayback();stopCompetencePlayback();stopSupervisedPlayback();stopFixedPlayback();stopCoveragePlayback();}});
await loadData();
