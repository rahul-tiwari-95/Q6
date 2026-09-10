import { applyMode, bindModeToggle } from './app.js';

const $ = id => document.getElementById(id);
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => Number.isFinite(value) ? Math.round(value).toLocaleString() : '—';
const percent = value => Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : '—';
const precisePercent = value => Number.isFinite(value) ? `${(100 * value).toFixed(2)}%` : '—';
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
const signedDecimal = value => Number.isFinite(value) ? `${value>0?'+':''}${value.toFixed(3)}` : '—';
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
const coverageStudies={coverage:null,equal_support:null,panel_evaluation:null,bank_replication:null,map_replay:null,within_map:null,recorded_actions:null,constrained_bootstrap:null,logged_graph:null,familiar_starts:null,guided_collection:null};
const coverageIsEqual=()=>$('coverage-study').value==='equal_support';
const coverageDirection=()=>coverageIsEqual()?'uniform subset minus collected unique states':'collected unique states minus exhaustive states';
const coverageColors={exhaustive:'#a6e5c1',collected_unique:'#98b8d1',uniform_subset:'#e7b985',map_balanced:'#a6e5c1',within_map_uniform:'#c4aadf',recorded_actions:'#d6ba83',constrained_bootstrap:'#9bd1c9',logged_graph:'#aeb6f4',guided_collection:'#e7b985'};
const coverageLabel=condition=>({exhaustive:'Exhaustive states',collected_unique:'Collected unique states',uniform_subset:'Uniform subset',map_balanced:'Map-balanced replay',within_map_uniform:'Within-map uniform states',recorded_actions:'Recorded actions',constrained_bootstrap:'Constrained bootstrap',logged_graph:'Logged-graph exact targets',guided_collection:'Mixed collection',shared:'Shared references'}[condition] || condition);
const coverageMode=()=>$('coverage-epsilon').value;
const latestCoverageUpdate=()=>Math.max(0,...(coverage?.aggregate || []).map(r=>r.checkpoint).filter(Number.isFinite));
const signed=(value,scale=1,suffix='')=>Number.isFinite(value)?`${value>0?'+':''}${(value*scale).toFixed(1)}${suffix}`:'—';

function robustnessData(){return coverageStudies.panel_evaluation;}
const robustnessConditions=['collected_unique','uniform_subset','exhaustive'];
let robustnessTrajectory=null,robustnessFrame=0,robustnessTimer=null;
const robustnessMode=()=>$('robustness-epsilon').value;
const robustnessEligible=()=>robustnessData()?.descriptive_thresholds?.eligible===true && robustnessData()?.run?.status==='complete' && robustnessData()?.protocol?.smoke!==true && !(robustnessData()?.protocol?.deviations?.length);
const robustnessPanelLabel=panel=>panel==='all'?'All panels':`Panel ${Number(String(panel).split('_').at(-1))+1}`;
function robustnessPanels(){
  const study=robustnessData(),declared=study?.protocol?.panels;
  return Array.isArray(declared)?declared.map(p=>typeof p==='string'?p:p.id || p.panel).filter(Boolean):[...new Set((study?.aggregate || []).map(r=>r.panel).filter(p=>p!=='all'))].sort((a,b)=>Number(a.split('_').at(-1))-Number(b.split('_').at(-1)));
}
function renderRobustness(){
  stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();
  const study=robustnessData(),ready=Boolean(study?.aggregate?.length),run=study?.run || {};
  $('robustness-empty').hidden=ready;$('robustness-content').hidden=!ready;
  if(!ready){
    $('robustness-empty-title').textContent=run.status?.startsWith('inconsistent')?'Consistency check failed.':study?'No complete panel result available.':'No saved panel-robustness result yet.';
    $('robustness-empty-message').textContent=study?`Status: ${(run.status || 'unavailable').replaceAll('_',' ')}. ${run.stop_reason || study.provenance?.stop_reason || 'No completed result is substituted.'}`:'Earlier learning studies remain available through the study selector.';
    for(const id of ['robustness-success-chart','robustness-efficient-chart','robustness-steps-chart','robustness-difference-chart','robustness-outcomes'])$(id).innerHTML='';return;
  }
  $('robustness-study-label').textContent=`${run.id || study.protocol?.id || 'Saved evaluation'} · ${(run.status || 'unavailable').replaceAll('_',' ')}`;
  const eligible=robustnessEligible();
  $('robustness-note').textContent=run.status?.startsWith('inconsistent')?`Consistency check failed. No descriptive threshold or sign-count conclusion is available. ${run.stop_reason || study.provenance?.stop_reason || ''}`:run.status!=='complete'?'Incomplete evaluation. Inspect completed panel measurements; incomplete coverage cannot support the declared panel-wide summaries.':!eligible?'Smoke or protocol-deviation evaluation. Measurements check execution; threshold frequencies and panel sign counts are not research evidence.':'These are repeated evaluations of the same saved policies. Compare variation across fresh panels and paired route efficiency. Historical thresholds provide context only; no new competence gate, significance test, or equivalence claim is established.';
  $('robustness-stats').innerHTML=[[number(robustnessConditions.length),'Frozen training conditions','Each retains its saved weights'],[number(study.protocol?.seeds?.length),'Saved learner seeds per condition','Reused across every panel'],[number(robustnessPanels().length),'New evaluation panels','Same maps for every policy'],['None','New optimization or collection','This study only evaluates saved policies']].map(([value,label,detail])=>`<div class="stat-card"><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join('');
  $('robustness-method').textContent=`${number(run.wall_seconds)} seconds elapsed. Saved policies are evaluated without optimization, replay insertion, or new training collection. Earlier training studies remain separate, including their fresh panels and fit diagnostics.`;
  if(Number.isFinite(run.peak_rss_bytes))$('robustness-method').textContent+=` Peak process memory ${(run.peak_rss_bytes/1024**3).toFixed(2)} GiB.`;
  listItems('robustness-limitations',run.limitations);
  $('robustness-evidence-links').innerHTML='<a href="data/panel_evaluation.json" download>Download displayed data ↓</a>'+[['protocol_document','Study design'],['protocol','Saved protocol'],['report','Measured findings'],['evaluations','Raw episodes'],['paired_differences','Paired differences'],['provenance','Frozen-weight provenance'],['manifest','Artifact checksums']].flatMap(([key,label])=>{const path=study.artifacts?.[key];return typeof path==='string' && /^(docs|experiments)\/[a-zA-Z0-9_./-]+$/.test(path) && !path.split('/').includes('..')?[`<a href="../${escape(path)}">${label} ↗</a>`]:[];}).join('');
  renderRobustnessComparison();
}
function renderPanelChart(id,series,panels,{title,rates=false,reference=null,difference=false,axisLabel="Fixed evaluation panels · no training axis",groupLabel="evaluation panels",pointLabel=robustnessPanelLabel}={}){
  const width=500,height=265,p={l:48,r:18,t:24,b:43},values=series.flatMap(s=>s.rows.map(r=>r.value)).filter(Number.isFinite);
  const top=difference?Math.max(.01,...values.map(Math.abs))*1.12:rates?1:Math.max(1,...values)*1.1,bottom=difference?-top:0;
  const x=panel=>p.l+(panels.indexOf(panel)+.5)/Math.max(1,panels.length)*(width-p.l-p.r),y=value=>p.t+(top-value)/(top-bottom)*(height-p.t-p.b),fmt=difference?value=>signed(value,100,' pp'):rates?percent:value=>value.toFixed(1);
  let svg=`<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg"><title>${escape(title)} across ${escape(groupLabel)}</title>`;
  for(let i=0;i<=4;i++){const value=bottom+i*(top-bottom)/4;svg+=`<line x1="${p.l}" y1="${y(value)}" x2="${width-p.r}" y2="${y(value)}" stroke="#26302a"/><text x="${p.l-7}" y="${y(value)+4}" text-anchor="end" fill="#8a978f" font-size="9">${fmt(value)}</text>`;}
  if(Number.isFinite(reference)){svg+=`<line x1="${p.l}" y1="${y(reference)}" x2="${width-p.r}" y2="${y(reference)}" stroke="#8a978f" stroke-dasharray="4 4"/><text x="${width-p.r}" y="${y(reference)-5}" text-anchor="end" fill="#a4afa8" font-size="9">${percent(reference)} historical reference</text>`;}
  panels.forEach(panel=>{svg+=`<text x="${x(panel)}" y="${height-24}" text-anchor="middle" fill="#8a978f" font-size="10">${escape(pointLabel(panel).replace('Panel ','P'))}</text>`;});
  svg+=`<text x="${width/2}" y="${height-7}" text-anchor="middle" fill="#8a978f" font-size="10">${escape(axisLabel)}</text>`;
  for(const s of series){let segment=[];const segments=[];for(const panel of panels){const row=s.rows.find(r=>r.panel===panel);if(Number.isFinite(row?.value))segment.push(row);else if(segment.length){segments.push(segment);segment=[];}}if(segment.length)segments.push(segment);
    for(const points of segments){if(!difference)svg+=`<path d="${points.map((r,i)=>`${i?'L':'M'}${x(r.panel)},${y(r.value)}`).join(' ')}" stroke="${s.color}" stroke-width="2" fill="none"/>`;
      for(const r of points){const tooltip=`${s.label}, ${pointLabel(r.panel)}: ${fmt(r.value)}`;
        if(difference){const barWidth=Math.min(30,(width-p.l-p.r)/Math.max(1,panels.length)*.6);svg+=`<rect x="${x(r.panel)-barWidth/2}" y="${Math.min(y(0),y(r.value))}" width="${barWidth}" height="${Math.max(1,Math.abs(y(r.value)-y(0)))}" fill="${r.value>=0?'#a6e5c1':'#e7b985'}"><title>${escape(tooltip)}</title></rect>`;}
        else svg+=`<circle cx="${x(r.panel)}" cy="${y(r.value)}" r="3.5" fill="${s.color}"><title>${escape(tooltip)}</title></circle>`;
      }
    }
  }
  if(!values.length)svg+=`<text x="${width/2}" y="${height/2}" text-anchor="middle" fill="#8a978f" font-size="12">No complete measurements</text>`;
  $(id).innerHTML=svg+'</svg>';
}
function renderRobustnessComparison(){
  const study=robustnessData();if(!study?.aggregate?.length)return;
  const mode=robustnessMode(),panels=robustnessPanels(),rows=study.aggregate.filter(r=>r.mode===mode),panelRows=rows.filter(r=>r.panel!=='all'),eligible=robustnessEligible();
  const series=metric=>robustnessConditions.map(condition=>({label:coverageLabel(condition),color:coverageColors[condition],rows:panelRows.filter(r=>r.condition===condition).map(r=>({panel:r.panel,value:r[metric]}))}));
  renderPanelChart('robustness-success-chart',series('success_rate'),panels,{title:'Episode success',rates:true,reference:study.protocol?.descriptive_reference_lines?.success});
  renderPanelChart('robustness-efficient-chart',series('efficient_success_rate'),panels,{title:'Efficient episode success',rates:true,reference:study.protocol?.descriptive_reference_lines?.efficient_success});
  renderPanelChart('robustness-steps-chart',series('mean_steps'),panels,{title:'Mean episode steps'});
  $('robustness-legend').innerHTML=robustnessConditions.map(c=>`<span><i style="background:${coverageColors[c]}"></i>${coverageLabel(c)}</span>`).join('');
  $('robustness-mode-note').textContent=`Each point pools the same saved learner seeds on one panel using ${mode==='greedy'?'greedy actions':'ε = 0.1 exploration'}. Lines connect evaluation groups; they do not show learning. Efficient success means success within twice the shortest-path length, with all episodes in the denominator. The dashed 70% / 80% lines are historical references only. Threshold frequency counts remain greedy when this mode changes.`;
  $('robustness-pooled-caption').textContent='Pooled values use all completed new panels. Ranges are the minimum and maximum of condition-level panel means, not confidence intervals. The same networks appear in every panel.';
  $('robustness-outcomes').innerHTML=robustnessConditions.map(condition=>{const pooled=rows.find(r=>r.condition===condition && r.panel==='all'),perPanel=panelRows.filter(r=>r.condition===condition);return `<div class="robustness-outcome-card"><strong style="color:${coverageColors[condition]}">${coverageLabel(condition)}</strong>${[['Success','success_rate',percent],['Efficient success','efficient_success_rate',percent],['Mean steps','mean_steps',value=>Number.isFinite(value)?value.toFixed(1):'—']].map(([label,key,format])=>{const values=perPanel.map(r=>r[key]).filter(Number.isFinite);return `<div><span>${label}</span><b>${format(pooled?.[key])}</b><small>Panel range ${values.length?`${format(Math.min(...values))}–${format(Math.max(...values))}`:'—'}</small></div>`;}).join('')}</div>`;}).join('');
  const pairs=(study.paired_differences?.aggregate || []).filter(r=>r.mode===mode),difference=pairs.filter(r=>r.comparison==='uniform_minus_collected' && r.panel!=='all');
  renderPanelChart('robustness-difference-chart',[{label:'Uniform subset minus collected states',color:'#a6e5c1',rows:difference.map(r=>({panel:r.panel,value:r.mean_seed_efficient_success_rate_delta}))}],panels,{title:'Paired efficient-success difference',difference:true});
  const summary=study.robustness?.per_comparison?.find(r=>r.comparison==='uniform_minus_collected' && r.mode===mode)?.metrics?.efficient_success_rate_delta;
  $('robustness-difference-caption').textContent=mode!=='greedy'?'Paired differences were declared for greedy episodes only. Switch to greedy to inspect the route-advantage comparison; exploration curves and replays remain available.':eligible && summary?`${mode==='greedy'?'Greedy':'ε = 0.1'} uniform subset minus collected states: positive in ${number(summary.positive_panels)} panels, negative in ${number(summary.negative_panels)}, tied in ${number(summary.zero_panels)}. Panel-mean differences range ${signed(summary.minimum,100,' pp')} to ${signed(summary.maximum,100,' pp')}. These are descriptive counts, not independent training replications or a significance test.`:'Panel sign counts are not evidence for this smoke, deviating, inconsistent, incomplete, or missing-summary evaluation. Recorded paired differences remain inspectable.';
  $('robustness-threshold-table').innerHTML='<thead><tr><th>Condition</th><th>Panels with every seed ≥70% greedy success and above panel random</th><th>Panels with every seed ≥80% greedy efficient success</th></tr></thead><tbody>'+robustnessConditions.map(condition=>{const counts=study.descriptive_thresholds?.per_condition?.find(r=>r.condition===condition);return `<tr><td>${coverageLabel(condition)}</td><td>${eligible && counts?`${number(counts.success_reference_panels)} / ${number(counts.panels)}`:'Not eligible'}</td><td>${eligible && counts?`${number(counts.efficiency_reference_panels)} / ${number(counts.panels)}`:'Not eligible'}</td></tr>`;}).join('')+'</tbody>';
  const table=(id,selected,seed)=>{$(id).innerHTML=`<thead><tr><th>Condition</th><th>${seed?'Learner seed':'Panel'}</th><th>Success</th><th>Efficient success</th><th>Mean steps</th><th>No-op rate</th><th>Episodes</th></tr></thead><tbody>`+selected.map(r=>`<tr><td>${coverageLabel(r.condition)}</td><td>${seed?number(r.seed):robustnessPanelLabel(r.panel)}</td><td>${percent(r.success_rate)}</td><td>${percent(r.efficient_success_rate)}</td><td>${decimal(r.mean_steps)}</td><td>${percent(r.noop_rate)}</td><td>${number(r.episodes)}</td></tr>`).join('')+'</tbody>';};
  table('robustness-panel-table',rows,false);table('robustness-learner-table',(study.seed_results || []).filter(r=>r.mode===mode && r.panel==='all'),true);
  const comparisons=study.paired_differences?.comparisons || [],pairLabel=id=>{const c=comparisons.find(c=>c.id===id);return c?`${coverageLabel(c.right)} − ${coverageLabel(c.left)}`:id.replaceAll('_',' ');};
  $('robustness-paired-table').innerHTML='<thead><tr><th>Comparison</th><th>Panel</th><th>Success Δ</th><th>Efficient success Δ</th><th>Mean steps Δ</th><th>Mean seed no-op Δ</th></tr></thead><tbody>'+pairs.map(r=>`<tr><td>${escape(pairLabel(r.comparison))}</td><td>${robustnessPanelLabel(r.panel)}</td><td>${signed(r.mean_seed_success_rate_delta,100,' pp')}</td><td>${signed(r.mean_seed_efficient_success_rate_delta,100,' pp')}</td><td>${signed(r.mean_seed_mean_steps_delta)}</td><td>${signed(r.mean_seed_noop_rate_delta,100,' pp')}</td></tr>`).join('')+'</tbody>';
  if(!pairs.length)$('robustness-paired-table').innerHTML='<caption>No paired comparisons were saved for this action mode; the declared paired analysis uses greedy episodes.</caption>'+$('robustness-paired-table').innerHTML;
  selectRobustnessTrajectory();
}
function stopRobustnessPlayback(){if(robustnessTimer)clearInterval(robustnessTimer);robustnessTimer=null;$('robustness-replay-play').textContent='▶ Play';$('robustness-replay-play').setAttribute('aria-label','Play frozen-policy recording');}
function selectRobustnessTrajectory(){
  stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();const study=robustnessData(),rows=(study?.trajectories || []).filter(r=>r.policy!=='learner' || r.mode===robustnessMode());
  selectOptions('robustness-replay-panel',robustnessPanels().filter(p=>rows.some(r=>r.panel===p)).map(p=>[p,robustnessPanelLabel(p)]),robustnessPanels()[0]);
  const atPanel=rows.filter(r=>r.panel===$('robustness-replay-panel').value),controller=r=>r.policy==='learner'?r.condition:r.policy;
  selectOptions('robustness-replay-controller',[...new Set(atPanel.map(controller))].map(c=>[c,robustnessConditions.includes(c)?coverageLabel(c):policyLabel(c)]),'collected_unique');
  const available=atPanel.filter(r=>controller(r)===$('robustness-replay-controller').value);
  selectOptions('robustness-replay-seed',[...new Set(available.map(r=>r.seed))].sort((a,b)=>a-b).map(s=>[String(s),String(s)]),'0');
  robustnessTrajectory=available.find(r=>r.seed===Number($('robustness-replay-seed').value)) || null;robustnessFrame=0;$('robustness-replay-scrub').value='0';$('robustness-replay-scrub').max=robustnessTrajectory?.steps.length || 0;
  for(const id of ['robustness-replay-play','robustness-replay-reset','robustness-replay-scrub'])$(id).disabled=!robustnessTrajectory;
  drawRobustnessWorld();renderRobustnessReferences();
}
function drawRobustnessWorld(){const t=robustnessTrajectory;drawRecordedWorld('robustness',t,robustnessFrame,t?`${robustnessPanelLabel(t.panel)} · ${t.policy==='learner'?coverageLabel(t.condition):policyLabel(t.policy)} · seed ${t.seed} · map ${t.map_seed}. ${t.policy==='learner'?(t.mode==='greedy'?'Greedy actions.':'ε = 0.1 exploration.'):'Reference policy.'} Frozen policy; no learning. The first map of this panel was selected before outcomes.`:'',t?`${robustnessPanelLabel(t.panel)}, ${coverageLabel(t.condition)}`:'');}
function renderRobustnessReferences(){
  const study=robustnessData();if(!study)return;const panel=$('robustness-replay-panel').value;
  const rows=robustnessConditions.map(c=>[study.aggregate.find(r=>r.condition===c && r.panel===panel && r.mode===robustnessMode()),coverageLabel(c),coverageColors[c]]);
  for(const [policy,color] of [['random_actions','#b5a7d2'],['shortest_path','#ddd5b0']])rows.push([study.references?.find(r=>r.policy===policy && r.panel===panel),policyLabel(policy),color]);
  $('robustness-reference-title').textContent=robustnessPanelLabel(panel);
  $('robustness-reference-bars').innerHTML=rows.filter(([r])=>r).map(([r,label,color])=>`<div class="diagnostic-row"><span>${label}</span><div class="diagnostic-track"><div class="diagnostic-fill" style="width:${100*r.success_rate}%;background:${color}"></div></div><strong>${percent(r.success_rate)}</strong></div>`).join('');
  $('robustness-reference-caption').textContent=`Same selected panel; networks use ${robustnessMode()==='greedy'?'greedy actions':'ε = 0.1'}. References retain their own policies. No earlier-study scores are included.`;
}
$('robustness-epsilon').addEventListener('change',renderRobustnessComparison);
for(const id of ['robustness-replay-panel','robustness-replay-controller','robustness-replay-seed'])$(id).addEventListener('change',selectRobustnessTrajectory);
$('robustness-replay-play').addEventListener('click',()=>{if(robustnessTimer){stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();return;}if(!robustnessTrajectory)return;if(robustnessFrame>=robustnessTrajectory.steps.length)robustnessFrame=0;$('robustness-replay-play').textContent='Ⅱ Pause';$('robustness-replay-play').setAttribute('aria-label','Pause frozen-policy recording');robustnessTimer=setInterval(()=>{robustnessFrame=Math.min(robustnessFrame+1,robustnessTrajectory.steps.length);$('robustness-replay-scrub').value=String(robustnessFrame);drawRobustnessWorld();if(robustnessFrame>=robustnessTrajectory.steps.length)stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();},160);});
$('robustness-replay-reset').addEventListener('click',()=>{stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();robustnessFrame=0;$('robustness-replay-scrub').value='0';drawRobustnessWorld();});
$('robustness-replay-scrub').addEventListener('input',event=>{stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();robustnessFrame=Number(event.target.value);drawRobustnessWorld();});

const banksIsMapReplay=()=>$('coverage-study').value==='map_replay';
const banksIsWithinMap=()=>$('coverage-study').value==='within_map';
const banksIsRecorded=()=>$('coverage-study').value==='recorded_actions';
const banksIsConstrained=()=>$('coverage-study').value==='constrained_bootstrap';
const banksIsGuided=()=>$('coverage-study').value==='guided_collection';
const banksIsGraph=()=>$('coverage-study').value==='logged_graph';
const banksHasRecordedSupervision=()=>banksIsRecorded() || banksIsConstrained() || banksIsGraph();
const bankStudySpecs={
  guided_collection:{conditions:['constrained_bootstrap','guided_collection'],direction:'Mixed minus random',short:'Mixed'},
  logged_graph:{conditions:['constrained_bootstrap','logged_graph'],direction:'Logged graph minus constrained',short:'Logged graph'},
  bank_replication:{conditions:['collected_unique','uniform_subset'],direction:'Uniform minus collected',short:'Uniform'},
  map_replay:{conditions:['collected_unique','map_balanced'],direction:'Map-balanced minus collected',short:'Balanced'},
  within_map:{conditions:['collected_unique','within_map_uniform'],direction:'Within-map uniform minus collected',short:'Within-map'},
  recorded_actions:{conditions:['collected_unique','recorded_actions'],direction:'Recorded minus all-action collected',short:'Recorded'},
  constrained_bootstrap:{conditions:['recorded_actions','constrained_bootstrap'],direction:'Constrained minus recorded',short:'Constrained'},
};
const banksStudyKey=()=>bankStudySpecs[$('coverage-study').value]?$('coverage-study').value:'bank_replication';
const banksHasArchivedControl=()=>banksStudyKey()!=='bank_replication';
function banksData(){return coverageStudies[banksStudyKey()];}
const banksConditions=()=>bankStudySpecs[banksStudyKey()].conditions;
const banksDirection=()=>bankStudySpecs[banksStudyKey()].direction;
const banksTreatmentShort=()=>bankStudySpecs[banksStudyKey()].short;
const banksControlShort=()=>banksIsGuided()?'Random':banksIsGraph()?'Constrained':banksIsConstrained()?'Recorded':'Collected';
const banksLabel=condition=>banksIsGuided()?(condition==='constrained_bootstrap'?'Random collection · archived learner':condition==='guided_collection'?'Mixed collection · new learner':coverageLabel(condition)):banksHasArchivedControl() && condition===banksConditions()[0]?(banksIsGraph()?'Constrained bootstrap · archived baseline':banksIsConstrained()?'Recorded actions · archived baseline':banksIsRecorded()?'All-action collected · archived baseline':'Collected replay · archived baseline'):banksHasArchivedControl() && condition===banksConditions()[1]?`${coverageLabel(condition)} · new fit`:coverageLabel(condition);
let banksTrajectory=null,banksFrame=0,banksTimer=null;
const banksMode=()=>$('banks-epsilon').value;
const banksEligible=()=>banksData()?.descriptive_thresholds?.eligible===true && banksData()?.run?.status==='complete' && banksData()?.protocol?.smoke!==true && !(banksData()?.protocol?.deviations?.length);
const banksRecords=()=>banksData()?.banks || [];
const banksPanels=()=>(banksData()?.protocol?.panels || []).map(p=>typeof p==='string'?p:p.id);
function renderBanksStudyCopy(){
  const map=banksIsMapReplay(),within=banksIsWithinMap(),recorded=banksHasRecordedSupervision(),constrained=banksIsConstrained(),graph=banksIsGraph();
  $('banks-intro-eyebrow').textContent=graph?'SAME RECORDED GRAPH / EXACT FIXED TARGETS':constrained?'SAME RECORDED SUPERVISION / CONSTRAINED NEXT-ACTION TARGET':recorded?'SAME STATE REPLAY / RECORDED ACTION OUTCOMES':within?'SAME MAP QUOTAS / DIFFERENT INCLUDED STATES':map?'SAME COLLECTED STATES / CHANGED REPLAY DISTRIBUTION':'NEW TRAINING BANKS / REPEATED PAIRED COMPARISON';
  $('banks-intro-title').textContent=graph?'Does fitting exact logged-graph targets improve learning?':constrained?'Can a recorded-action constraint improve bootstrap targets?':recorded?'How much does all-action supervision contribute?':within?'Does state choice inside each map matter?':map?'Can the same experience teach more through balanced replay?':'Does the difference survive a new bank of experience?';
  $('banks-intro-copy').textContent=graph?'Both conditions use the same recorded states, actions, rewards, endings, successors and replay stream. The treatment solves the finite recorded graph before training and fits fixed exact labels. The archived constrained-bootstrap control learned moving targets using neural successor values. Exact logged-graph labels are not full-world optimal values. All treatment fits finish before final support-fit diagnostics. Both final policies are reevaluated on new panels with all four actions available.':constrained?'Both conditions replay identical recorded states and supervise identical logged actions. The archived recorded-action control selects the next action from all four predictions. During training only, the treatment limits that selection to actions recorded at the logged successor. Network, replay, labels, loss and update budget are retained. Both final policies are reevaluated on new panels with all four actions available. No new competence gate is defined.':recorded?'Both conditions replay the same original collected states in the same order. The archived control learned from all four action outcomes at each sampled state. The new treatment applies loss only to actions actually recorded at that state, using recorded rewards, endings and successors. It receives fewer supervised action targets under the same state replay and optimizer budget. Archived controls are reevaluated on the same new panels. This is offline learning, not an online RL result.':within?'Each treatment selects states uniformly within each training map, matching that map’s exact collected-state count. The original sequence of maps and within-map selection ranks is preserved, including within-batch map diversity. Different states occupy those ranks. The included states and their clock or goal-distance composition change. The same network, initialization, Double DQN and update budget are retained. Archived controls are reevaluated on the same new panels as the new treatments. No new competence gate is defined.':map?'Each treatment uses exactly the archived collected states. Each update samples 64 distinct maps uniformly, then one supported state uniformly within each selected map. The archived baseline samples uniformly over all supported states, so maps with larger supports receive more training. The same network, initial weights, all-action Double DQN targets and update budget are retained. Only the treatments are trained again; archived baseline policies are reevaluated on the same new panels. This changes both map exposure and within-batch map diversity. No new competence gate is defined.':'Each bank pair contains a new collected-state bank and a uniform subset with the same number of states. Train the same Double DQN setup from the paired initializations, then evaluate final policies on shared fresh panels. Bank sizes can differ between pairs. Inspect each paired effect before the equal-bank average; no new competence gate is defined.';
  $('banks-flow').innerHTML=(graph?[['Same recorded graph','No new outcomes or actions'],['Solve exact graph labels','DP preparation before neural fitting'],['Same neural replay budget','Zero treatment neural successor queries'],['Final fit and fresh rollouts','Unrestricted four-action evaluation']]:constrained?[['Same recorded outcomes','Same state replay and action supervision'],['Constrain training argmax','Logged actions at the logged successor'],['Same main target budget','No additional action outcomes'],['Unrestricted evaluation','All four actions on new panels']]:recorded?[['Same collected support','Same state and map replay sequence'],['Recorded actions only','Mask unobserved action outcomes from loss'],['Same state-update budget','Fewer supervised action targets'],['Same new panels','Archived control versus new treatment']]:within?[['Same map quotas','Collected / new uniform states per map'],['Same replay map sequence','Matched exposure and batch diversity'],['New treatment fits','Archived controls; same initial weights'],['Same new panels','Final-policy comparison for every bank']]:map?[['Same collected states','No new collection or support enlargement'],['Change replay exposure','Uniform states / uniform map then state'],['New treatment fits','Paired archive initial weights and budget'],['Same new evaluation panels','Frozen baseline versus new treatment']]:[['New bank pairs','Collected / same-size uniform subset'],['Same learner setup','Paired initial weights, all actions, budget'],['Final policies only','Same new panels for every condition'],['Each bank’s result','Then an explicitly equal-bank average']]).map(([title,detail])=>`<li><strong>${title}</strong><small>${detail}</small></li>`).join('');
  $('banks-primary-definition').textContent=`${banksDirection()} efficient success, in percentage points. Efficient success means finishing within twice the planner’s route length, with failed episodes included in the denominator. The final card gives each bank equal weight. Individual reversals remain visible.`;
  $('banks-difference-chart').setAttribute('aria-label',`${banksDirection()} efficient-success differences by panel`);
  $('banks-exposure-panel').hidden=!(map || within || recorded);
  $('banks-actions-panel').hidden=!recorded;
  $('banks-bootstrap-panel').hidden=!constrained;
  $('banks-graph-panel').hidden=!graph;
  $('banks-actions-eyebrow').textContent=graph?'SAME RECORDED ACTION LABEL COUNT / EXACT TARGET VALUES':constrained?'SAME STATES / SAME RECORDED ACTION SUPERVISION':'SAME STATES / LESS ACTION SUPERVISION';
  $('banks-actions-definition').textContent=graph?'Both arms supervise the same distinct recorded actions. The new treatment fits exact values of the logged graph, with equal state weighting. It adds no observed action outcome. Nonterminal exact labels are looked up directly during fitting, with zero neural successor queries; graph preparation is counted separately.':constrained?'Both arms use the same distinct recorded actions at each state. Loss averages those actions within each state, then averages states equally. The constraint changes only training-time next-action selection; it adds no observed outcome and does not restrict fresh evaluation actions.':'A state may have one, two, three, or four distinct recorded actions. Loss averages recorded actions within each state, then averages the sampled states equally. Only those outcomes contribute to the treatment loss. Four Q predictions still support ordinary Double DQN action selection; they are not four observed outcomes.';
  $('banks-composition-help').textContent=graph?'Coverage counts describe included current states. The all-action edge census describes the bank only; it is not the treatment’s training-query count. Exact labels use the recorded graph only. During treatment fitting, no next-state network predictions are queried.':recorded?'Coverage counts describe included current states. The all-action edge census describes the bank only; it is not the treatment’s training-query count. Treatment targets use recorded action edges. Four Q predictions for next-action argmax are model estimates, not observed outcomes.':'Coverage counts describe included current states. A detached successor query can reach outside a bank without adding that state to its direct training support. Both conditions still receive all four action transitions.';
  $('banks-exposure-eyebrow').textContent=graph?'SAME STATE REPLAY / SAME ACTION-TARGET PRESENTATIONS':constrained?'SAME STATE EXPOSURE / SAME RECORDED ACTION TARGET COUNTS':recorded?'SAME STATE EXPOSURE / FEWER ACTION TARGETS':within?'DIFFERENT INCLUDED STATES / MATCHED MAP EXPOSURE':'SAME INCLUDED STATES / DIFFERENT REPLAY EXPOSURE';
  $('guided-collection-panel').hidden=!banksIsGuided();$('guided-collection-replay-panel').hidden=!banksIsGuided();
  if(banksIsGuided()){
    $('banks-intro-eyebrow').textContent='CHANGED COLLECTION / SAME CONSTRAINED-DDQN LEARNER';
    $('banks-intro-title').textContent='Do better recorded journeys improve fresh-map learning?';
    $('banks-intro-copy').textContent='The control uses archived random collection and its final constrained-DDQN learners. The treatment replaces half the complete collection episodes with greedy routes from one frozen historical collector, then trains the same network and learner procedure. Episode allocations match; steps, unique states, recorded actions and replay can change. Fresh actions remain unrestricted. This is frozen-policy collection followed by offline learning, not online RL.';
    $('banks-flow').innerHTML=[['Same complete-episode allocation','Random 16 / random 8 + guided 8 per map'],['Frozen historical collector','One fixed bank-1 / seed-0 policy'],['Same learner and update budget','Recorded-action constrained Double DQN'],['New fresh-map evaluation','All four actions; final policies only']].map(([title,detail])=>`<li><strong>${title}</strong><small>${detail}</small></li>`).join('');
    $('banks-exposure-panel').hidden=false;$('banks-actions-panel').hidden=true;
    $('banks-exposure-eyebrow').textContent='DIFFERENT RECORDED SUPPORT / ACTUAL TRAINING EXPOSURE';
    $('banks-composition-help').textContent='Membership counts describe each arm’s own collected support. The all-action edge census describes the bank only; training uses recorded action outcomes and successor masks. Collection changes support, action coverage and replay together. It adds no full-world action outcomes or oracle targets.';
  }
}
function renderBanksExposure(){
  if(!banksHasArchivedControl())return;
  const study=banksData(),bank=Number($('banks-selected-bank').value),all=(study?.exposure?.per_map || []).filter(r=>r.bank_id===bank);
  selectOptions('banks-exposure-seed',[...new Set(all.map(r=>r.seed))].sort((a,b)=>a-b).map(seed=>[String(seed),String(seed)]),'0');
  const seed=Number($('banks-exposure-seed').value),rows=all.filter(r=>r.seed===seed),maps=[...new Set(rows.map(r=>r.map_seed))].sort((a,b)=>a-b),conditions=banksConditions();
  $('banks-exposure-caption').textContent=banksIsGraph()?'The state replay and action-target counts match at equal budgets. The new treatment looks up fixed recorded-graph labels; it does not query a neural successor during optimization. These exposure curves count sampled current states. Graph preparation and final fit diagnostics are separate workloads.':banksIsConstrained()?'State and action-target presentation counts are identical at matching budgets. The same recorded rewards, endings and successors supply both arms. These curves count sampled states; only next-action selection inside training targets changes. Fresh evaluation allows all four actions.':banksIsRecorded()?'State presentations are identical at matching budgets, including map exposure, state order and within-batch diversity. The solid baseline and dashed treatment curves overlap by design. This graph counts sampled states, not supervised action outcomes; the treatment’s action loss mask changes those separately.':banksIsWithinMap()?'Actual minibatch presentations, including repeats. Each map retains its exact quota and ordered replay exposure; different states are assigned to those presentations. With matching budgets, the solid baseline and dashed treatment map curves overlap by design. Baseline counts are archived; treatment counts are new. Clock and state-category exposure can change.':'Actual minibatch state presentations, including repeats. The plotted share is each map’s presentations divided by that fit’s total. Support membership is unchanged. The treatment selects 64 distinct maps per update, changing both map exposure and within-batch diversity. Baseline counts are archived; treatment counts are new.';
  if(banksIsGuided())$('banks-exposure-caption').textContent='Actual minibatch state presentations on each arm’s own collected support. The same main update/state-presentation budget need not produce identical local, state or map replay because support sizes and masks differ. Historical baseline exposure is reconstructed separately; new treatment exposure is saved. Four predictions are not four observed action outcomes.';
  $('banks-quota-check').hidden=!banksIsWithinMap();
  if(banksIsWithinMap()){
    const quotas=banksRecords().find(r=>r.bank_id===bank)?.quotas,quotaRows=quotas?.by_map || [],matched=quotaRows.filter(r=>r.identical===true && r.collected_states===r.treatment_states).length;
    const check=study?.provenance?.paired_map_sampling_consistency?.find(r=>r.bank_id===bank && r.seed===seed),verified=check?.complete===true && check?.map_digest_identical===true && check?.map_counts_identical===true && check?.local_digest_identical===true && check?.local_counts_identical===true;
    $('banks-quota-check').textContent=`Saved membership quota checks: ${quotaRows.length?`${number(matched)} / ${number(quotaRows.length)} maps match`:'unavailable'}. ${study?.protocol?.smoke?`Saved ${number(check?.compared_updates)}-update baseline-prefix replay check`:'Saved full-budget paired replay check'}: ${verified?'identical ordered map/local-index streams and counts':'not verified'}. ${study?.protocol?.smoke?`The plotted baseline still uses its full archived ${number(check?.baseline_updates)}-update budget.`:''}`;
  }
  $('banks-exposure-legend').innerHTML=conditions.map(c=>`<span><i style="background:${coverageColors[c]}"></i>${banksLabel(c)}</span>`).join('');
  selectOptions('banks-exposure-map',maps.map(map=>[String(map),`Map ${number(map)}`]),String(maps[0]));
  const selectedMap=Number($('banks-exposure-map').value);
  const groups=conditions.map(condition=>({condition,rows:rows.filter(r=>r.condition===condition).sort((a,b)=>a.map_seed-b.map_seed)}));
  $('banks-exposure-selected').textContent=`Map ${number(selectedMap)} · `+groups.map(g=>{const r=g.rows.find(r=>r.map_seed===selectedMap);return `${banksLabel(g.condition)}: ${number(r?.supported_states)} supported states, ${number(r?.presentations)} presentations (${precisePercent(r?.presentation_fraction)} of fit).`;}).join(' ');
  $('banks-exposure-totals').textContent=groups.map(g=>`${banksLabel(g.condition)}: ${number(g.rows[0]?.updates)} updates, ${number(g.rows.length?g.rows.reduce((sum,r)=>sum+r.presentations,0):undefined)} state presentations; source ${g.rows[0]?.source?.replaceAll('_',' ') || 'unavailable'}.`).join(' ')+(study?.protocol?.smoke?' Smoke budgets differ: these exposure shares are execution diagnostics, not an equal-budget learning comparison.':'');
  if(!rows.length){$('banks-exposure-selected').textContent='';$('banks-exposure-totals').textContent='';$('banks-exposure-chart').innerHTML='<p class="help-text">No saved exposure measurements.</p>';$('banks-exposure-table').innerHTML='';$('banks-exposure-summary').innerHTML='';$('banks-exposure-distribution').innerHTML='';$('banks-exposure-categories').innerHTML='';return;}
  const W=780,H=290,pad={left:60,right:20,top:22,bottom:47},plotW=W-pad.left-pad.right,plotH=H-pad.top-pad.bottom,max=Math.max(1/maps.length,...rows.map(r=>r.presentation_fraction))*1.08,x=i=>pad.left+(maps.length===1?plotW/2:i/(maps.length-1)*plotW),y=v=>pad.top+plotH-v/max*plotH;
  const grid=Array.from({length:5},(_,i)=>{const v=max*i/4;return `<line x1="${pad.left}" x2="${W-pad.right}" y1="${y(v)}" y2="${y(v)}" stroke="#ffffff12"/><text x="${pad.left-8}" y="${y(v)+4}" text-anchor="end" fill="#8a978f" font-size="11">${(v*100).toFixed(2)}%</text>`;}).join('');
  $('banks-exposure-chart').innerHTML=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg"><title>Bank ${bank}, seed ${seed}: fraction of actual training presentations by map</title>${grid}<line x1="${x(maps.indexOf(selectedMap))}" x2="${x(maps.indexOf(selectedMap))}" y1="${pad.top}" y2="${pad.top+plotH}" stroke="#ffffff60" stroke-dasharray="2 3"/><line x1="${pad.left}" x2="${W-pad.right}" y1="${y(1/maps.length)}" y2="${y(1/maps.length)}" stroke="#ddd5b0" stroke-dasharray="4 5"/><text x="${W-pad.right}" y="15" text-anchor="end" fill="#ddd5b0" font-size="10">Equal-map share ${(100/maps.length).toFixed(2)}%</text>${groups.map(g=>`<polyline points="${g.rows.map(r=>`${x(maps.indexOf(r.map_seed))},${y(r.presentation_fraction)}`).join(' ')}" fill="none" stroke="${coverageColors[g.condition]}" stroke-width="${(banksIsWithinMap() || banksHasRecordedSupervision()) && g.condition===banksConditions()[0]?3:1.5}" stroke-dasharray="${(banksIsWithinMap() || banksHasRecordedSupervision()) && g.condition===banksConditions()[1]?'5 4':'none'}"/>${g.rows.map(r=>`<circle cx="${x(maps.indexOf(r.map_seed))}" cy="${y(r.presentation_fraction)}" r="${r.map_seed===selectedMap?5:2}" fill="${coverageColors[g.condition]}"><title>${banksLabel(g.condition)} · Map ${number(r.map_seed)}: ${number(r.presentations)} presentations (${(100*r.presentation_fraction).toFixed(3)}%); ${number(r.supported_states)} supported states; ${number(r.unique_states_sampled)} distinct states sampled</title></circle>`).join('')}`).join('')}<text x="${pad.left}" y="${H-22}" fill="#8a978f" font-size="11">${number(maps[0])}</text><text x="${W-pad.right}" y="${H-22}" text-anchor="end" fill="#8a978f" font-size="11">${number(maps.at(-1))}</text><text x="${W/2}" y="${H-5}" text-anchor="middle" fill="#8a978f" font-size="11">Training map ID (same ordered maps in both conditions)</text></svg>`;
  const clocks=(study.exposure?.per_clock || []).filter(r=>r.bank_id===bank && r.seed===seed);
  $('banks-exposure-table').innerHTML='<thead><tr><th>Replay source</th><th>Remaining time</th><th>Supported states</th><th>Presentations</th><th>Share of fit</th><th>Distinct states sampled</th></tr></thead><tbody>'+clocks.map(r=>`<tr><td>${banksLabel(r.condition)}</td><td>${r.bucket==='all'?'All clocks':escape(r.bucket.replaceAll('_','–'))}</td><td>${number(r.supported_states)}</td><td>${number(r.presentations)}</td><td>${percent(r.presentation_fraction)}</td><td>${number(r.unique_states_sampled)}</td></tr>`).join('')+'</tbody>';
  const summaries=(study.exposure?.summaries || []).filter(r=>r.bank_id===bank && r.seed===seed);
  $('banks-exposure-summary').innerHTML=summaries.map(r=>`<p><strong>${banksLabel(r.condition)}</strong> · Map-share range ${precisePercent(r.map_fraction_min)}–${precisePercent(r.map_fraction_max)} · coefficient of variation ${decimal(r.map_fraction_cv)} · ${number(r.unique_states_sampled)} distinct states sampled.</p>`).join('');
  $('banks-exposure-distribution').innerHTML='<thead><tr><th>Replay source</th><th>Supported states</th><th>Never sampled</th><th>Presentations per state: min / median / 95th pct. / max</th><th>Direct off-support samples</th><th>Query-weighted outside-support successors</th></tr></thead><tbody>'+summaries.map(r=>{const d=r.state_count_distribution || {},q=r.successor_queries || {};return `<tr><td>${banksLabel(r.condition)}</td><td>${number(d.states)}</td><td>${number(d.zero_states)}</td><td>${number(d.minimum)} / ${decimal(d.median)} / ${decimal(d.q95)} / ${number(d.maximum)}</td><td>${number(r.outside_support_direct_samples)}</td><td>${percent(q.outside_support_fraction)} · ${number(q.outside_support_queries)} / ${number(q.nonterminal_queries)}</td></tr>`;}).join('')+'</tbody>';
  $('banks-exposure-categories').innerHTML='<thead><tr><th>Replay source</th><th>State category</th><th>Supported states</th><th>Presentations</th><th>Share of fit</th><th>Distinct states sampled</th></tr></thead><tbody>'+summaries.flatMap(r=>Object.entries(r.category_exposure || {}).map(([category,c])=>`<tr><td>${banksLabel(r.condition)}</td><td>${escape(category.replaceAll('_',' '))}</td><td>${number(c.supported_states)}</td><td>${number(c.presentations)}</td><td>${percent(c.presentation_fraction)}</td><td>${number(c.unique_states_sampled)}</td></tr>`)).join('')+'</tbody>';

}
for(const id of ['banks-exposure-seed','banks-exposure-map'])$(id).addEventListener('change',renderBanksExposure);

function renderBanksActions(){
  if(!banksHasRecordedSupervision())return;
  const study=banksData(),bank=Number($('banks-selected-bank').value),coverage=study?.action_coverage?.per_bank?.find(r=>r.bank_id===bank),rows=(study?.action_exposure?.per_seed || []).filter(r=>r.bank_id===bank);
  $('banks-action-count-bars').innerHTML=(coverage?.states_by_observed_actions || []).map(r=>`<div class="diagnostic-row"><span>${number(r.actions)} recorded ${r.actions===1?'action':'actions'}<small>${number(r.states)} / ${number(coverage.supported_states)} supported states</small></span><div class="diagnostic-track"><div class="diagnostic-fill" style="width:${coverage.supported_states?100*r.states/coverage.supported_states:0}%;background:${coverageColors.recorded_actions}"></div></div><strong>${percent(coverage.supported_states?r.states/coverage.supported_states:undefined)}</strong></div>`).join('');
  $('banks-actions-caption').textContent=coverage?`Bank ${bank}: ${number(coverage.recorded_edges)} distinct recorded state-action edges out of ${number(coverage.all_action_edges)} possible supported-state action edges (${precisePercent(coverage.observed_action_fraction)}). ${number(coverage.collector_steps)} archived collector steps; ${number(coverage.duplicate_records)} repeated records. Repeated visits do not create extra distinct action labels. ${number(coverage.nonterminal_recorded_edges)} recorded edges are nonterminal and ${number(coverage.terminal_recorded_edges)} are terminal. Recorded nonterminal successors: ${number(coverage.nonterminal_successors_in_support)} inside support, ${number(coverage.nonterminal_successors_outside_support)} outside. Saved collector source/copy hashes: ${coverage.source_sha256 && coverage.source_sha256===coverage.copied_sha256?'match':'not verified'}.`:'No saved recorded-action diagnostics. No missing action outcomes are inferred.';
  $('banks-action-target-table').innerHTML='<thead><tr><th>Condition / source</th><th>Seed</th><th>Updates</th><th>State presentations</th><th>Supervised action-target presentations</th><th>Action targets per state presentation</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${banksLabel(r.condition)}<br><small>${escape(r.source?.replaceAll('_',' ') || '')}</small></td><td>${number(r.seed)}</td><td>${number(r.updates)}</td><td>${number(r.state_presentations)}</td><td>${number(r.action_target_presentations)}</td><td>${decimal(r.state_presentations?r.action_target_presentations/r.state_presentations:undefined)}</td></tr>`).join('')+'</tbody>';
  $('banks-action-query-table').innerHTML='<thead><tr><th>Condition</th><th>Seed</th><th>Terminal action targets</th><th>Nonterminal action labels</th><th>Neural successor queries in optimizer</th><th>Outside-support target queries</th><th>Outside-support share</th><th>Saved query provenance</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${banksLabel(r.condition)}</td><td>${number(r.seed)}</td><td>${number(r.terminal_action_targets)}</td><td>${number(r.nonterminal_action_targets)}</td><td>${number(r.nonterminal_target_queries)}</td><td>${number(r.outside_support_target_queries)}</td><td>${percent(r.outside_support_fraction)}</td><td>${escape(r.query_provenance || 'Unavailable')}</td></tr>`).join('')+'</tbody>';
  $('banks-action-query-note').textContent=(banksIsGraph()?'Nonterminal labels and neural successor queries are distinct. The exact-label treatment makes zero neural successor queries during optimization, while the archived constrained-bootstrap control made neural queries. Dynamic-programming preparation over recorded graph edges is counted separately. Four predicted action values are not four supervised action outcomes.':banksIsConstrained()?'Both conditions use logged action edges and their successor observations. During training, the treatment’s next-action argmax is restricted to recorded actions at that successor. Fresh evaluation remains unrestricted. Four predicted Q values are not four supervised action outcomes. Terminal outcomes need no successor query.':'These counts describe actual target queries, including repeated training presentations. The recorded-action treatment uses logged edges and their successor observations. Double DQN still evaluates four predicted next-action Q values for argmax; those predictions are not four supervised action outcomes. Terminal outcomes need no successor query.')+(study?.protocol?.smoke?' Smoke compares a short treatment with the archived 30,000-update baseline; raw counts and outcomes are not equal-budget evidence.':'');
  const checks=(study?.provenance?.paired_map_sampling_consistency || []).filter(r=>r.bank_id===bank),keys=['complete','map_digest_identical','map_counts_identical','local_digest_identical','local_counts_identical','global_digest_identical','global_counts_identical'],verified=checks.filter(r=>keys.every(k=>r[k]===true)).length;
  $('banks-action-replay-check').textContent=checks.length?`Saved ${study?.protocol?.smoke?'treatment-length baseline-prefix':'full-budget'} replay identity: ${number(verified)} / ${number(checks.length)} learner streams verified for exact state, local-index and map sequences/counts. ${verified===checks.length?(banksIsGraph()?'The recorded action set is unchanged; exact fixed graph labels replace bootstrapped targets.':banksIsConstrained()?'Recorded-action supervision is unchanged; training target selection differs.':'Only action supervision changes within those replay streams.'):'Unverified streams do not establish the replay control.'}`:'Exact state replay verification unavailable.';
}

function renderBanksBootstrap(){
  if(!banksIsConstrained())return;
  const study=banksData(),bank=Number($('banks-selected-bank').value),diagnostics=study?.bootstrap_diagnostics || {},rows=(diagnostics.per_seed || []).filter(r=>r.bank_id===bank && r.condition==='constrained_bootstrap'),windows=(diagnostics.windows || []).filter(r=>r.bank_id===bank && r.condition==='constrained_bootstrap');
  const queries=rows.reduce((sum,r)=>sum+r.query_count,0),outside=rows.reduce((sum,r)=>sum+r.outside_count,0),mean=prefix=>queries?rows.reduce((sum,r)=>sum+r[prefix+'_sum'],0)/queries:undefined,range=prefix=>{const mins=rows.map(r=>r[prefix+'_min']).filter(Number.isFinite),maxs=rows.map(r=>r[prefix+'_max']).filter(Number.isFinite);return mins.length && maxs.length?`${signedDecimal(Math.min(...mins))} to ${signedDecimal(Math.max(...maxs))}`:'—';};
  $('banks-bootstrap-cards').innerHTML=[[percent(queries?outside/queries:undefined),'Constraint activated',`${number(outside)} / ${number(queries)} nonterminal training queries: unrestricted argmax outside logged successor actions`],[decimal(mean('online_gap')),'Mean online-value gap','Unrestricted online maximum minus constrained online selected value; query-weighted'],[signedDecimal(mean('target_delta')),'Mean signed DDQN target change',`Constrained minus unrestricted, same treatment weights. Per-query range ${range('target_delta')}`]].map(([value,label,detail])=>`<div class="stat-card"><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join('');
  $('banks-bootstrap-note').textContent='Activation and online-value gap use the online network. The DDQN target uses a separate target network, so its signed change can be positive, negative, or zero. These are descriptive training diagnostics on the treatment’s same weights, not a historical baseline curve or a competence gate. All means include every nonterminal training query, including inactive constraints.'+(study?.protocol?.smoke?' Smoke is execution evidence only; the archived baseline and new treatment have different update budgets.':'');
  $('banks-bootstrap-signs').textContent=rows.length?`Saved per-query target-change signs: ${number(rows.reduce((sum,r)=>sum+r.positive_target_deltas,0))} positive, ${number(rows.reduce((sum,r)=>sum+r.negative_target_deltas,0))} negative, ${number(rows.reduce((sum,r)=>sum+r.zero_target_deltas,0))} zero. Sign tolerance 10⁻¹²; these are descriptive query counts, not a significance test.`:'No saved target-change sign counts.';
  $('banks-bootstrap-table').innerHTML='<thead><tr><th>Treatment seed</th><th>Updates</th><th>Nonterminal queries</th><th>Unrestricted action outside mask</th><th>Activated fraction</th><th>Mean online gap</th><th>Online-gap range</th><th>Mean signed target Δ</th><th>Signed target-Δ range</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${number(r.seed)}</td><td>${number(r.updates)}</td><td>${number(r.query_count)}</td><td>${number(r.outside_count)}</td><td>${percent(r.outside_fraction)}</td><td>${decimal(r.online_gap_mean)}</td><td>${decimal(r.online_gap_min)} to ${decimal(r.online_gap_max)}</td><td>${signedDecimal(r.target_delta_mean)}</td><td>${signedDecimal(r.target_delta_min)} to ${signedDecimal(r.target_delta_max)}</td></tr>`).join('')+'</tbody>';
  const seeds=[...new Set(windows.map(r=>r.seed))].sort((a,b)=>a-b),colors=['#9bd1c9','#d6ba83','#c4aadf'],series=key=>seeds.map((seed,i)=>({label:`Treatment seed ${seed}`,color:colors[i%colors.length],rows:windows.filter(r=>r.seed===seed).map(r=>({checkpoint:r.checkpoint,value:r[key]}))}));
  renderUpdateChart('banks-bootstrap-activation-chart',series('outside_fraction'),windows.map(r=>r.checkpoint),true,'Constraint activation during treatment training');
  renderUpdateChart('banks-bootstrap-delta-chart',series('target_delta_mean'),windows.map(r=>r.checkpoint),false,'Signed DDQN target change during treatment training',{signedValues:true});
  $('banks-bootstrap-legend').innerHTML=seeds.map((seed,i)=>`<span><i style="background:${colors[i%colors.length]}"></i>Treatment seed ${number(seed)}</span>`).join('');
}

function renderBanksGraph(){
  if(!banksIsGraph())return;
  const study=banksData(),bank=Number($('banks-selected-bank').value),rows=(study?.fit_diagnostics?.per_seed || []).filter(r=>r.bank_id===bank),fmt=v=>Number.isFinite(v)?v.toPrecision(4):'—';
  const grouped=condition=>rows.filter(r=>r.condition===condition),mean=(condition,key)=>{const r=grouped(condition),n=r.reduce((sum,x)=>sum+x.states,0);return n?r.reduce((sum,x)=>sum+x.states*x[key],0)/n:undefined;};
  $('banks-graph-fit-cards').innerHTML=[['Mean state MAE','state_mean_abs_error',fmt],['Recorded-action agreement','restricted_action_agreement',percent],['Mean recorded-graph regret','mean_graph_regret',fmt]].map(([label,key,format])=>`<div class="stat-card"><span>${label}</span><div class="fixed-outcome-pair">${banksConditions().map(condition=>`<div><small>${banksLabel(condition)}</small><strong style="color:${coverageColors[condition]}">${format(mean(condition,key))}</strong></div>`).join('')}</div><small>Final weights · own full support · state-weighted across learner seeds</small></div>`).join('');
  $('banks-graph-fit-note').textContent='These are final fit diagnostics collected after every treatment fit finishes, alongside the archived controls. State MAE/MSE average recorded-action errors within each state before averaging states; edge MAE gives each observed edge equal weight. Agreement allows 1e-6 absolute graph-value tolerance; ties use the lowest action label. Regret compares the exact best recorded action with the graph value of the learner’s chosen recorded action. Unrestricted argmax outside the logged set has no assigned graph value. No full-world Q* fit, fresh-state regression, training-fit gate, or intermediate-checkpoint selection is implied.';
  $('banks-graph-fit-table').innerHTML='<thead><tr><th>Final policy</th><th>Seed</th><th>Support states</th><th>Recorded edges</th><th>State MAE</th><th>State MSE</th><th>Edge MAE</th><th>Max edge error</th><th>Recorded-action agreement</th><th>Mean graph regret</th><th>Max graph regret</th><th>Unrestricted argmax outside recorded actions</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${banksLabel(r.condition)}</td><td>${number(r.seed)}</td><td>${number(r.states)}</td><td>${number(r.observed_edges)}</td><td>${fmt(r.state_mean_abs_error)}</td><td>${fmt(r.state_mean_squared_error)}</td><td>${fmt(r.edge_mean_abs_error)}</td><td>${fmt(r.edge_max_abs_error)}</td><td>${percent(r.restricted_action_agreement)}</td><td>${fmt(r.mean_graph_regret)}</td><td>${fmt(r.max_graph_regret)}</td><td>${percent(r.unrestricted_argmax_outside_logged_fraction)}</td></tr>`).join('')+'</tbody>';
  const preparation=study?.logged_graph?.per_bank?.find(r=>r.bank_id===bank),fit=study?.fit_diagnostics;
  $('banks-graph-preparation').innerHTML=[['Recorded graph edges',number(preparation?.observed_edges),`${number(preparation?.supported_states)} supported states · bank ${number(bank)}`],['Preparation successor backups',number(preparation?.graph_preparation_successor_backups),`${number(preparation?.terminal_edges)} terminal edges need no backup`],['Max Bellman residual',fmt(preparation?.bellman_residual_max),`Saved float64 graph · tolerance ${fmt(preparation?.bellman_residual_tolerance)}`]].map(([label,value,detail])=>`<div class="stat-card"><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join('');
  $('banks-graph-preparation-note').textContent=`Exact recorded-graph labels are prepared once before fitting, then cast to float32 optimizer labels. Bank ${number(bank)}: ${number(preparation?.residual_validation_successor_lookups)} additional successor lookups validate the graph residual; ${fmt(preparation?.preparation_wall_seconds)} seconds of preparation. This work is separate from repeated optimizer labels and neural successor queries. Across this entire study, final fit diagnostics use ${number(fit?.inference_state_rows)} current-state inference rows and ${number(fit?.observed_edge_comparisons)} recorded-edge comparisons (${fmt(fit?.wall_seconds)} seconds), after all treatment fits. These diagnostic counts are neither optimizer targets nor rollout episodes.`;
}

function renderBanks(){
  stopBanksPlayback();stopFamiliarPlayback();const study=banksData(),run=study?.run || {},ready=Boolean(study?.aggregate?.length),guided=banksIsGuided(),mapReplay=banksHasArchivedControl(),within=banksIsWithinMap(),recorded=banksHasRecordedSupervision(),constrained=banksIsConstrained(),graph=banksIsGraph();
  renderBanksStudyCopy();
  $('banks-empty').hidden=ready;$('banks-content').hidden=!ready;
  if(!ready){$('banks-empty-title').textContent=run.status?.startsWith('inconsistent')?'Consistency check failed.':study?'No complete bank result available.':`No saved ${guided?'guided collection':graph?'logged graph':constrained?'constrained bootstrap':recorded?'recorded actions':within?'within-map states':mapReplay?'map-balanced replay':'bank-replication'} result yet.`;$('banks-empty-message').textContent=study?`Status: ${(run.status || 'unavailable').replaceAll('_',' ')}. ${run.stop_reason || study.provenance?.stop_reason || 'No completed result is substituted.'}`:'Earlier studies remain available through the study selector.';for(const id of ['banks-effects','banks-success-chart','banks-efficient-chart','banks-steps-chart','banks-difference-chart'])$(id).innerHTML='';return;}
  const eligible=banksEligible();
  $('banks-note').textContent=run.status?.startsWith('inconsistent')?`Consistency check failed. No replication conclusion is available. ${run.stop_reason || study.provenance?.stop_reason || ''}`:run.status!=='complete'?'Incomplete bank comparison. Inspect completed measurements; missing bank pairs cannot be replaced by the pooled result.':!eligible?`Smoke or protocol-deviation run. Measurements verify execution and do not establish ${guided?'a collection-policy benefit':graph?'an exact logged-graph comparison':constrained?'a constrained-bootstrap comparison':recorded?'a recorded-action comparison':within?'a within-map state-selection benefit':mapReplay?'a replay benefit':'bank replication'}.`:`${guided?'Compare the total effect of changed collection under the same constrained-DDQN learner. Complete-episode allocations match; interactions, support, recorded action targets and replay can differ. The frozen collector contributes additional historical experience and compute.':graph?'Both arms replay the same recorded graph. The new treatment fits exact fixed graph labels; the archived control is the constrained-bootstrap learner. These are logged-graph values, not full-world Q*.':constrained?'Both arms use recorded-action supervision and identical state replay; only training-time bootstrap selection is constrained. The control is the archived recorded-action learner, and fresh evaluation permits all four actions.':recorded?'State replay is matched; only recorded actions contribute treatment loss. The all-action control is archived, and both policies are evaluated on this study’s new panels.':within?'State counts and replay exposure are matched per map; the included states change. Archived collected baselines are compared with new within-map uniform fits.':mapReplay?'The same collected supports are reused. Archived collected baselines are compared with new map-balanced fits.':'Inspect each independently constructed bank pair and its paired learner seeds.'} The equal-bank mean summarizes these pairs; it does not hide individual directions, create new competence gates, or establish significance or equivalence.`;
  $('banks-stats').innerHTML=[[number(banksRecords().length),within?'Matched map-quota banks':mapReplay?'Reused collected banks':'New bank pairs',within?'Same per-map counts; different included states':mapReplay?'Same states in baseline and treatment':'Each uniform bank matches its collected bank’s size'],[number(study.protocol?.seeds?.length),'Learner seeds per condition','Paired within each bank'],[number(banksPanels().length),'Shared fresh panels','Final policies evaluated on the same maps'],[number(banksRecords().length*(mapReplay?1:2)*(study.protocol?.seeds?.length || 0)),mapReplay?'New treatment fits':'Declared model fits',mapReplay?'Baseline fits are reused from the archive':'Same Double DQN setup; final-only evaluation']].map(([value,label,detail])=>`<div class="stat-card"><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join('');
  const paired=study.paired_differences?.aggregate || [],mean=study.pooled?.paired?.find(r=>r.panel==='all' && r.mode==='greedy');
  const bankEffect=bank=>paired.find(r=>r.bank_id===bank.bank_id && r.panel==='all' && r.mode==='greedy');
  const effects=banksRecords().map(bank=>({label:`Bank ${bank.bank_id}`,value:bankEffect(bank)?.mean_seed_efficient_success_rate_delta,detail:guided?guidedSupportDetail(bank):`${number(bank.support_size)} states in each condition`}));
  effects.push({label:'Equal-bank mean',value:mean?.mean_bank_efficient_success_rate_delta,detail:'Each bank has the same weight'});
  $('banks-effects').innerHTML=effects.map(r=>`<div class="stat-card"><span>${r.label}</span><strong style="color:${!Number.isFinite(r.value)?'#8a978f':r.value>1e-12?'#a6e5c1':r.value < -1e-12?'#e7b985':'#8a978f'}">${signed(r.value,100,' pp')}</strong><small>${r.detail}</small></div>`).join('');
  const final=(bank,condition)=>study.aggregate.find(r=>r.bank_id===bank && r.condition===condition && r.panel==='all' && r.mode==='greedy');
  const summaryRow=(label,left,right,delta)=>`<tr><td>${label}</td><td>${percent(left?.success_rate)}</td><td>${percent(right?.success_rate)}</td><td>${percent(left?.efficient_success_rate)}</td><td>${percent(right?.efficient_success_rate)}</td><td>${decimal(left?.mean_steps)}</td><td>${decimal(right?.mean_steps)}</td><td>${signed(delta,100,' pp')}</td></tr>`;
  $('banks-summary-table').innerHTML=`<thead><tr><th>Bank scope</th><th>${banksControlShort()} success</th><th>${banksTreatmentShort()} success</th><th>${banksControlShort()} efficient</th><th>${banksTreatmentShort()} efficient</th><th>${banksControlShort()} steps</th><th>${banksTreatmentShort()} steps</th><th>Efficient Δ</th></tr></thead><tbody>`+banksRecords().map(bank=>summaryRow(`Bank ${bank.bank_id}`,final(bank.bank_id,banksConditions()[0]),final(bank.bank_id,banksConditions()[1]),bankEffect(bank)?.mean_seed_efficient_success_rate_delta)).join('')+summaryRow('Equal-bank mean',study.pooled?.aggregate?.find(r=>r.condition===banksConditions()[0] && r.panel==='all' && r.mode==='greedy'),study.pooled?.aggregate?.find(r=>r.condition===banksConditions()[1] && r.panel==='all' && r.mode==='greedy'),mean?.mean_bank_efficient_success_rate_delta)+'</tbody>';
  const values=effects.slice(0,-1).map(r=>r.value).filter(Number.isFinite);
  $('banks-effects-caption').textContent=eligible?`Bank-mean efficiency differences: ${number(values.filter(v=>v>1e-12).length)} positive, ${number(values.filter(v=>v < -1e-12).length)} negative, ${number(values.filter(v=>Math.abs(v)<=1e-12).length)} tied (zero tolerance 10⁻¹²). These descriptive bank counts are not a significance test. All values above use final greedy evaluations; the mode toggle below does not change this primary comparison.`:`This run is not eligible for a ${guided?'guided-collection':graph?'logged-graph':constrained?'constrained-bootstrap':recorded?'recorded-action':within?'within-map state-selection':mapReplay?'replay-benefit':'bank-replication'} interpretation. Saved primary measurements remain visible; no incomplete or smoke result is promoted by pooling.`;
  selectOptions('banks-selected-bank',banksRecords().map(bank=>[String(bank.bank_id),guided?`Bank ${bank.bank_id} · collection pair`:`Bank ${bank.bank_id} · ${number(bank.support_size)} states / condition`]),String(banksRecords()[0]?.bank_id));
  $('banks-method').textContent=`${number(run.wall_seconds)} seconds elapsed. ${mapReplay?`No new states are collected. Only ${graph?'logged-graph':constrained?'constrained-bootstrap':recorded?'recorded-action':within?'within-map uniform':'map-balanced'} treatments are fitted; baseline models and training exposure come from the ${graph?'constrained-bootstrap':constrained?'recorded-actions':'bank-replication'} archive. Both final policies are evaluated on this study’s new panels.`:'New collected banks and matching uniform subsets are fitted with the same learner setup.'} Only final policies are evaluated. Intermediate model snapshots and training losses are archived for reproduction; they are not intermediate evaluation results.`;
  if(guided){$('banks-method').textContent=`${number(run.wall_seconds)} seconds elapsed. Only the mixed-collection learners are fitted now; random-collection learner finals are archived. One historical constrained-DDQN collector is frozen throughout collection and shared by every bank. All final policies are evaluated on this study’s new fresh panels with all four actions available. Saved intermediate snapshots are not evaluation results.`;$('banks-stats').innerHTML=[[number(banksRecords().length),'Collection pairs','Random versus half random / half frozen DDQN'],[number(study.protocol?.seeds?.length),'Learner seeds per condition','Paired initializations; changed experience'],[number(banksPanels().length),'Shared fresh panels','Final policies; unrestricted action selection'],[number(banksRecords().length*(study.protocol?.seeds?.length || 0)),'New treatment fits','Historical baseline fits reused']].map(([value,label,detail])=>`<div class="stat-card"><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join('');}
  if(within)$('banks-method').textContent+=` New support construction: ${number(run.replacement_supports)} replacement supports from ${number(run.per_map_subset_draws)} per-map subset draws; ${number(run.collection_steps)} new collection steps.`;
  if(Number.isFinite(run.peak_rss_bytes))$('banks-method').textContent+=` Peak process memory ${(run.peak_rss_bytes/1024**3).toFixed(2)} GiB.`;
  listItems('banks-limitations',run.limitations);
  $('banks-evidence-links').innerHTML=`<a href="data/${banksStudyKey()}.json" download>Download displayed data ↓</a>`+[['protocol_document','Study design'],['protocol','Saved protocol'],['report','Measured findings'],['evaluations','Raw final episodes'],['training',mapReplay?'New treatment loss log':'Training loss log'],['losses',mapReplay?'New treatment loss log':'Training loss log'],['exposure','Actual training exposure'],['collection','Collection costs and components'],['collection_trajectories','Preselected collection routes'],['route_ceilings','Recorded-route ceilings'],['collector','Frozen collector provenance'],['bootstrap_diagnostics','Training target diagnostics'],['bootstrap_probes','Fixed training probes'],['logged_graph','Recorded-graph preparation'],['fit_diagnostics','Final recorded-graph fit'],['banks','Bank composition'],['sampling','Sampling provenance'],['paired_differences','Paired differences'],['manifest','Artifact checksums']].flatMap(([key,label])=>{const path=study.artifacts?.[key];return typeof path==='string' && /^(docs|experiments)\/[a-zA-Z0-9_./-]+$/.test(path) && !path.split('/').includes('..')?[`<a href="../${escape(path)}">${label} ↗</a>`]:[];}).join('');
  renderBanksComparison();
}
function renderBanksComparison(){
  const study=banksData();if(!study?.aggregate?.length)return;
  const bank=Number($('banks-selected-bank').value),mode=banksMode(),panels=banksPanels(),rows=study.aggregate.filter(r=>r.bank_id===bank && r.mode===mode && r.panel!=='all');
  const series=key=>banksConditions().map(condition=>({label:banksLabel(condition),color:coverageColors[condition],rows:rows.filter(r=>r.condition===condition).map(r=>({panel:r.panel,value:r[key]}))}));
  renderPanelChart('banks-success-chart',series('success_rate'),panels,{title:`Bank ${bank} success`,rates:true,reference:.7});renderPanelChart('banks-efficient-chart',series('efficient_success_rate'),panels,{title:`Bank ${bank} efficient success`,rates:true,reference:.8});renderPanelChart('banks-steps-chart',series('mean_steps'),panels,{title:`Bank ${bank} mean steps`});
  $('banks-legend').innerHTML=banksConditions().map(c=>`<span><i style="background:${coverageColors[c]}"></i>${banksLabel(c)}</span>`).join('');
  $('banks-mode-note').textContent=`Bank ${bank} · ${mode==='greedy'?'greedy actions':'ε = 0.1 exploration'}. Points are final-policy evaluations across panels, not a learning curve. Historical threshold counts and the top primary comparison remain greedy.`;
  const pairs=(study.paired_differences?.aggregate || []).filter(r=>r.bank_id===bank && r.mode===mode && r.panel!=='all');
  renderPanelChart('banks-difference-chart',[{label:banksDirection(),color:'#a6e5c1',rows:pairs.map(r=>({panel:r.panel,value:r.mean_seed_efficient_success_rate_delta}))}],panels,{title:`Bank ${bank} paired efficient-success difference`,difference:true});
  const values=pairs.map(r=>r.mean_seed_efficient_success_rate_delta).filter(Number.isFinite);
  $('banks-difference-caption').textContent=mode!=='greedy'?'Paired effects were declared for greedy episodes only. Exploration outcomes and replays remain available.':banksEligible()?`Bank ${bank}: ${banksDirection().toLowerCase()} efficiency is positive in ${number(values.filter(v=>v>1e-12).length)} panels, negative in ${number(values.filter(v=>v < -1e-12).length)}, tied in ${number(values.filter(v=>Math.abs(v)<=1e-12).length)}. Panel range ${values.length?`${signed(Math.min(...values),100,' pp')} to ${signed(Math.max(...values),100,' pp')}`:'—'}. Descriptive only; panels reuse the same learners.`:'Panel sign counts are not evidence for this ineligible run.';
  const selected=(study.seed_results || []).filter(r=>r.bank_id===bank && r.mode===mode);
  $('banks-learner-table').innerHTML='<thead><tr><th>Condition</th><th>Seed</th><th>Panel</th><th>Success</th><th>Efficient success</th><th>Mean steps</th><th>No-op rate</th><th>Episodes</th></tr></thead><tbody>'+selected.map(r=>`<tr><td>${banksLabel(r.condition)}</td><td>${number(r.seed)}</td><td>${robustnessPanelLabel(r.panel)}</td><td>${percent(r.success_rate)}</td><td>${percent(r.efficient_success_rate)}</td><td>${decimal(r.mean_steps)}</td><td>${percent(r.noop_rate)}</td><td>${number(r.episodes)}</td></tr>`).join('')+'</tbody>';
  $('banks-paired-table').innerHTML='<thead><tr><th>Bank</th><th>Panel</th><th>Success Δ</th><th>Efficient Δ</th><th>Mean steps Δ</th><th>Mean seed no-op Δ</th></tr></thead><tbody>'+(study.paired_differences?.aggregate || []).filter(r=>r.mode==='greedy').map(r=>`<tr><td>${number(r.bank_id)}</td><td>${robustnessPanelLabel(r.panel)}</td><td>${signed(r.mean_seed_success_rate_delta,100,' pp')}</td><td>${signed(r.mean_seed_efficient_success_rate_delta,100,' pp')}</td><td>${signed(r.mean_seed_mean_steps_delta)}</td><td>${signed(r.mean_seed_noop_rate_delta,100,' pp')}</td></tr>`).join('')+'</tbody>';
  const thresholds=study.descriptive_thresholds || {},counts=thresholds.per_condition || (thresholds.per_bank || []).flatMap(b=>(b.per_condition || []).map(c=>({...c,bank_id:b.bank_id})));
  $('banks-threshold-table').innerHTML='<thead><tr><th>Bank / condition</th><th>Panels: every seed ≥70% success and above random</th><th>Panels: every seed ≥80% efficient success</th></tr></thead><tbody>'+counts.filter(r=>r.bank_id===bank).map(r=>`<tr><td>${number(r.bank_id)} · ${banksLabel(r.condition)}</td><td>${banksEligible()?`${number(r.success_reference_panels)} / ${number(r.panels)}`:'Not eligible'}</td><td>${banksEligible()?`${number(r.efficiency_reference_panels)} / ${number(r.panels)}`:'Not eligible'}</td></tr>`).join('')+'</tbody>';
  renderBanksComposition();renderBanksExposure();renderBanksActions();renderBanksBootstrap();renderBanksGraph();selectBanksTrajectory();renderGuidedCollection();
}
function supportMapSVG(maps,color='#98b8d1'){
  if(!maps.length)return '<p class="help-text">No saved layout coverage.</p>';
  const cols=Math.min(16,maps.length),cell=24,w=cols*cell,h=Math.ceil(maps.length/cols)*cell;
  return `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg"><title>Included states across ${number(maps.length)} training layouts</title>${maps.map((r,i)=>`<rect x="${(i%cols)*cell}" y="${Math.floor(i/cols)*cell}" width="21" height="21" rx="2" fill="${color}" fill-opacity="${.08+.92*(Number.isFinite(r.coverage_rate)?Math.max(0,Math.min(1,r.coverage_rate)):0)}"><title>Map ${number(r.map_seed)}: ${number(r.visited_states)} / ${number(r.states)} included states (${percent(r.coverage_rate)})</title></rect>`).join('')}</svg>`;
}
function renderBanksComposition(){
  const bank=banksRecords().find(r=>r.bank_id===Number($('banks-selected-bank').value));if(!bank)return;
  const records=bank.coverage?.per_condition || [],intersection=bank.intersection || bank.coverage?.intersection || {},collection=bank.collection || {};
  $('banks-composition-caption').textContent=`Bank ${bank.bank_id}: ${number(bank.support_size)} included states per condition. Collection cost: ${number(collection.collection_episodes)} episodes and ${number(collection.collection_steps)} actual action steps. ${banksIsGraph()?'This is archived collection cost; no new collection is performed. Both conditions reuse identical collected states and recorded actions; the treatment fits exact recorded-graph labels.':banksIsConstrained()?'This is archived collection cost; no new collection is performed. Both conditions reuse identical collected states and recorded-action supervision.':banksIsRecorded()?'This is archived collection cost; no new collection is performed. Both conditions reuse identical collected states, but treatment action supervision is limited to recorded outcomes.':banksIsWithinMap()?'This is archived collection cost; no new collection is performed. New uniform states are drawn within each map, matching its exact collected quota; membership differs.':banksIsMapReplay()?'This is archived collection cost; no new collection is performed. Baseline and treatment have identical support.':'Uniform states are selected from the exhaustive universe; they are not additional collected visits.'} Support intersection: ${number(intersection.states)}; union ${number(intersection.union_states)}; Jaccard ${percent(intersection.jaccard)}.`;
  if(banksIsGuided())$('banks-composition-caption').textContent=`Bank ${bank.bank_id}: ${guidedSupportDetail(bank)}. Supports come from different collection-policy mixtures; sizes need not match. Support intersection ${number(intersection.states)}; union ${number(intersection.union_states)}; Jaccard ${percent(intersection.jaccard)}. Recorded route quality, actual interactions and fresh-policy behavior are reported separately.`;
  $('banks-composition-table').innerHTML='<thead><tr><th>Condition</th><th>Included states</th><th>State coverage</th><th>Winnable coverage</th><th>Goal-near coverage (physical distance ≤2, all clocks)</th><th>Nonterminal edges outside support</th></tr></thead><tbody>'+records.map(r=>`<tr><td>${banksLabel(r.condition)}</td><td>${number(r.unique_current_states)}</td><td>${percent(r.current_state_fraction)}</td><td>${percent(r.winnable_current_state_fraction)} · ${number(r.overall?.visited_winnable_states)} / ${number(r.overall?.winnable_states)}</td><td>${percent(r.goal_near_current_state_fraction)} · ${number(r.overall?.visited_goal_near_states)} / ${number(r.overall?.goal_near_states)}</td><td>${percent(r.successor_queries?.outside_support_fraction)} · ${number(r.successor_queries?.outside_support_nonterminal_transitions)} / ${number(r.successor_queries?.nonterminal_transitions)}</td></tr>`).join('')+'</tbody>';
  $('banks-map-heatmaps').innerHTML=records.map(r=>`<div><strong style="color:${coverageColors[r.condition]}">${banksLabel(r.condition)}</strong>${supportMapSVG(r.by_map || [],coverageColors[r.condition])}</div>`).join('');
  $('banks-time-bars').innerHTML=records.flatMap(r=>(r.by_time_bucket || []).map(row=>`<div class="diagnostic-row"><span>${banksLabel(r.condition)}<small>${row.time_bucket==='all'?'All clocks':escape(row.time_bucket)+' remaining'} · ${number(row.visited_states)} / ${number(row.states)}</small></span><div class="diagnostic-track"><div class="diagnostic-fill" style="width:${100*(row.coverage_rate || 0)}%;background:${coverageColors[r.condition]}"></div></div><strong>${percent(row.coverage_rate)}</strong></div>`)).join('');
}
function stopBanksPlayback(){stopGuidedCollectionPlayback();if(banksTimer)clearInterval(banksTimer);banksTimer=null;$('banks-replay-play').textContent='▶ Play';$('banks-replay-play').setAttribute('aria-label','Play bank-replication recording');}
function selectBanksTrajectory(){
  stopBanksPlayback();stopFamiliarPlayback();const study=banksData(),bank=Number($('banks-selected-bank').value),rows=(study?.trajectories || []).filter(r=>r.policy==='learner'?r.bank_id===bank && r.mode===banksMode():true);
  selectOptions('banks-replay-panel',banksPanels().filter(p=>rows.some(r=>r.panel===p)).map(p=>[p,robustnessPanelLabel(p)]),banksPanels()[0]);const atPanel=rows.filter(r=>r.panel===$('banks-replay-panel').value),controller=r=>r.policy==='learner'?r.condition:r.policy;
  selectOptions('banks-replay-controller',[...new Set(atPanel.map(controller))].map(c=>[c,banksConditions().includes(c)?banksLabel(c):policyLabel(c)]),banksConditions()[0]);const available=atPanel.filter(r=>controller(r)===$('banks-replay-controller').value);
  selectOptions('banks-replay-seed',[...new Set(available.map(r=>r.seed))].sort((a,b)=>a-b).map(s=>[String(s),String(s)]),'0');banksTrajectory=available.find(r=>r.seed===Number($('banks-replay-seed').value)) || null;banksFrame=0;$('banks-replay-scrub').value='0';$('banks-replay-scrub').max=banksTrajectory?.steps.length || 0;for(const id of ['banks-replay-play','banks-replay-reset','banks-replay-scrub'])$(id).disabled=!banksTrajectory;drawBanksWorld();renderBanksReferences();
}
function drawBanksWorld(){const t=banksTrajectory;drawRecordedWorld('banks',t,banksFrame,t?`${t.policy==='learner'?`Bank ${t.bank_id} · ${banksLabel(t.condition)} · final policy`:'Shared '+policyLabel(t.policy)} · ${robustnessPanelLabel(t.panel)} · seed ${t.seed} · map ${t.map_seed}. ${t.policy==='learner'?(t.mode==='greedy'?'Greedy actions.':'ε = 0.1 exploration.'):''} The first map of each panel was selected before outcomes; no learning occurs during replay.`:'',t?`${robustnessPanelLabel(t.panel)}, ${t.policy==='learner'?banksLabel(t.condition):policyLabel(t.policy)}`:'');}
function renderBanksReferences(){const study=banksData();if(!study)return;const bank=Number($('banks-selected-bank').value),panel=$('banks-replay-panel').value,rows=banksConditions().map(c=>[study.aggregate.find(r=>r.bank_id===bank && r.condition===c && r.panel===panel && r.mode===banksMode()),banksLabel(c),coverageColors[c]]);for(const [policy,color] of [['random_actions','#b5a7d2'],['shortest_path','#ddd5b0']])rows.push([study.references?.find(r=>r.policy===policy && r.panel===panel),policyLabel(policy),color]);$('banks-reference-title').textContent=`Bank ${bank} · ${robustnessPanelLabel(panel)}`;$('banks-reference-bars').innerHTML=rows.filter(([r])=>r).map(([r,label,color])=>`<div class="diagnostic-row"><span>${label}</span><div class="diagnostic-track"><div class="diagnostic-fill" style="width:${100*r.success_rate}%;background:${color}"></div></div><strong>${percent(r.success_rate)}</strong></div>`).join('');$('banks-reference-caption').textContent=`Selected bank and panel; networks use ${banksMode()==='greedy'?'greedy actions':'ε = 0.1'}. Random and planner references are shared across bank pairs and retain their own policies.`;}
$('banks-selected-bank').addEventListener('change',renderBanksComparison);$('banks-epsilon').addEventListener('change',renderBanksComparison);
for(const id of ['banks-replay-panel','banks-replay-controller','banks-replay-seed'])$(id).addEventListener('change',selectBanksTrajectory);
$('banks-replay-play').addEventListener('click',()=>{stopGuidedCollectionPlayback();if(banksTimer){stopBanksPlayback();stopFamiliarPlayback();return;}if(!banksTrajectory)return;if(banksFrame>=banksTrajectory.steps.length)banksFrame=0;$('banks-replay-play').textContent='Ⅱ Pause';$('banks-replay-play').setAttribute('aria-label','Pause bank-replication recording');banksTimer=setInterval(()=>{banksFrame=Math.min(banksFrame+1,banksTrajectory.steps.length);$('banks-replay-scrub').value=String(banksFrame);drawBanksWorld();if(banksFrame>=banksTrajectory.steps.length)stopBanksPlayback();stopFamiliarPlayback();},160);});
$('banks-replay-reset').addEventListener('click',()=>{stopBanksPlayback();stopFamiliarPlayback();banksFrame=0;$('banks-replay-scrub').value='0';drawBanksWorld();});$('banks-replay-scrub').addEventListener('input',event=>{stopBanksPlayback();stopFamiliarPlayback();banksFrame=Number(event.target.value);drawBanksWorld();});

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
const familiarData=()=>coverageStudies.familiar_starts;
const familiarConditions=['constrained_bootstrap','logged_graph'],familiarActionSets=['unrestricted','logged'];
const familiarActionLabel=value=>value==='logged'?'Recorded-action mask':value==='unrestricted'?'All four actions':value || 'Reference policy';
const familiarLabel=value=>value==='logged_q'?'Exact logged-Q policy':value==='shortest_path'?'Full-world shortest path':coverageLabel(value);
const familiarBanks=()=>[...new Set((familiarData()?.aggregate || []).map(r=>r.bank_id).filter(Number.isFinite))].sort((a,b)=>a-b);
const familiarEligible=()=>familiarData()?.robustness?.eligible===true && familiarData()?.run?.status==='complete' && familiarData()?.protocol?.smoke!==true && !(familiarData()?.protocol?.deviations?.length);
const familiarAll=row=>!row.block || row.block==='all';
const familiarSummary=(bank,condition,actionSet)=>(familiarData()?.aggregate || []).find(r=>r.bank_id===bank && r.condition===condition && r.action_set===actionSet && familiarAll(r));
const familiarTable=(id,rows,columns)=>{$(id).innerHTML='<thead><tr>'+columns.map(([,label])=>`<th>${label}</th>`).join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+columns.map(([key,,format=number])=>`<td>${escape(format(row[key]))}</td>`).join('')+'</tr>').join('')+'</tbody>';};
let familiarTrajectory=null,familiarFrame=0,familiarTimer=null;
function stopFamiliarPlayback(){if(familiarTimer)clearInterval(familiarTimer);familiarTimer=null;$('familiar-replay-play').textContent='▶ Play';$('familiar-replay-play').setAttribute('aria-label','Play familiar-start recording');}
function renderFamiliar(){
  stopFamiliarPlayback();const study=familiarData(),run=study?.run || {},ready=Boolean(study?.aggregate?.length);
  $('familiar-empty').hidden=ready;$('familiar-content').hidden=!ready;
  if(!ready){$('familiar-empty-title').textContent=run.status?.startsWith('inconsistent')?'Consistency check failed.':study?'No complete familiar-start result available.':'No saved familiar-start diagnostic yet.';$('familiar-empty-message').textContent=study?`Status: ${(run.status || 'unavailable').replaceAll('_',' ')}. ${run.stop_reason || study.provenance?.stop_reason || 'No other study is substituted.'}`:'Earlier measured studies remain available through the selector.';return;}
  $('familiar-note').textContent=run.status?.startsWith('inconsistent')?`Consistency check failed. ${run.stop_reason || study.provenance?.stop_reason || ''}`:run.status!=='complete'?'Incomplete familiar-start diagnostic. Missing cells do not establish a masking interaction.':!familiarEligible()?'Smoke or protocol-deviation diagnostic. These measurements verify execution; they do not establish a masking effect.':'Privileged familiar-start diagnostic: frozen weights, original training starts, no new learning or fresh maps. The recorded-action mask is historical information supplied at evaluation. These results do not establish fresh-map generalization, significance, or a new competence gate.';
  const banks=familiarBanks();selectOptions('familiar-bank',banks.map(b=>[String(b),`Bank ${b}`]),String(banks[0]));
  const interactions=banks.map(bank=>(study.paired_differences?.aggregate || []).find(r=>r.bank_id===bank && r.block==='all' && r.comparison==='interaction')?.efficient_success_delta);
  const pooled=study.pooled?.paired?.find(r=>r.block==='all' && r.comparison==='interaction')?.efficient_success_delta;
  const contrast=(bank,comparison)=>(bank==='Equal-bank mean'?study.pooled?.paired:study.paired_differences?.aggregate)?.find(r=>r.block==='all'&&r.comparison===comparison&&(bank==='Equal-bank mean'||String(r.bank_id)===bank))?.efficient_success_delta;
  $('familiar-effects').innerHTML=[...banks.map((bank,i)=>[String(bank),interactions[i]]),['Equal-bank mean',pooled]].map(([bank,value])=>`<div class="stat-card"><span>${bank==='Equal-bank mean'?bank:`Bank ${bank}`} · interaction</span><strong>${signed(value,100,' pp')}</strong><small>Mask effect · constrained: ${signed(contrast(bank,'mask_effect_constrained'),100,' pp')}</small><small>Mask effect · exact graph: ${signed(contrast(bank,'mask_effect_graph'),100,' pp')}</small><small>Masked graph minus constrained: ${signed(contrast(bank,'graph_minus_constrained_logged'),100,' pp')}</small><small>${bank==='Equal-bank mean'?'Each bank has the same weight; not independent learner replications':'Exact masking effect minus constrained masking effect'}</small></div>`).join('');
  const variation=study.robustness || {};$('familiar-effects-caption').textContent=familiarEligible()?`Interaction range ${signed(variation.minimum,100,' pp')} to ${signed(variation.maximum,100,' pp')}; ${number(variation.positive_banks)} positive banks, ${number(variation.negative_banks)} negative, ${number(variation.zero_banks)} tied (1e-12 tolerance). Descriptive only: banks share training maps and learner initializations.`:'Bank sign counts are not evidence for this ineligible diagnostic.';
  renderFamiliarComparison();
  $('familiar-method').textContent=`${number(run.wall_seconds)} seconds elapsed. Final networks remain frozen. Masks and graph references use each bank’s archived recorded information; world planning is a separate reference. All comparisons use the same original starts. Replays were chosen before outcomes. No previous fresh-map scores are mixed into this diagnostic.`;
  $('familiar-evidence-links').innerHTML='<a href="data/familiar_starts.json" download>Download displayed data ↓</a><a href="../docs/experiments/familiar_starts_protocol_v1.md">Study design ↗</a>'+[['protocol_document','Study design'],['protocol','Saved protocol'],['report','Measured findings'],['episodes','Raw familiar episodes'],['steps','All recorded decision steps'],['paired_differences','Same-start paired differences'],['reachability','Logged reachability'],['prediction_slices','Archived fit slices'],['manifest','Artifact checksums']].flatMap(([key,label])=>{const p=study.artifacts?.[key];return typeof p==='string'&&/^(docs|experiments)\/[a-zA-Z0-9_./-]+$/.test(p)&&!p.split('/').includes('..')?[`<a href="../${escape(p)}">${label} ↗</a>`]:[];}).join('');
}
function selectFamiliarTrajectory(){
  stopFamiliarPlayback();const bank=Number($('familiar-bank').value),rows=(familiarData()?.trajectories || []).filter(r=>r.bank_id===bank || r.bank_id==='shared'),controller=r=>r.policy==='learner'?r.condition:r.policy;
  selectOptions('familiar-replay-map',[...new Set(rows.map(r=>r.map_seed))].sort((a,b)=>a-b).map(m=>[String(m),`Map ${number(m)}`]),String(Math.min(...rows.map(r=>r.map_seed))));const atMap=rows.filter(r=>r.map_seed===Number($('familiar-replay-map').value));
  selectOptions('familiar-replay-controller',[...new Set(atMap.map(controller))].map(c=>[c,familiarLabel(c)]),familiarConditions[0]);const byController=atMap.filter(r=>controller(r)===$('familiar-replay-controller').value);
  selectOptions('familiar-replay-action-set',[...new Set(byController.map(r=>r.action_set || 'reference'))].map(a=>[a,familiarActionLabel(a)]),'unrestricted');const available=byController.filter(r=>(r.action_set || 'reference')===$('familiar-replay-action-set').value);
  selectOptions('familiar-replay-seed',[...new Set(available.map(r=>r.seed))].sort((a,b)=>a-b).map(seed=>[String(seed),String(seed)]),'0');familiarTrajectory=available.find(r=>String(r.seed)===$('familiar-replay-seed').value) || null;familiarFrame=0;$('familiar-replay-scrub').value='0';$('familiar-replay-scrub').max=familiarTrajectory?.steps.length || 0;for(const id of ['familiar-replay-play','familiar-replay-reset','familiar-replay-scrub'])$(id).disabled=!familiarTrajectory;drawFamiliarWorld();
}
function drawFamiliarWorld(){
  const t=familiarTrajectory;drawRecordedWorld('familiar',t,familiarFrame,t?`Original training map ${number(t.map_seed)} · bank ${t.bank_id} · ${familiarLabel(t.policy==='learner'?t.condition:t.policy)} · ${familiarActionLabel(t.action_set)}. Greedy frozen-policy diagnostic; this is not a fresh-map result.`:'','Familiar-start recording');
  $('familiar-step-diagnostics').innerHTML=$('familiar-step-diagnostics').innerHTML.replace('Regret compares the chosen action with the optimal choice for that state.','No full-world optimal-action values are supplied in this diagnostic.');
  const decision=t && (familiarFrame?t.steps[familiarFrame-1]:t.steps[0]);$('familiar-mask-diagnostics').innerHTML='';if(!decision)return;
  const mask=decision.recorded_mask,arrows=['↑','↓','←','→'];
  $('familiar-mask-diagnostics').innerHTML=`<p><strong>Pre-action recorded support:</strong> ${decision.current_supported===true?'Supported':decision.current_supported===false?'Outside support':'Unavailable'} · remaining time ${number(decision.remaining_before)}. ${decision.current_supported===false?'No recorded mask or graph values are assigned outside support.':''}</p>`;
  if(Array.isArray(mask) && decision.current_supported===true)$('familiar-mask-diagnostics').innerHTML+='<div class="table-scroll"><table class="evidence-table"><thead><tr><th>Recorded information before action</th>'+arrows.map(a=>`<th>${a}</th>`).join('')+'</tr></thead><tbody><tr><td>Action recorded</td>'+mask.map(v=>`<td>${v?'Yes':'No'}</td>`).join('')+'</tr><tr><td>Exact logged-graph Q</td>'+arrows.map((_,i)=>`<td>${decimal(decision.logged_q_values?.[i])}</td>`).join('')+'</tr></tbody></table></div>';
  $('familiar-mask-diagnostics').innerHTML+=`<p>Off-mask choice at a supported state: ${decision.off_mask_action===true?'Yes':decision.off_mask_action===false?'No':'—'}. Nonterminal support exit: ${decision.support_exit===true?'Yes':decision.support_exit===false?'No':'—'}. Logged success reachable before action: ${decision.logged_success_reachable===true?'Yes':decision.logged_success_reachable===false?'No':'—'}. Logged success reachability lost: ${decision.logged_reachability_lost===true?'Yes':decision.logged_reachability_lost===false?'No':'—'}. Recorded-action regret: ${decimal(decision.logged_value_regret)}. Logged values describe recorded actions only; they are not full-world Q*.</p>`;
}
$('familiar-bank').addEventListener('change',renderFamiliarComparison);
for(const id of ['familiar-replay-map','familiar-replay-controller','familiar-replay-action-set','familiar-replay-seed'])$(id).addEventListener('change',selectFamiliarTrajectory);
$('familiar-replay-play').addEventListener('click',()=>{if(familiarTimer){stopFamiliarPlayback();return;}if(!familiarTrajectory)return;if(familiarFrame>=familiarTrajectory.steps.length)familiarFrame=0;$('familiar-replay-play').textContent='Ⅱ Pause';$('familiar-replay-play').setAttribute('aria-label','Pause familiar-start recording');familiarTimer=setInterval(()=>{familiarFrame=Math.min(familiarFrame+1,familiarTrajectory.steps.length);$('familiar-replay-scrub').value=String(familiarFrame);drawFamiliarWorld();if(familiarFrame>=familiarTrajectory.steps.length)stopFamiliarPlayback();},160);});
$('familiar-replay-reset').addEventListener('click',()=>{stopFamiliarPlayback();familiarFrame=0;$('familiar-replay-scrub').value='0';drawFamiliarWorld();});$('familiar-replay-scrub').addEventListener('input',event=>{stopFamiliarPlayback();familiarFrame=Number(event.target.value);drawFamiliarWorld();});

function renderFamiliarComparison(){
  const study=familiarData();if(!study?.aggregate?.length)return;const bank=Number($('familiar-bank').value),rows=study.aggregate.filter(r=>r.bank_id===bank && familiarAll(r)),blocks=[...new Set(study.aggregate.map(r=>r.block).filter(b=>b && b!=='all'))].sort(),pairLabel=id=>study.paired_differences?.comparisons?.find(c=>c.id===id)?.label || id;
  const colors=['#98b8d1','#a6e5c1','#c4aadf','#e7b985'];
  $('familiar-stats').innerHTML=[[number(familiarBanks().length),'Archived banks'],[number(study.aggregate.filter(familiarAll).reduce((sum,r)=>sum+r.episodes,0)),'Learner episodes'],[number((study.references || []).filter(familiarAll).reduce((sum,r)=>sum+r.episodes,0)),'Reference episodes'],['Frozen','Weights · no new learning']].map(([value,label])=>`<div class="stat-card"><span>${label}</span><strong>${value}</strong></div>`).join('');
  $('familiar-matrix').innerHTML=familiarConditions.flatMap((condition,c)=>familiarActionSets.map((actionSet,a)=>{const r=familiarSummary(bank,condition,actionSet),color=colors[2*c+a];return `<div class="stat-card"><span>${familiarLabel(condition)}</span><h3>${familiarActionLabel(actionSet)}</h3><strong style="color:${color}">${percent(r?.success_rate)}</strong><small>Task success · ${number(r?.episodes)} familiar-start episodes</small><div class="diagnostic-row"><span>Efficient success</span><div class="diagnostic-track"><div class="diagnostic-fill" style="width:${Number.isFinite(r?.efficient_success_rate)?100*r.efficient_success_rate:0}%;background:${color}"></div></div><strong>${percent(r?.efficient_success_rate)}</strong></div><p class="help-text">Mean steps ${decimal(r?.mean_steps)} · blocked-step rate ${percent(r?.noop_rate)}</p></div>`;})).join('');
  $('familiar-matrix-note').textContent=`Bank ${number(bank)} · greedy final policies · all original starts in this diagnostic. Efficient success counts success within twice the full-world planner’s path length; failed episodes remain in its denominator and in mean steps. Masked behavior uses privileged recorded information. Successful-only route means are not substituted.`;
  const series=metric=>familiarConditions.flatMap((condition,c)=>familiarActionSets.map((actionSet,a)=>({label:`${familiarLabel(condition)} · ${familiarActionLabel(actionSet)}`,color:colors[2*c+a],rows:study.aggregate.filter(r=>r.bank_id===bank&&r.condition===condition&&r.action_set===actionSet&&r.block!=='all').map(r=>({panel:r.block,value:r[metric]}))})));
  const chartOptions={rates:true,axisLabel:'Original training-map blocks · no new learning',groupLabel:'familiar map blocks',pointLabel:b=>`Block ${Number(b.replace('block_',''))+1}`};
  renderPanelChart('familiar-success-chart',series('success_rate'),blocks,{...chartOptions,title:`Bank ${bank} familiar success`});renderPanelChart('familiar-efficient-chart',series('efficient_success_rate'),blocks,{...chartOptions,title:`Bank ${bank} familiar efficient success`});
  $('familiar-legend').innerHTML=series('success_rate').map(s=>`<span><i style="background:${s.color}"></i>${escape(s.label)}</span>`).join('');
  const pairColumns=[['bank_id','Bank'],['comparison','Same-start contrast',pairLabel],['starts','Paired starts'],['success_delta','Success Δ',v=>signed(v,100,' pp')],['efficient_success_delta','Efficient Δ',v=>signed(v,100,' pp')],['steps_delta','Mean steps Δ',signed],['noop_steps_delta','Mean blocked steps Δ',signed],['base_return_delta','Base return Δ',signedDecimal],['shaped_return_delta','Shaped return Δ',signedDecimal]];
  familiarTable('familiar-paired-table',(study.paired_differences?.aggregate || []).filter(familiarAll),pairColumns);
  familiarTable('familiar-layout-table',(study.paired_differences?.per_start || []).filter(r=>r.bank_id===bank),[...pairColumns.slice(0,1),['seed','Seed'],['map_seed','Original map'],...pairColumns.slice(1).filter(([key])=>key!=='starts')]);
  familiarTable('familiar-learner-table',(study.seed_results || []).filter(r=>r.bank_id===bank&&familiarAll(r)),[['condition','Frozen network',familiarLabel],['action_set','Action set',familiarActionLabel],['seed','Seed'],['episodes','Episodes'],['success_rate','Success',percent],['efficient_success_rate','Efficient success',percent],['mean_steps','Mean steps incl. failures',decimal],['noop_rate','Pooled blocked-step rate',percent],['base_return','Mean base return',decimal],['shaped_return','Mean shaped return',decimal]]);
  $('familiar-support-note').textContent='Off-mask actions count only decisions made while the current state belongs to the recorded support. A nonterminal successor outside support is a support exit; choosing an unrecorded action need not cause an exit, and terminal completion is never an exit. Off-support occupancy uses all episode decisions as denominator. All counts below are observed rollout decisions, not training queries. The unrestricted-argmax diagnostic uses each policy’s actual visited states; a mask also changes the route and therefore that state distribution.';
  familiarTable('familiar-support-table',rows,[['condition','Frozen network',familiarLabel],['action_set','Action set',familiarActionLabel],['off_mask_actions','Off-mask actions'],['supported_steps','Supported decisions'],['off_mask_rate','Off-mask / supported',percent],['unrestricted_argmax_outside_steps','Unrestricted argmax outside mask'],['unrestricted_argmax_outside_rate','Outside argmax / supported',percent],['off_mask_episodes','Episodes with off-mask choice'],['off_mask_episode_rate','Off-mask episode rate',percent],['support_exit_episodes','Episodes with support exit'],['episodes','All episodes'],['support_exits','Support exits'],['support_reentries','Reentries'],['unsupported_steps','Off-support decisions'],['evaluation_steps','All decisions'],['unsupported_step_fraction','Off-support / all decisions',percent],['logged_reachability_losses','Logged reachability losses'],['logged_reachable_steps','Reachable supported decisions'],['mean_logged_regret','Mean defined logged regret',decimal],['logged_regret_steps','Defined-regret decisions'],['restricted_action_agreement','Restricted action agreement',percent],['restricted_agreement_steps','Agreement decisions']]);
  renderFamiliarReferences();renderFamiliarSlices();selectFamiliarTrajectory();
}

function renderFamiliarReferences(){
  const study=familiarData(),bank=Number($('familiar-bank').value),refs=(study?.references || []).filter(r=>familiarAll(r)&&(r.bank_id===bank || r.bank_id==='shared')),reach=study?.reachability?.per_bank?.find(r=>r.bank_id===bank);
  const rows=refs.flatMap(r=>[[`${familiarLabel(r.policy)} · success`,r.success_rate],[`${familiarLabel(r.policy)} · efficient success`,r.efficient_success_rate]]);
  rows.push(['Recorded-graph success ceiling',reach?.success_ceiling],['Recorded-graph efficiency ceiling',reach?.efficient_success_ceiling]);
  $('familiar-reference-bars').innerHTML=rows.map(([label,value])=>`<div class="diagnostic-row"><span>${escape(label)}</span><div class="diagnostic-track">${Number.isFinite(value)?`<div class="diagnostic-fill" style="width:${100*value}%;background:#a6e5c1"></div>`:''}</div><strong>${percent(value)}</strong></div>`).join('');
  $('familiar-reference-note').textContent=`Bank ${number(bank)}: recorded success is reachable from ${number(reach?.success_reachable_starts)} / ${number(reach?.starts)} starts; success within twice the physical planner length is reachable from ${number(reach?.efficient_success_reachable_starts)} / ${number(reach?.starts)}. These ceilings come from successful-path reachability, not the exact logged-Q policy’s realized score. The logged-Q policy maximizes discounted return within recorded actions. Full-world shortest paths are shared references, not learner action advice.`;
}
function renderFamiliarSlices(){
  const study=familiarData(),bank=Number($('familiar-bank').value),axis=$('familiar-slice-axis').value,rows=(study?.prediction_slices?.per_slice || []).filter(r=>r.bank_id===bank&&r.axis===axis),fmt=v=>Number.isFinite(v)?v.toPrecision(4):'—';
  familiarTable('familiar-slices-table',rows,[['condition','Frozen network',familiarLabel],['seed','Seed'],['group','Slice',v=>String(v).replaceAll('_','–')],['states','Support states'],['observed_edges','Recorded edges'],['restricted_action_agreement','Recorded-action agreement',percent],['state_mean_signed_error','State signed bias',fmt],['state_mean_abs_error','State MAE',fmt],['state_mean_squared_error','State MSE',fmt],['state_mean_squared_offset','Mean squared state offset',fmt],['multiple_action_states','Multiple-action states'],['centered_state_mean_abs_error','Centered state MAE · multiple only',fmt],['centered_state_mean_squared_error','Centered state MSE · multiple only',fmt],['mean_target_top_two_gap','Logged target top-two gap · multiple only',fmt],['mean_unrestricted_prediction_gap','Unrestricted minus best-logged prediction',fmt],['mean_graph_regret','Mean recorded-graph regret',fmt]]);
  $('familiar-slices-note').textContent=`${axis==='original_start'?'These exploratory original-start slices, added after inspecting prior archived data,':'These predeclared descriptive slices'} reuse archived final predictions: ${number(study?.prediction_slices?.new_inference_rows)} new inference rows and no fitting. One-action states have automatic restricted agreement. Signed bias averages predicted minus logged-graph values within each state. Centered errors remove each state’s mean observed-action error and then average only multiple-action states. The top-two target gap also uses multiple-action states; ties remain included. Agreement allows 1e-6 absolute graph-value tolerance. Missing action targets remain undefined.`;
}
$('familiar-slice-axis').addEventListener('change',renderFamiliarSlices);

const guidedCollectionLabel=condition=>condition==='constrained_bootstrap'?'Random 16 · archived collection':'Random 8 + guided 8 · new collection';
const guidedSupportDetail=bank=>(bank.coverage?.per_condition || []).map(r=>`${r.condition==='constrained_bootstrap'?'Random':'Mixed'} ${number(r.unique_current_states)} states`).join(' · ');
let guidedCollectionTrajectory=null,guidedCollectionFrame=0,guidedCollectionTimer=null;
function stopGuidedCollectionPlayback(){if(guidedCollectionTimer)clearInterval(guidedCollectionTimer);guidedCollectionTimer=null;$('guided-collection-replay-play').textContent='▶ Play';$('guided-collection-replay-play').setAttribute('aria-label','Play collection recording');}
function selectGuidedCollectionTrajectory(){
  stopBanksPlayback();stopFamiliarPlayback();const study=banksData(),bank=Number($('banks-selected-bank').value),rows=(study?.collection_trajectories || []).filter(r=>r.bank_id===bank);
  selectOptions('guided-collection-replay-map',[...new Set(rows.map(r=>r.map_seed))].sort((a,b)=>a-b).map(m=>[String(m),`Map ${number(m)}`]),String(Math.min(...rows.map(r=>r.map_seed))));const atMap=rows.filter(r=>r.map_seed===Number($('guided-collection-replay-map').value));
  selectOptions('guided-collection-replay-condition',banksConditions().filter(c=>atMap.some(r=>r.condition===c)).map(c=>[c,guidedCollectionLabel(c)]),banksConditions()[0]);const selected=atMap.filter(r=>r.condition===$('guided-collection-replay-condition').value);
  selectOptions('guided-collection-replay-slot',[...new Set(selected.map(r=>r.repetition))].sort((a,b)=>a-b).map(slot=>[String(slot),`Slot ${slot} · ${slot===0?'shared random episode':'random / guided replacement'}`]),'8');guidedCollectionTrajectory=selected.find(r=>r.repetition===Number($('guided-collection-replay-slot').value)) || null;guidedCollectionFrame=0;$('guided-collection-replay-scrub').value='0';$('guided-collection-replay-scrub').max=guidedCollectionTrajectory?.steps.length || 0;for(const id of ['guided-collection-replay-play','guided-collection-replay-reset','guided-collection-replay-scrub'])$(id).disabled=!guidedCollectionTrajectory;drawGuidedCollectionWorld();
}
function drawGuidedCollectionWorld(){
  const t=guidedCollectionTrajectory;drawRecordedWorld('guided-collection',t,guidedCollectionFrame,t?`Bank ${number(t.bank_id)} · original map ${number(t.map_seed)} · slot ${number(t.repetition)} · ${t.policy_component==='guided'?'frozen greedy DDQN collector':'uniform random actions'} · ${t.source==='new_collection'?'new collection':'archived collection'}. This is a complete collection episode before learner fitting, not fresh-policy evaluation.`:'','Collection recording');
  const rows=(banksData()?.collection_trajectories || []).filter(r=>r.bank_id===t?.bank_id&&r.map_seed===t?.map_seed&&r.repetition===t?.repetition),routes=(banksData()?.route_ceilings?.by_start || []).filter(r=>r.bank_id===t?.bank_id&&r.map_seed===t?.map_seed);
  $('guided-collection-map-comparison').textContent=t?`Same predetermined map and slot: ${rows.map(r=>`${r.condition==='constrained_bootstrap'?'Random arm':'Mixed arm'} ${r.success?'success':'failure'} in ${number(r.steps.length)} actual steps`).join('; ')}. ${t.repetition===0?'Slot 0 reuses the same random stream; matching recorded trajectories verify the shared portion.':'Slot 8 compares the archived random journey with its fixed-policy replacement.'} Recorded shortest successful paths: ${routes.map(r=>`${r.condition==='constrained_bootstrap'?'random':'mixed'} ${r.logged_success_reachable?number(r.logged_shortest_steps)+' steps':'unreachable'}`).join('; ')}. Physical planner ${number(routes[0]?.planner_steps)} steps.`:'';
}
for(const id of ['guided-collection-replay-map','guided-collection-replay-condition','guided-collection-replay-slot'])$(id).addEventListener('change',selectGuidedCollectionTrajectory);
$('guided-collection-replay-play').addEventListener('click',()=>{if(guidedCollectionTimer){stopGuidedCollectionPlayback();return;}if(!guidedCollectionTrajectory)return;stopBanksPlayback();stopFamiliarPlayback();if(guidedCollectionFrame>=guidedCollectionTrajectory.steps.length)guidedCollectionFrame=0;$('guided-collection-replay-play').textContent='Ⅱ Pause';$('guided-collection-replay-play').setAttribute('aria-label','Pause collection recording');guidedCollectionTimer=setInterval(()=>{guidedCollectionFrame=Math.min(guidedCollectionFrame+1,guidedCollectionTrajectory.steps.length);$('guided-collection-replay-scrub').value=String(guidedCollectionFrame);drawGuidedCollectionWorld();if(guidedCollectionFrame>=guidedCollectionTrajectory.steps.length)stopGuidedCollectionPlayback();},160);});
$('guided-collection-replay-reset').addEventListener('click',()=>{stopBanksPlayback();guidedCollectionFrame=0;$('guided-collection-replay-scrub').value='0';drawGuidedCollectionWorld();});$('guided-collection-replay-scrub').addEventListener('input',event=>{stopBanksPlayback();guidedCollectionFrame=Number(event.target.value);drawGuidedCollectionWorld();});
function renderGuidedCollection(){
  if(!banksIsGuided())return;const study=banksData(),bank=Number($('banks-selected-bank').value),record=banksRecords().find(r=>r.bank_id===bank),collections=(study?.collection?.per_condition || []).filter(r=>r.bank_id===bank),actions=record?.recorded_actions?.per_condition || [],collector=study?.collector;
  const collect=condition=>collections.find(r=>r.condition===condition),metrics=[['Complete collection episodes',r=>number(r?.complete_collection_episodes)],['Actual collection interactions',r=>number(r?.collection_steps)],['Collection success',r=>percent(r?.collection_episodes?r.collection_successes/r.collection_episodes:undefined)]];
  $('guided-collection-costs').innerHTML=metrics.map(([label,format])=>`<div class="stat-card"><span>${label}</span><div class="fixed-outcome-pair">${banksConditions().map(condition=>`<div><small>${condition==='constrained_bootstrap'?'Random · archived':'Mixed · new'}</small><strong style="color:${coverageColors[condition]}">${format(collect(condition))}</strong></div>`).join('')}</div><small>Original training maps · complete episodes</small></div>`).join('');
  $('guided-collector-provenance').textContent=`Additional historical input: one frozen bank-1 / seed-0 / 30,000-update constrained-DDQN collector, fixed by index. Its prior training used ${number(collector?.historical_interactions)} collected interactions, ${number(collector?.historical_updates)} optimizer updates, ${number(collector?.historical_state_presentations)} sampled states and ${number(collector?.historical_action_targets)} action targets. This history includes the random slots replaced here. Collector identity ${collector?.unchanged===true?'verified unchanged':collector?.unchanged===false?'changed — verification failed':'not yet verified'}. The same guided routes are shared across banks; the study matches complete episodes, not total historical cost or independent demonstrations.`;
  familiarTable('guided-collection-policy-table',collections.flatMap(r=>(r.per_policy || []).map(p=>({...p,condition:r.condition}))),[['condition','Collection arm',guidedCollectionLabel],['policy','Component',v=>v==='guided'?'Frozen greedy DDQN':'Uniform random'],['episodes','Episodes'],['complete_episodes','Complete episodes'],['steps','Actual interactions'],['successes','Successful episodes'],['mean_steps','Mean steps',decimal],['noop_steps','Blocked moves'],['unique_trajectories','Distinct recorded trajectories']]);
  const checks=study?.provenance?.random_slot_replication || [],check=checks.find(r=>r.bank_id===bank),mixed=collect(banksConditions()[1]),duplicates=mixed?.guided_duplicate_checks || [];
  $('guided-collection-integrity').textContent=`Bank ${number(bank)}: guided-route repetition records cover ${number(duplicates.length)} maps. ${number(duplicates.filter(r=>r.unique_trajectories===1).length)} maps have one distinct guided route across their repeated guided episodes. Shared random slots 0–7: ${check?.identical===true?'verified identical to the archive':check?.identical===false?'verification failed':'see saved collection provenance'} (${number(check?.compared_steps)} compared action steps). Repeated visits and actions are deduplicated before learner replay; they do not receive extra loss weight.`;
  const routes=(study?.route_ceilings?.per_condition || []).filter(r=>r.bank_id===bank);
  $('guided-route-ceilings').innerHTML=routes.flatMap(r=>[['Success reachable','success_ceiling','success_reachable_starts'],['Efficient success reachable','efficient_success_ceiling','efficient_success_reachable_starts']].map(([label,key,count])=>`<div class="diagnostic-row"><span>${r.condition==='constrained_bootstrap'?'Random':'Mixed'} · ${label}<small>${number(r[count])} / ${number(r.starts)} original starts</small></span><div class="diagnostic-track"><div class="diagnostic-fill" style="width:${Number.isFinite(r[key])?100*r[key]:0}%;background:${coverageColors[r.condition]}"></div></div><strong>${percent(r[key])}</strong></div>`)).join('');
  $('guided-route-ceilings-note').textContent='These ceilings use successful-path reachability and shortest successful paths through recorded outcomes only, from original training starts. Efficient means within twice the full-world shortest route. They measure the available recorded journeys; they do not predict the learned policy’s score. Fresh-map behavior above has a separate evaluation population.';
  $('guided-action-bars').innerHTML=actions.flatMap(r=>(r.states_by_observed_actions || []).map(a=>`<div class="diagnostic-row"><span>${r.condition==='constrained_bootstrap'?'Random':'Mixed'} · ${number(a.actions)} recorded actions<small>${number(a.states)} / ${number(r.supported_states)} supported states</small></span><div class="diagnostic-track"><div class="diagnostic-fill" style="width:${r.supported_states?100*a.states/r.supported_states:0}%;background:${coverageColors[r.condition]}"></div></div><strong>${percent(r.supported_states?a.states/r.supported_states:undefined)}</strong></div>`)).join('');
  $('guided-action-caption').textContent=actions.map(r=>`${r.condition==='constrained_bootstrap'?'Random':'Mixed'}: ${number(r.supported_states)} unique current states, ${number(r.recorded_edges)} distinct recorded state-action edges (${precisePercent(r.observed_action_fraction)} of ${number(r.all_action_edges)} possible supported-state actions), ${number(r.duplicate_records)} repeated edge records.`).join(' ')+' Four network outputs do not create four observed action outcomes. State support and action coverage may change along with route quality.';
  familiarTable('guided-targets-table',(study?.action_exposure?.per_seed || []).filter(r=>r.bank_id===bank),[['condition','Learner source',banksLabel],['seed','Seed'],['updates','Optimizer updates'],['state_presentations','State presentations'],['action_target_presentations','Action targets'],['terminal_action_targets','Terminal targets'],['nonterminal_target_queries','Neural successor queries'],['outside_support_target_queries','Queries outside own support']]);
  selectGuidedCollectionTrajectory();
}


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
  stopCoveragePlayback();stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();
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
  stopCoveragePlayback();stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();coverage=coverageStudies[$('coverage-study').value] || null;
  coverageConditions=coverageIsEqual()?['collected_unique','uniform_subset']:['exhaustive','collected_unique'];
  for(const id of ['coverage-replay-condition','coverage-replay-controller','coverage-replay-checkpoint','coverage-replay-panel','coverage-replay-seed','coverage-support-condition','robustness-replay-panel','robustness-replay-controller','robustness-replay-seed','banks-selected-bank','banks-exposure-seed','banks-exposure-map','banks-replay-panel','banks-replay-controller','banks-replay-seed','familiar-bank','familiar-replay-map','familiar-replay-controller','familiar-replay-action-set','familiar-replay-seed','guided-collection-replay-map','guided-collection-replay-condition','guided-collection-replay-slot'])$(id).value='';
  renderCoverage();
}
$('coverage-study').addEventListener('change',selectCoverageStudy);
function renderCoverage(){
  stopCoveragePlayback();stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();
  const familiar=$('coverage-study').value==='familiar_starts',frozen=$('coverage-study').value==='panel_evaluation',replicated=['bank_replication','map_replay','within_map','recorded_actions','constrained_bootstrap','logged_graph','guided_collection'].includes($('coverage-study').value);
  $('banks-view').hidden=!replicated;$('robustness-view').hidden=!frozen;$('coverage-learning-intro').hidden=frozen || replicated || familiar;$('familiar-view').hidden=!familiar;
  if(familiar){$('coverage-study-context').textContent='Privileged familiar-start diagnostic · frozen networks · no new learning or fresh maps.';$('coverage-empty').hidden=true;$('coverage-content').hidden=true;renderFamiliar();return;}
  if(replicated){$('coverage-study-context').textContent=banksIsGuided()?'Equal complete episodes; unequal interactions and historical costs. Frozen collection followed by offline learning.':banksIsGraph()?'Exact recorded-graph labels versus archived constrained bootstrap; no full-world Q* training or new collection.':banksIsConstrained()?'Same recorded supervision and replay; constrain training targets only. All four actions remain available at evaluation.':banksIsRecorded()?'Same collected states and replay; treatment loss uses recorded actions only. Fewer action targets, no new collection.':banksIsWithinMap()?'Same per-map state counts and replay exposure; selected states change. New fits versus archived controls.':banksIsMapReplay()?'Same collected supports; new balanced replay fits versus archived baseline. No new collection.':'New bank pairs and final-policy evaluation; inspect each bank before the equal-bank mean.';$('coverage-empty').hidden=true;$('coverage-content').hidden=true;renderBanks();return;}
  if(frozen){$('coverage-study-context').textContent='Frozen policies on several new panels; no training or new competence gate.';$('coverage-empty').hidden=true;$('coverage-content').hidden=true;renderRobustness();return;}
  renderCoveragePanelSensitivity();
  const equal=coverageIsEqual();
  $('coverage-study-context').textContent=equal?'Same number of states; different bank composition. Fresh panel belongs only to this study.':'Earlier measured coverage comparison, with its own fresh panel and larger exhaustive bank.';
  $('coverage-intro-copy').textContent=equal?'Both conditions use the same Double DQN learner and the same number of training states. One reuses the archived bank of unique states reached by random collection; the other uses an equally sized uniform subset of the exhaustive state universe. This tests which states are included while holding bank size fixed. Both remain offline and receive all four action transitions.':'Both conditions use the same Double DQN target procedure, network, all four actions, and update budget. One samples every training state. The other samples unique states encountered during random collection on the same layouts. The support and sampling distribution change together; this is offline learning, with the movement rule visible initially.';
  $('coverage-condition-preview').innerHTML=coverageConditions.map(c=>`<span>${coverageLabel(c)}</span>`).join('');
  $('coverage-flow').innerHTML=(equal?[
    ['Same state count','Archived collected bank / uniform subset'],['Different included states','Same exhaustive training universe'],['Same Double DQN','All actions, network, loss, updates'],['Separate fresh panel','Own rollouts; no oracle actions'],
  ]:[['Same training layouts','Every possible state / random trajectories'],['Two fixed state banks','Exhaustive / unique collected states'],['Same Double DQN','All actions, network, loss, update budget'],['New fresh layouts','Own rollouts; no oracle actions']]).map(([title,detail])=>`<li><strong>${title}</strong><small>${detail}</small></li>`).join('');
  $('coverage-flow-note').textContent=equal?'No new collection is performed. The collected bank is reused unchanged, and the uniform bank is sampled before training. Both banks are shared across learner seeds. Detached successor queries may reach states outside a bank without adding them to its direct training support.':'Random collection happens before learning and never adapts to either network. Collected states are sampled uniformly after duplicates are removed. Both conditions still receive all four action transitions at each training state, beyond ordinary trajectory-only experience.';
  $('coverage-paired-caption').textContent=`Every difference is ${coverageDirection()}. Success, efficient-success, and no-op-rate differences are percentage points; a negative step difference means fewer steps. Failed episodes still count. These are descriptive paired measurements from this study's own final training and fresh panels.`;
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
function renderCoveragePanelSensitivity(){
  const note=$('coverage-panel-sensitivity');note.hidden=!coverageIsEqual();
  note.textContent='This study uses a separate fresh panel. Scores and gates belong to the selected study; inspect the earlier coverage study separately.';
  if(!coverageIsEqual() || !coverage)return;
  const prior=coverageStudies.coverage,provenance=coverage.provenance || {},seeds=coverage.protocol?.seeds || [],priorSeeds=prior?.protocol?.seeds || [];
  const checkpoint=coverage.protocol?.budget?.updates_per_seed,priorCheckpoint=prior?.protocol?.budget?.updates_per_seed;
  const maps=coverage.protocol?.dataset?.heldout_map_seeds || [],priorMaps=prior?.protocol?.dataset?.heldout_map_seeds || [];
  const checks=provenance.collected_replication || [],flags=['applicable','final_weights_identical','target_weights_identical','initial_weights_identical','batch_index_sha256_identical','global_counts_identical'];
  const archiveMatch=typeof provenance.archive==='string' && provenance.archive===prior?.artifacts?.directory;
  const replicated=seeds.length>0 && seeds.length===priorSeeds.length && seeds.every(seed=>priorSeeds.includes(seed)) && checks.length===seeds.length && seeds.every(seed=>{const rows=checks.filter(r=>r.seed===seed);return rows.length===1 && flags.every(key=>rows[0][key]===true) && rows[0].archive_model===`${provenance.archive}/models/collected_unique_seed${seed}_update${priorCheckpoint}.pt`;});
  const eligible=coverage.run?.status==='complete' && coverage.gates?.eligible===true && prior?.run?.status==='complete' && prior?.gates?.eligible===true;
  if(!eligible || !archiveMatch || !replicated || provenance.replication_required!==true || provenance.training_arrays_identical!==true || provenance.transition_arrays_identical!==true || checkpoint!==priorCheckpoint || !maps.length || !priorMaps.length || !maps.some(seed=>!priorMaps.includes(seed)))return;
  const final=(study,update)=>study.aggregate?.find(r=>r.condition==='collected_unique' && r.panel==='heldout' && r.mode==='greedy' && r.checkpoint===update);
  const current=final(coverage,checkpoint),previous=final(prior,priorCheckpoint);
  if(!Number.isFinite(current?.success_rate) || !Number.isFinite(previous?.success_rate))return;
  note.textContent=`The collected networks have identical saved weights in both studies. Their final greedy success is ${percent(previous.success_rate)} on the earlier panel (first map ${priorMaps[0]}) and ${percent(current.success_rate)} on this panel (first map ${maps[0]}). The evaluation maps changed; this is not a learning gain. Use the study selector to inspect either panel.`;
}

function renderCoverageSupport(){
  const equal=coverageIsEqual(),description=coverage?.coverage || {},records=equal?(description.per_condition || []):[{condition:'collected_unique',...description}];
  $('coverage-bank-comparison').hidden=!equal;$('coverage-support-selector-label').hidden=!equal;
  selectOptions('coverage-support-condition',records.map(r=>[r.condition,coverageLabel(r.condition)]),'collected_unique');
  const support=records.find(r=>r.condition===$('coverage-support-condition').value) || {},maps=support.by_map || [],buckets=support.by_time_bucket || [],queries=support.successor_queries || {};

  const overall=support.overall || {};
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
  $('coverage-map-heatmap').innerHTML=supportMapSVG(maps);
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
  stopCoveragePlayback();stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();
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
$('coverage-replay-play').addEventListener('click',()=>{if(coverageTimer){stopCoveragePlayback();stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();return;}if(!coverageTrajectory)return;if(coverageFrame>=coverageTrajectory.steps.length)coverageFrame=0;$('coverage-replay-play').textContent='Ⅱ Pause';$('coverage-replay-play').setAttribute('aria-label','Pause saved experience-coverage trajectory');coverageTimer=setInterval(()=>{coverageFrame=Math.min(coverageFrame+1,coverageTrajectory.steps.length);$('coverage-replay-scrub').value=String(coverageFrame);drawCoverageWorld();if(coverageFrame>=coverageTrajectory.steps.length)stopCoveragePlayback();stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();},160);});
$('coverage-replay-reset').addEventListener('click',()=>{stopCoveragePlayback();stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();coverageFrame=0;$('coverage-replay-scrub').value='0';drawCoverageWorld();});
$('coverage-replay-scrub').addEventListener('input',event=>{stopCoveragePlayback();stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();coverageFrame=Number(event.target.value);drawCoverageWorld();});

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

function renderUpdateChart(id,series,checkpoints,rates,title,{signedValues=false}={}) {
  const steps=[...new Set(checkpoints || [])].filter(Number.isFinite).sort((a,b)=>a-b);
  const maximum=Math.max(1,...steps),values=series.flatMap(s=>s.rows.flatMap(r=>[r.value,r.high])).filter(Number.isFinite);
  const ceiling=rates?1:Math.max(.001,...values.map(v=>signedValues?Math.abs(v):v))*1.05,floor=signedValues?-ceiling:0,w=500,h=265,p={l:48,r:17,t:18,b:45};
  const x=step=>p.l+step/maximum*(w-p.l-p.r),y=value=>p.t+(ceiling-value)/(ceiling-floor)*(h-p.t-p.b),fmt=rates?percent:signedValues?signedDecimal:decimal;
  let svg=`<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg"><title>${escape(title)} by optimizer update</title>`;
  for(let i=0;i<=4;i++){const value=floor+i*(ceiling-floor)/4;svg+=`<line x1="${p.l}" y1="${y(value)}" x2="${w-p.r}" y2="${y(value)}" stroke="#26302a"/><text x="${p.l-8}" y="${y(value)+4}" text-anchor="end" fill="#8a978f" font-size="10">${fmt(value)}</text>`;}
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
  stopPlayback();stopCompetencePlayback();stopSupervisedPlayback();stopFixedPlayback();stopCoveragePlayback();stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();
  const adaptationPath=$('study-run').value==='pilot_v1'?'../experiments/adaptation/pilot_v1/results.json':'data/adaptation.json';
  const results=await Promise.allSettled([getData(adaptationPath),getData('data/provenance.json'),getData('data/competence.json'),getData('data/supervised.json'),getData('data/fixed_targets.json'),getData('data/coverage.json'),getData('data/equal_support.json'),getData('data/panel_evaluation.json'),getData('data/bank_replication.json'),getData('data/map_replay.json'),getData('data/within_map.json'),getData('data/recorded_actions.json'),getData('data/constrained_bootstrap.json'),getData('data/logged_graph.json'),getData('data/familiar_starts.json'),getData('data/guided_collection.json')]);
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
  if(results[7].status==='fulfilled')coverageStudies.panel_evaluation=results[7].value;else {coverageStudies.panel_evaluation=null;errors.push(results[7].reason.message);}
  if(results[8].status==='fulfilled')coverageStudies.bank_replication=results[8].value;else {coverageStudies.bank_replication=null;errors.push(results[8].reason.message);}
  if(results[9].status==='fulfilled')coverageStudies.map_replay=results[9].value;else {coverageStudies.map_replay=null;errors.push(results[9].reason.message);}
  if(results[10].status==='fulfilled')coverageStudies.within_map=results[10].value;else {coverageStudies.within_map=null;errors.push(results[10].reason.message);}
  if(results[11].status==='fulfilled')coverageStudies.recorded_actions=results[11].value;else {coverageStudies.recorded_actions=null;errors.push(results[11].reason.message);}
  if(results[12].status==='fulfilled')coverageStudies.constrained_bootstrap=results[12].value;else {coverageStudies.constrained_bootstrap=null;errors.push(results[12].reason.message);}
  if(results[13].status==='fulfilled')coverageStudies.logged_graph=results[13].value;else {coverageStudies.logged_graph=null;errors.push(results[13].reason.message);}
  if(results[14].status==='fulfilled')coverageStudies.familiar_starts=results[14].value;else {coverageStudies.familiar_starts=null;errors.push(results[14].reason.message);}
  if(results[15].status==='fulfilled')coverageStudies.guided_collection=results[15].value;else {coverageStudies.guided_collection=null;errors.push(results[15].reason.message);}
  coverage=coverageStudies[$('coverage-study').value] || null;coverageConditions=coverageIsEqual()?['collected_unique','uniform_subset']:['exhaustive','collected_unique'];
  renderCoverage();
  if(errors.length) $('load-status').textContent=errors.join(' · ');
  else {
    const loaded=[coverageStudies.guided_collection&&'Guided collection',coverageStudies.familiar_starts&&'Familiar starts',coverageStudies.logged_graph&&'Logged graph',coverageStudies.constrained_bootstrap&&'Constrained bootstrap',coverageStudies.recorded_actions&&'Recorded actions',coverageStudies.within_map&&'Within-map states',coverageStudies.map_replay&&'Map-balanced replay',coverageStudies.bank_replication&&'Bank replications',coverageStudies.panel_evaluation&&'Panel robustness',coverageStudies.equal_support&&'Equal-size banks',coverageStudies.coverage&&'Experience coverage',fixed&&'Fixed-data targets',supervised&&'Exact targets',competence&&'Competence',adaptation&&'Adaptation',provenance&&'Provenance'].filter(Boolean);
    $('load-status').textContent=`${loaded.join(' + ') || 'No'} saved ${loaded.length===1?'study':'studies'} loaded · ${new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}`;
  }
  $('refresh').disabled=false;$('study-run').disabled=false;loadInProgress=false;
}
$('refresh').addEventListener('click',loadData);
$('study-run').addEventListener('change',loadData);
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopPlayback();stopCompetencePlayback();stopSupervisedPlayback();stopFixedPlayback();stopCoveragePlayback();stopRobustnessPlayback();stopBanksPlayback();stopFamiliarPlayback();}});
await loadData();
