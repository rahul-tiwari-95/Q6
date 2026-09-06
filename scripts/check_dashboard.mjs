// Exercise saved-study controls with DOM/canvas stubs; does not test browser layout.
// Run from any directory: node /path/to/Q6/scripts/check_dashboard.mjs [competence-results.json] [--supervised supervised-results.json] [--missing-supervised] [--fixed fixed-targets-results.json] [--missing-fixed] [--coverage coverage-results.json] [--missing-coverage] [--equal-support results.json] [--missing-equal-support]
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
let competencePath=null,supervisedPath=null,missingSupervised=false,fixedPath=null,missingFixed=false,coveragePath=null,missingCoverage=false,equalSupportPath=null,missingEqualSupport=false;
for(let i=2;i<process.argv.length;i++){
 const arg=process.argv[i];
 if(arg==='--supervised'){supervisedPath=process.argv[++i];assert(supervisedPath,'--supervised needs a real results path');}
 else if(arg==='--missing-supervised')missingSupervised=true;
 else if(arg==='--fixed'){fixedPath=process.argv[++i];assert(fixedPath,'--fixed needs a real results path');}
 else if(arg==='--missing-fixed')missingFixed=true;
 else if(arg==='--coverage'){coveragePath=process.argv[++i];assert(coveragePath,'--coverage needs a real results path');}
 else if(arg==='--missing-coverage')missingCoverage=true;
 else if(arg==='--equal-support'){equalSupportPath=process.argv[++i];assert(equalSupportPath,'--equal-support needs a real results path');}
 else if(arg==='--missing-equal-support')missingEqualSupport=true;
 else if(!arg.startsWith('-')&&!competencePath)competencePath=arg;
 else throw Error('Unknown argument '+arg);
}
const html=fs.readFileSync(path.join(root,'dashboard/lab.html'),'utf8');
const source=fs.readFileSync(path.join(root,'dashboard/lab.js'),'utf8').replace(/^import .*;\n/,'const applyMode=()=>{}, bindModeToggle=()=>{};\n');
const decode=s=>s.replaceAll('&quot;','"').replaceAll('&#39;',"'").replaceAll('&amp;','&').replaceAll('&lt;','<').replaceAll('&gt;','>');
class Element {
  constructor(id='',tag='div',attrs='') {this.id=id;this.tag=tag;this.listeners={};this.attributes={};this.dataset={};this._value='';this.options=[];this.hidden=/\bhidden\b/.test(attrs);this.disabled=false;this.width=560;this.classList={toggle(){}};for(const [,k,v] of attrs.matchAll(/([\w-]+)="([^"]*)"/g)){this.attributes[k]=v;if(k.startsWith('data-'))this.dataset[k.slice(5)]=v;}this._html='';}
  addEventListener(name,cb){this.listeners[name]=cb;}
  setAttribute(name,value){this.attributes[name]=value;}
  get innerHTML(){return this._html;}
  set innerHTML(value){this._html=value;this.options=[...value.matchAll(/<option value="([^"]*)">([^<]*)<\/option>/g)].map(([,value,text])=>({value:decode(value),text:decode(text)}));if(this.options.length)this._value=this.options[0].value;this.buttons=[...value.matchAll(/<button\b([^>]*)>/g)].map(([,attrs])=>new Element('','button',attrs));}
  get value(){return this._value;}
  set value(value){this._value=String(value);}
  querySelectorAll(selector){if(selector==='button')return this.buttons||[];throw Error('Unhandled selector '+selector);}
  querySelector(selector){const id=/data-support="([^"]*)"/.exec(selector)?.[1];return this.buttons?.find(b=>b.dataset.support===id)||null;}
  focus(){}
  getContext(){return new Proxy({}, {get:(obj,key)=>obj[key]||(()=>{}),set:(obj,key,value)=>{obj[key]=value;return true;}});}
}
const elements=new Map();
for(const [,tag,attrs,id] of html.matchAll(/<(\w+)\b([^>]*?\bid="([^"]+)"[^>]*)>/g)){
 const e=new Element(id,tag,attrs);elements.set(id,e);
 if(tag==='select'){const markup=html.slice(html.indexOf(`<${tag}${attrs}>`)+(`<${tag}${attrs}>`).length).split('</select>')[0];e.innerHTML=markup;}
}
const allButtons=[...html.matchAll(/<button\b([^>]*)>/g)].map(([,attrs])=>new Element('','button',attrs));
const document={getElementById(id){assert(elements.has(id),'Unknown DOM id '+id);return elements.get(id);},querySelectorAll(selector){if(selector==='.phase-card')return allButtons.filter(b=>b.attributes.class?.includes('phase-card'));if(selector==='[data-reference]')return allButtons.filter(b=>b.dataset.reference);throw Error('Unhandled document selector '+selector);},addEventListener(){},hidden:false};
const files=[];
const intervals=new Map();let intervalId=0;
const sandbox={document,location:{hash:''},history:{replaceState(){}},setInterval:callback=>{const id=++intervalId;intervals.set(id,callback);return id;},clearInterval(id){intervals.delete(id);},console,fetch:async relative=>{
 const full=path.resolve(root,'dashboard',relative);files.push(relative);
 if(relative==='data/supervised.json'&&missingSupervised)return {status:404,ok:false};
 if(relative==='data/fixed_targets.json'&&missingFixed)return {status:404,ok:false};
 if(relative==='data/coverage.json'&&missingCoverage)return {status:404,ok:false};
 if(relative==='data/equal_support.json'&&missingEqualSupport)return {status:404,ok:false};
 const override=relative==='data/competence.json'?competencePath:relative==='data/supervised.json'?supervisedPath:relative==='data/fixed_targets.json'?fixedPath:relative==='data/coverage.json'?coveragePath:relative==='data/equal_support.json'?equalSupportPath:null;
 const target=override||full;
 if(!fs.existsSync(target))return {status:404,ok:false};
 return {status:200,ok:true,json:async()=>JSON.parse(fs.readFileSync(target,'utf8'))};
}};
const context=vm.createContext(sandbox);
await vm.runInContext(`(async()=>{${source}\n globalThis.labDebug={get coverage(){return coverage;},renderCoverage,get coverageSelected(){return coverageTrajectory;},set coverageFrame(value){coverageFrame=value;},drawCoverageWorld,get fixed(){return fixed;},renderFixed,get fixedSelected(){return fixedTrajectory;},set fixedFrame(value){fixedFrame=value;},drawFixedWorld,activateTab,renderCompetence,renderCompetenceComparison,selectCompetenceTrajectory,drawCompetenceWorld,loadData,renderSupervised,get supervised(){return supervised;},get supervisedSelected(){return supervisedTrajectory;},drawSupervisedWorld,set supervisedFrame(value){supervisedFrame=value;},get competence(){return competence;},get selected(){return competenceTrajectory;},get condition(){return competenceCondition;},set frame(value){competenceFrame=value;}};})()`,context);
const get=id=>elements.get(id),debug=sandbox.labDebug;
assert.equal(get('adaptation-content').hidden,false,'Existing adaptation data render');
assert.equal(get('provenance-content').hidden,false,'Existing provenance data render');
assert(get('success-chart').innerHTML.includes('<svg'),'Existing adaptation curve render');
assert(get('provenance-bars').innerHTML.includes('bar-row'),'Existing provenance bars render');
const expectedMissing=[];if(missingSupervised)expectedMissing.push('data/supervised.json');if(missingFixed || (!fixedPath&&!fs.existsSync(path.join(root,'dashboard/data/fixed_targets.json'))))expectedMissing.push('data/fixed_targets.json');
if(missingCoverage || (!coveragePath&&!fs.existsSync(path.join(root,'dashboard/data/coverage.json'))))expectedMissing.push('data/coverage.json');
if(missingEqualSupport || (!equalSupportPath&&!fs.existsSync(path.join(root,'dashboard/data/equal_support.json'))))expectedMissing.push('data/equal_support.json');
const unexpectedErrors=(get('load-status').textContent || '').split(' · ').filter(text=>text.includes('Could not load')&&!expectedMissing.some(path=>text.includes(path)));assert.deepEqual(unexpectedErrors,[],'No unexpected data errors');
assert.equal(get('tab-coverage').attributes['aria-selected'],'true','Experience coverage is the default track');
assert.equal(get('coverage-study').value,'equal_support','Equal-size banks is the default coverage study');
for(const studyKey of ['equal_support','coverage']){
 get('coverage-study').value=studyKey;get('coverage-study').listeners.change();
 const equal=studyKey==='equal_support',conditions=equal?['collected_unique','uniform_subset']:['exhaustive','collected_unique'];
 const label=condition=>({exhaustive:'Exhaustive states',collected_unique:'Collected unique states',uniform_subset:'Uniform subset'}[condition]);
 get('coverage-epsilon').value='greedy';get('coverage-epsilon').listeners.change();
if(debug.coverage?.aggregate?.length){
 assert.equal(get('coverage-content').hidden,false);
 const chartIds=['coverage-train-chart','coverage-fresh-chart','coverage-train-agreement','coverage-fresh-agreement','coverage-loss-chart'];
 for(const id of chartIds)assert(get(id).innerHTML.includes('<svg'));
 for(const condition of conditions)if(debug.coverage.aggregate.some(r=>r.condition===condition))assert(get('coverage-fresh-chart').innerHTML.includes(label(condition)));
 assert(get('coverage-gate-results').innerHTML.includes('Efficient:'),'Separate efficiency gates visible');
 assert(get('coverage-state-table').innerHTML.includes('Winnable states'),'Full-state denominator visible');
 assert(get('coverage-paired-table').innerHTML.includes('Success Δ'),'Paired comparison visible');
 assert(get('coverage-evidence-links').innerHTML.includes(`href="data/${studyKey}.json"`),'Coverage download points to displayed study');
 const checkCoverageOutcomeCards=mode=>{
  const finalUpdate=debug.coverage.protocol.budget.updates_per_seed;
  for(const [id,key,format] of [['success','success_rate',value=>(100*value).toFixed(1)+'%'],['efficient','efficient_success_rate',value=>(100*value).toFixed(1)+'%'],['steps','mean_steps',value=>value.toFixed(1)]]){
   const values=[...get('coverage-outcome-'+id).innerHTML.matchAll(/<strong[^>]*>([^<]*)<\/strong>/g)].map(match=>match[1]);
   const expected=conditions.map(condition=>{const value=debug.coverage.aggregate.find(r=>r.condition===condition && r.panel==='heldout' && r.mode===mode && r.checkpoint===finalUpdate)?.[key];return Number.isFinite(value)?format(value):'—';});
   assert.deepEqual(values,expected,'Final fresh '+key+' cards match saved '+mode+' aggregates');
  }
 };
 checkCoverageOutcomeCards('greedy');
 assert(get('coverage-outcome-success').innerHTML.includes('Primary'));assert(get('coverage-outcome-efficient').innerHTML.includes('Secondary'));
 assert(debug.coverageSelected,'Default experience-coverage recording selected');
 const supportRecords=equal?debug.coverage.coverage.per_condition:[debug.coverage.coverage];
 for(const support of supportRecords){
  if(equal){get('coverage-support-condition').value=support.condition;get('coverage-support-condition').listeners.change();}

 assert.equal((get('coverage-map-heatmap').innerHTML.match(/<rect /g)||[]).length,support.by_map.length,'Every declared layout has one coverage square');
 for(const row of support.by_map)assert(get('coverage-map-heatmap').innerHTML.includes(`Map ${row.map_seed.toLocaleString()}: ${row.visited_states.toLocaleString()} / ${row.states.toLocaleString()}`),'Layout tooltip reports saved numerator and denominator');
 assert.equal((get('coverage-time-bars').innerHTML.match(/class="diagnostic-row"/g)||[]).length,support.by_time_bucket.length,'Every time bucket rendered');
 for(const row of support.by_time_bucket)assert(get('coverage-time-bars').innerHTML.includes((100*row.coverage_rate).toFixed(1)+'%'),'Saved time coverage ratio rendered');
 assert(get('coverage-support-caption').textContent.includes(support.unique_current_states.toLocaleString()));
 const diagnosticCards=[...get('coverage-support-stats').innerHTML.matchAll(/<div><span>([^<]*)<\/span><strong>([^<]*)<\/strong><small>([^<]*)<\/small><\/div>/g)];
 assert.equal(diagnosticCards.length,6,'Coverage summary preserves two rows of three diagnostic cards');
 for(const [offset,rate,numerator,denominator] of [
  [3,support.winnable_current_state_fraction,support.overall.visited_winnable_states,support.overall.winnable_states],
  [4,support.goal_near_current_state_fraction,support.overall.visited_goal_near_states,support.overall.goal_near_states],
  [5,support.successor_queries.outside_support_fraction,support.successor_queries.outside_support_nonterminal_transitions,support.successor_queries.nonterminal_transitions],
 ]){assert.equal(diagnosticCards[offset][2],(100*rate).toFixed(2)+'%');assert(diagnosticCards[offset][3].includes(`${numerator.toLocaleString()} / ${denominator.toLocaleString()}`),'Coverage subset numerator and denominator match saved data');}
 assert(diagnosticCards[4][3].includes('physical shortest path ≤ 2 moves; all remaining clocks'),'Goal-near coverage definition stays independent of remaining time');
 assert(get('coverage-successor-caption').textContent.includes(support.successor_queries.outside_support_nonterminal_transitions.toLocaleString()));
 }
 const layoutCells=[...get('coverage-layout-table').innerHTML.matchAll(/<tr>(<td>.*?)<\/tr>/g)].map(match=>[...match[1].matchAll(/<td>(.*?)<\/td>/g)].map(cell=>cell[1]));
 assert.equal(layoutCells.length,debug.coverage.paired_differences.per_layout.length);
 const signedDelta=(value,scale=1,suffix='')=>`${value>0?'+':''}${(value*scale).toFixed(1)}${suffix}`;
 debug.coverage.paired_differences.per_layout.forEach((row,index)=>{assert.equal(layoutCells[index][4],signedDelta(row.efficient_success_delta),'Per-layout efficient-success delta matches saved data');assert.equal(layoutCells[index][7],signedDelta(row.noop_rate_delta,100,' pp'),'Per-layout no-op-rate delta matches saved data');});


 assert(get('coverage-successor-caption').textContent.includes('not optimizer samples'));
 assert(html.includes('does not add it to the training support'),'Current support and detached successor queries explicitly distinguished');
 if(equal){
  const membership=debug.coverage.coverage;
  assert(get('coverage-intersection-caption').textContent.includes(membership.support_size.toLocaleString()),'Equal bank size is visible');
  assert(get('coverage-intersection-caption').textContent.includes(membership.intersection.states.toLocaleString()),'Support intersection is visible');
  assert(get('coverage-intersection-caption').textContent.includes('No new collection'));
  assert(get('coverage-support-caption').textContent.includes('not new visits'));
  assert(get('coverage-paired-caption').textContent.startsWith('Every difference is uniform subset minus collected unique states'),'Equal-size delta direction stays explicit');
  assert(get('coverage-layout-caption').textContent.includes('uniform subset minus collected unique states'));
  const ownRows=[...(debug.coverage.support_diagnostics?.per_seed || []),...(debug.coverage.support_diagnostics?.pooled || [])];
  assert.equal(get('coverage-support-fit-panel').hidden,!ownRows.length);
  const rendered=[...get('coverage-support-fit-table').innerHTML.matchAll(/<tr>(<td>.*?)<\/tr>/g)].map(match=>[...match[1].matchAll(/<td>(.*?)<\/td>/g)].map(cell=>cell[1]));
  assert.equal(rendered.length,ownRows.length);
  ownRows.forEach((row,index)=>{assert.equal(rendered[index][2],row.subset==='support'?'Own bank':'Outside own bank');assert.equal(rendered[index][3],row.states.toLocaleString());assert.equal(rendered[index][5],row.winnable_states.toLocaleString());assert.equal(rendered[index][7],(100*row.optimal_action_rate).toFixed(1)+'%');});
 }else{assert.equal(get('coverage-bank-comparison').hidden,true);assert.equal(get('coverage-support-fit-panel').hidden,true);}
 const originalRun={...debug.coverage.run},originalGates=debug.coverage.gates;
 for(const [run,gates,phrase] of [
  [{status:'complete'}, {eligible:true,per_condition:[{condition:conditions[0],fresh:true},{condition:conditions[1],fresh:true}]},equal?'Both equal-size banks pass':'Both coverage conditions pass'],
  [{status:'complete'}, {eligible:true,per_condition:[{condition:conditions[0],fresh:true},{condition:conditions[1],fresh:false}]},equal?'Collected unique states pass':'Exhaustive states pass'],
  [{status:'complete'}, {eligible:true,per_condition:[{condition:conditions[0],fresh:false},{condition:conditions[1],fresh:true}]},equal?'The uniform subset passes':'Collected unique states pass'],
  [{status:'complete'}, {eligible:true,per_condition:[{condition:conditions[0],fresh:false},{condition:conditions[1],fresh:false}]},equal?'Neither equal-size bank':'Neither coverage condition'],
  [{status:'complete',interpretation:'smoke_or_protocol_deviation_not_gate_evidence'}, {eligible:false},'Smoke or protocol-deviation'],
  [{status:'incomplete',interpretation:'incomplete_not_gate_evidence'}, {eligible:false},'Incomplete comparison'],
  [{status:'inconsistent_not_gate_evidence'}, {eligible:false},'Consistency check failed'],
 ]){
  debug.coverage.run=run;debug.coverage.gates=gates;debug.renderCoverage();assert(get('coverage-gate-note').textContent.includes(phrase),'Coverage-data interpretation '+phrase);
 }
 debug.coverage.run=originalRun;debug.coverage.gates=originalGates;debug.renderCoverage();
 const originalAggregate=debug.coverage.aggregate,originalStates=debug.coverage.state_aggregate,originalLosses=debug.coverage.loss_aggregate;
 debug.coverage.run={status:'incomplete_admission_cap',interpretation:'incomplete_not_gate_evidence'};debug.coverage.gates={eligible:false};
 debug.coverage.aggregate=originalAggregate.filter(r=>r.condition===conditions[0]);debug.coverage.state_aggregate=originalStates.filter(r=>r.condition===conditions[0]);debug.coverage.loss_aggregate=originalLosses.filter(r=>r.condition===conditions[0]);debug.renderCoverage();
 assert(get('coverage-gate-note').textContent.includes('Incomplete comparison'));
 for(const id of chartIds)assert(!get(id).innerHTML.includes(label(conditions[1])),'Incomplete comparison does not invent a missing curve');
 checkCoverageOutcomeCards('greedy');
 assert(get('coverage-outcome-success').innerHTML.includes('>—</strong>'),'Absent condition has no substituted final outcome');
 debug.coverage.aggregate=[];debug.renderCoverage();assert.equal(get('coverage-content').hidden,true);for(const id of [...chartIds,'coverage-outcome-success','coverage-outcome-efficient','coverage-outcome-steps'])assert.equal(get(id).innerHTML,'');
 const originalProvenance=debug.coverage.provenance;
 debug.coverage.provenance={stop_reason:'Archive mismatch <example> & paired check failed'};debug.coverage.run={status:'inconsistent_not_gate_evidence'};
 debug.renderCoverage();assert.equal(get('coverage-content').hidden,true);assert.equal(get('coverage-empty-title').textContent,'Consistency check failed.');assert(get('coverage-empty-message').textContent.includes('Archive mismatch <example> & paired check failed'),'Empty consistency state preserves literal reason through textContent');
 debug.coverage.aggregate=originalAggregate;debug.renderCoverage();assert(get('coverage-gate-note').textContent.includes('Consistency check failed'));assert(get('coverage-gate-note').textContent.includes('Archive mismatch <example> & paired check failed'),'Consistency banner preserves literal reason through textContent');
 debug.coverage.provenance=originalProvenance;
 debug.coverage.run=originalRun;debug.coverage.gates=originalGates;debug.coverage.aggregate=originalAggregate;debug.coverage.state_aggregate=originalStates;debug.coverage.loss_aggregate=originalLosses;debug.renderCoverage();
 const stableIds=['coverage-bank-table','coverage-support-fit-table','coverage-map-heatmap','coverage-time-bars','coverage-support-table','coverage-support-stats','coverage-train-agreement','coverage-fresh-agreement','coverage-loss-chart','coverage-paired-table','coverage-gate-results','coverage-fit-metrics'];
 const before=new Map(stableIds.map(id=>[id,get(id).innerHTML]));
 get('coverage-epsilon').value='epsilon_0_1';get('coverage-epsilon').listeners.change();
 checkCoverageOutcomeCards('epsilon_0_1');assert(get('coverage-outcome-caption').textContent.includes('ε = 0.1'));
 for(const id of stableIds)assert.equal(get(id).innerHTML,before.get(id),'Exploration preserves '+id);
 assert(get('coverage-fresh-chart').innerHTML.includes('Optimizer updates per seed'),'Coverage-data axis counts optimizer updates');
 let recordings=0;
 for(const row of debug.coverage.trajectories){
  get('coverage-epsilon').value=row.policy==='learner'?row.mode:'greedy';get('coverage-epsilon').listeners.change();
  get('coverage-replay-condition').value=row.condition;get('coverage-replay-condition').listeners.change();
  get('coverage-replay-controller').value=row.policy;get('coverage-replay-controller').listeners.change();
  get('coverage-replay-checkpoint').value=String(row.checkpoint);get('coverage-replay-checkpoint').listeners.change();
  get('coverage-replay-panel').value=row.panel;get('coverage-replay-panel').listeners.change();
  get('coverage-replay-seed').value=String(row.seed);get('coverage-replay-seed').listeners.change();
  assert.equal(debug.coverageSelected,row,'Every experience-coverage trajectory reachable');
  if(row.steps.length){
   assert(get('coverage-step-diagnostics').innerHTML.includes('Decision for step 1'));
   if(row.policy==='learner'){assert(get('coverage-step-diagnostics').innerHTML.includes('Learned Q'));assert(get('coverage-step-diagnostics').innerHTML.includes('Optimal Q'));}
   debug.coverageFrame=row.steps.length;debug.drawCoverageWorld();
   assert(get('coverage-step-diagnostics').innerHTML.includes('Decision for step '+row.steps.length));
   assert(get('coverage-step-diagnostics').innerHTML.includes('the position after that step'));
  }
  assert(get('coverage-world-caption').textContent.startsWith('Rule A'),'Rule visible from initial frame onward');
  recordings++;
 }
 // Verify each paired network sees its own mode-aligned curve and shared reference panel.
 for(const condition of conditions)for(const mode of ['greedy','epsilon_0_1'])for(const panel of ['train','heldout']){
  get('coverage-epsilon').value=mode;get('coverage-epsilon').listeners.change();
  get('coverage-replay-condition').value=condition;get('coverage-replay-condition').listeners.change();
  get('coverage-replay-controller').value='learner';get('coverage-replay-controller').listeners.change();
  get('coverage-replay-panel').value=panel;get('coverage-replay-panel').listeners.change();
  assert.equal(debug.coverageSelected.condition,condition);assert.equal(debug.coverageSelected.mode,mode);assert.equal(debug.coverageSelected.panel,panel);
  for(const condition of conditions)assert(get('coverage-reference-bars').innerHTML.includes(label(condition)));
  assert(!get('coverage-reference-bars').innerHTML.includes('Historical'),'No historical panel mixed into coverage comparison');
 }
 get('coverage-replay-reset').listeners.click();assert.equal(get('coverage-replay-step').textContent,`0 / ${debug.coverageSelected.steps.length}`);
 get('coverage-replay-play').listeners.click();assert.equal(intervals.size,1);assert.equal(get('coverage-replay-play').textContent,'Ⅱ Pause');
 intervals.values().next().value();assert.equal(get('coverage-replay-step').textContent,`1 / ${debug.coverageSelected.steps.length}`);
 if(intervals.size)get('coverage-replay-play').listeners.click();assert.equal(intervals.size,0);
 get('coverage-replay-scrub').listeners.input({target:{value:String(debug.coverageSelected.steps.length)}});
 assert.equal(get('coverage-replay-step').textContent,`${debug.coverageSelected.steps.length} / ${debug.coverageSelected.steps.length}`);
 get('coverage-replay-play').listeners.click();const tick=intervals.values().next().value;
 for(let i=0;i<debug.coverageSelected.steps.length;i++)tick();
 assert.equal(intervals.size,0);assert.equal(get('coverage-replay-play').textContent,'▶ Play');
 get('coverage-replay-reset').listeners.click();get('coverage-replay-play').listeners.click();debug.activateTab('supervised');assert.equal(intervals.size,0);
 await get('refresh').listeners.click();assert.equal(get('coverage-content').hidden,false);
 assert(!get('coverage-rollout-table').innerHTML.includes('NaN'));assert(!get('coverage-paired-table').innerHTML.includes('NaN'));
 get('coverage-replay-reset').listeners.click();get('coverage-replay-play').listeners.click();assert.equal(intervals.size,1);
 get('coverage-study').value=equal?'coverage':'equal_support';get('coverage-study').listeners.change();assert.equal(intervals.size,0,'Switching coverage studies stops replay');
 get('coverage-study').value=studyKey;get('coverage-study').listeners.change();assert.equal(get('coverage-content').hidden,false);assert.equal(debug.coverageSelected.map_seed,debug.coverage.protocol.dataset.heldout_map_seeds[0],'Study switch restores its own first fresh map');

 console.log(`${studyKey}: ${debug.coverage.aggregate.length} rollout aggregates, ${debug.coverage.state_aggregate.length} state aggregates, ${debug.coverage.loss_aggregate.length} loss windows, all ${recordings} recordings, both conditions and action modes, ${supportRecords[0].by_map.length} layout cells and ${supportRecords[0].by_time_bucket.length} time buckets, seven gate interpretations, final outcome cards and mode changes, paired differences, Q labels, playback and refresh passed.`);
}else{
 assert.equal(get('coverage-content').hidden,true);assert.equal(get('coverage-empty').hidden,false);
 for(const id of ['coverage-train-chart','coverage-fresh-chart','coverage-train-agreement','coverage-fresh-agreement','coverage-loss-chart'])assert.equal(get(id).innerHTML,'');
 console.log(`Missing ${studyKey} comparison stays empty; no fabricated measurements.`);
}

}

if(debug.fixed?.aggregate?.length){
 assert.equal(get('fixed-content').hidden,false);
 const chartIds=['fixed-train-chart','fixed-fresh-chart','fixed-train-agreement','fixed-fresh-agreement','fixed-loss-chart'];
 for(const id of chartIds)assert(get(id).innerHTML.includes('<svg'));
 for(const condition of ['exact_q','double_dqn'])if(debug.fixed.aggregate.some(r=>r.condition===condition))assert(get('fixed-fresh-chart').innerHTML.includes(condition==='exact_q'?'Exact targets':'Double DQN targets'));
 assert(get('fixed-gate-results').innerHTML.includes('Efficient:'),'Separate efficiency gates visible');
 assert(get('fixed-state-table').innerHTML.includes('Winnable states'),'Full-state denominator visible');
 assert(get('fixed-paired-table').innerHTML.includes('Success Δ'),'Paired comparison visible');
 const checkFixedOutcomeCards=mode=>{
  const finalUpdate=debug.fixed.protocol.budget.updates_per_seed;
  for(const [id,key,format] of [['success','success_rate',value=>(100*value).toFixed(1)+'%'],['efficient','efficient_success_rate',value=>(100*value).toFixed(1)+'%'],['steps','mean_steps',value=>value.toFixed(1)]]){
   const values=[...get('fixed-outcome-'+id).innerHTML.matchAll(/<strong[^>]*>([^<]*)<\/strong>/g)].map(match=>match[1]);
   const expected=['exact_q','double_dqn'].map(condition=>{const value=debug.fixed.aggregate.find(r=>r.condition===condition && r.panel==='heldout' && r.mode===mode && r.checkpoint===finalUpdate)?.[key];return Number.isFinite(value)?format(value):'—';});
   assert.deepEqual(values,expected,'Final fresh '+key+' cards match saved '+mode+' aggregates');
  }
 };
 checkFixedOutcomeCards('greedy');
 assert(get('fixed-outcome-success').innerHTML.includes('Primary'));assert(get('fixed-outcome-efficient').innerHTML.includes('Secondary'));
 assert(debug.fixedSelected,'Default fixed-data recording selected');
 const originalRun={...debug.fixed.run},originalGates=debug.fixed.gates;
 for(const [run,gates,phrase] of [
  [{status:'complete'}, {eligible:true,per_condition:[{condition:'exact_q',fresh:true},{condition:'double_dqn',fresh:true}]},'Both target procedures pass'],
  [{status:'complete'}, {eligible:true,per_condition:[{condition:'exact_q',fresh:true},{condition:'double_dqn',fresh:false}]},'Exact targets pass'],
  [{status:'complete'}, {eligible:true,per_condition:[{condition:'exact_q',fresh:false},{condition:'double_dqn',fresh:true}]},'Double DQN targets pass'],
  [{status:'complete'}, {eligible:true,per_condition:[{condition:'exact_q',fresh:false},{condition:'double_dqn',fresh:false}]},'Neither target procedure'],
  [{status:'complete',interpretation:'smoke_or_protocol_deviation_not_gate_evidence'}, {eligible:false},'Smoke or protocol-deviation'],
  [{status:'incomplete',interpretation:'incomplete_not_gate_evidence'}, {eligible:false},'Incomplete comparison'],
  [{status:'inconsistent_not_gate_evidence'}, {eligible:false},'Consistency check failed'],
 ]){
  debug.fixed.run=run;debug.fixed.gates=gates;debug.renderFixed();assert(get('fixed-gate-note').textContent.includes(phrase),'Fixed-data interpretation '+phrase);
 }
 debug.fixed.run=originalRun;debug.fixed.gates=originalGates;debug.renderFixed();
 const originalAggregate=debug.fixed.aggregate,originalStates=debug.fixed.state_aggregate,originalLosses=debug.fixed.loss_aggregate;
 debug.fixed.run={status:'incomplete_admission_cap',interpretation:'incomplete_not_gate_evidence'};debug.fixed.gates={eligible:false};
 debug.fixed.aggregate=originalAggregate.filter(r=>r.condition==='exact_q');debug.fixed.state_aggregate=originalStates.filter(r=>r.condition==='exact_q');debug.fixed.loss_aggregate=originalLosses.filter(r=>r.condition==='exact_q');debug.renderFixed();
 assert(get('fixed-gate-note').textContent.includes('Incomplete comparison'));
 for(const id of chartIds)assert(!get(id).innerHTML.includes('Double DQN targets'),'Incomplete comparison does not invent a missing curve');
 checkFixedOutcomeCards('greedy');
 assert(get('fixed-outcome-success').innerHTML.includes('>—</strong>'),'Absent condition has no substituted final outcome');
 debug.fixed.aggregate=[];debug.renderFixed();assert.equal(get('fixed-content').hidden,true);for(const id of [...chartIds,'fixed-outcome-success','fixed-outcome-efficient','fixed-outcome-steps'])assert.equal(get(id).innerHTML,'');
 const originalProvenance=debug.fixed.provenance;
 debug.fixed.provenance={stop_reason:'Archive mismatch <example> & paired check failed'};debug.fixed.run={status:'inconsistent_not_gate_evidence'};
 debug.renderFixed();assert.equal(get('fixed-content').hidden,true);assert.equal(get('fixed-empty-title').textContent,'Consistency check failed.');assert(get('fixed-empty-message').textContent.includes('Archive mismatch <example> & paired check failed'),'Empty consistency state preserves literal reason through textContent');
 debug.fixed.aggregate=originalAggregate;debug.renderFixed();assert(get('fixed-gate-note').textContent.includes('Consistency check failed'));assert(get('fixed-gate-note').textContent.includes('Archive mismatch <example> & paired check failed'),'Consistency banner preserves literal reason through textContent');
 debug.fixed.provenance=originalProvenance;
 debug.fixed.run=originalRun;debug.fixed.gates=originalGates;debug.fixed.aggregate=originalAggregate;debug.fixed.state_aggregate=originalStates;debug.fixed.loss_aggregate=originalLosses;debug.renderFixed();
 const stableIds=['fixed-train-agreement','fixed-fresh-agreement','fixed-loss-chart','fixed-paired-table','fixed-gate-results','fixed-fit-metrics'];
 const before=new Map(stableIds.map(id=>[id,get(id).innerHTML]));
 get('fixed-epsilon').value='epsilon_0_1';get('fixed-epsilon').listeners.change();
 checkFixedOutcomeCards('epsilon_0_1');assert(get('fixed-outcome-caption').textContent.includes('ε = 0.1'));
 for(const id of stableIds)assert.equal(get(id).innerHTML,before.get(id),'Exploration preserves '+id);
 assert(get('fixed-fresh-chart').innerHTML.includes('Optimizer updates per seed'),'Fixed-data axis counts optimizer updates');
 let recordings=0;
 for(const row of debug.fixed.trajectories){
  get('fixed-epsilon').value=row.policy==='learner'?row.mode:'greedy';get('fixed-epsilon').listeners.change();
  get('fixed-replay-condition').value=row.condition;get('fixed-replay-condition').listeners.change();
  get('fixed-replay-controller').value=row.policy;get('fixed-replay-controller').listeners.change();
  get('fixed-replay-checkpoint').value=String(row.checkpoint);get('fixed-replay-checkpoint').listeners.change();
  get('fixed-replay-panel').value=row.panel;get('fixed-replay-panel').listeners.change();
  get('fixed-replay-seed').value=String(row.seed);get('fixed-replay-seed').listeners.change();
  assert.equal(debug.fixedSelected,row,'Every fixed-data trajectory reachable');
  if(row.steps.length){
   assert(get('fixed-step-diagnostics').innerHTML.includes('Decision for step 1'));
   if(row.policy==='learner'){assert(get('fixed-step-diagnostics').innerHTML.includes('Learned Q'));assert(get('fixed-step-diagnostics').innerHTML.includes('Optimal Q'));}
   debug.fixedFrame=row.steps.length;debug.drawFixedWorld();
   assert(get('fixed-step-diagnostics').innerHTML.includes('Decision for step '+row.steps.length));
   assert(get('fixed-step-diagnostics').innerHTML.includes('the position after that step'));
  }
  assert(get('fixed-world-caption').textContent.startsWith('Rule A'),'Rule visible from initial frame onward');
  recordings++;
 }
 // Verify each paired network sees its own mode-aligned curve and shared reference panel.
 for(const condition of ['exact_q','double_dqn'])for(const mode of ['greedy','epsilon_0_1'])for(const panel of ['train','heldout']){
  get('fixed-epsilon').value=mode;get('fixed-epsilon').listeners.change();
  get('fixed-replay-condition').value=condition;get('fixed-replay-condition').listeners.change();
  get('fixed-replay-controller').value='learner';get('fixed-replay-controller').listeners.change();
  get('fixed-replay-panel').value=panel;get('fixed-replay-panel').listeners.change();
  assert.equal(debug.fixedSelected.condition,condition);assert.equal(debug.fixedSelected.mode,mode);assert.equal(debug.fixedSelected.panel,panel);
  assert(get('fixed-reference-bars').innerHTML.includes('Exact targets'));assert(get('fixed-reference-bars').innerHTML.includes('Double DQN targets'));
  assert(!get('fixed-reference-bars').innerHTML.includes('Historical'),'No historical panel mixed into fixed comparison');
 }
 get('fixed-replay-reset').listeners.click();assert.equal(get('fixed-replay-step').textContent,`0 / ${debug.fixedSelected.steps.length}`);
 get('fixed-replay-play').listeners.click();assert.equal(intervals.size,1);assert.equal(get('fixed-replay-play').textContent,'Ⅱ Pause');
 intervals.values().next().value();assert.equal(get('fixed-replay-step').textContent,`1 / ${debug.fixedSelected.steps.length}`);
 if(intervals.size)get('fixed-replay-play').listeners.click();assert.equal(intervals.size,0);
 get('fixed-replay-scrub').listeners.input({target:{value:String(debug.fixedSelected.steps.length)}});
 assert.equal(get('fixed-replay-step').textContent,`${debug.fixedSelected.steps.length} / ${debug.fixedSelected.steps.length}`);
 get('fixed-replay-play').listeners.click();const tick=intervals.values().next().value;
 for(let i=0;i<debug.fixedSelected.steps.length;i++)tick();
 assert.equal(intervals.size,0);assert.equal(get('fixed-replay-play').textContent,'▶ Play');
 get('fixed-replay-reset').listeners.click();get('fixed-replay-play').listeners.click();debug.activateTab('supervised');assert.equal(intervals.size,0);
 await get('refresh').listeners.click();assert.equal(get('fixed-content').hidden,false);
 assert(!get('fixed-rollout-table').innerHTML.includes('NaN'));assert(!get('fixed-paired-table').innerHTML.includes('NaN'));
 console.log(`Fixed-data targets: ${debug.fixed.aggregate.length} rollout aggregates, ${debug.fixed.state_aggregate.length} state aggregates, ${debug.fixed.loss_aggregate.length} loss windows, all ${recordings} recordings, both conditions and action modes, seven gate interpretations, final outcome cards and mode changes, paired differences, Q labels, playback and refresh passed.`);
}else{
 assert.equal(get('fixed-content').hidden,true);assert.equal(get('fixed-empty').hidden,false);
 for(const id of ['fixed-train-chart','fixed-fresh-chart','fixed-train-agreement','fixed-fresh-agreement','fixed-loss-chart'])assert.equal(get(id).innerHTML,'');
 console.log('Missing fixed-data comparison stays empty; no fabricated measurements.');
}

if(debug.supervised?.aggregate?.length){
 assert.equal(get('supervised-content').hidden,false);
 for(const id of ['supervised-success-chart','supervised-agreement-chart','supervised-loss-chart'])assert(get(id).innerHTML.includes('<svg'));
 assert(!get('supervised-gate-note').textContent.includes('_'),'Supervised status enum translated');
 // Exercise presentation branches in memory; never write modified study data.
 const originalInterpretation=debug.supervised.run.interpretation;
 for(const [status,phrase] of [
  ['fresh_competence_gate_met','Fresh-layout supervised gate met'],
  ['fresh_pass_fit_unresolved','training fit remains unresolved'],
  ['training_fit_only','Training fit established'],
  ['training_fit_not_established','Training fit was not established'],
  ['smoke_or_protocol_deviation_not_gate_evidence','Smoke or protocol-deviation run'],
  ['incomplete_not_gate_evidence','Incomplete study'],
 ]){
  debug.supervised.run.interpretation=status;debug.renderSupervised();
  assert(get('supervised-gate-note').textContent.includes(phrase),'Interpretation branch '+status);
 }
 debug.supervised.run.interpretation=originalInterpretation;debug.renderSupervised();
 assert(get('supervised-gate-results').innerHTML.includes('Training fit:'),'Supervised gates visible');
 assert(get('supervised-state-table').innerHTML.includes('Winnable states'),'State agreement denominator visible');
 assert(debug.supervisedSelected,'Default supervised recording selected');
 const agreement=get('supervised-agreement-chart').innerHTML,loss=get('supervised-loss-chart').innerHTML;
 get('supervised-epsilon').value='epsilon_0_1';get('supervised-epsilon').listeners.change();
 assert.equal(get('supervised-agreement-chart').innerHTML,agreement,'Rollout exploration does not alter exhaustive greedy agreement');
 assert.equal(get('supervised-loss-chart').innerHTML,loss,'Rollout exploration does not alter recorded training loss');
 assert(get('supervised-success-chart').innerHTML.includes('Optimizer updates per seed'),'Supervised axis counts updates');
 assert(!get('supervised-success-chart').innerHTML.includes('Training transitions'),'No RL transition axis for supervised fit');
 let modeCombinations=0;
 for(const mode of ['greedy','epsilon_0_1']){
  get('supervised-epsilon').value=mode;get('supervised-epsilon').listeners.change();
  for(const policy of ['learner','prior_stream','random_actions','shortest_path']){
   get('supervised-replay-controller').value=policy;get('supervised-replay-controller').listeners.change();
   for(const panel of policy==='prior_stream'?['heldout']:['train','heldout']){
    get('supervised-replay-panel').value=panel;get('supervised-replay-panel').listeners.change();
    assert.equal(debug.supervisedSelected.policy,policy);assert.equal(debug.supervisedSelected.panel,panel);
    if(['learner','prior_stream'].includes(policy))assert.equal(debug.supervisedSelected.mode,mode);
    modeCombinations++;
   }
  }
 }
 let recordings=0;
 for(const row of debug.supervised.trajectories){
  get('supervised-epsilon').value=['learner','prior_stream'].includes(row.policy)?row.mode:'greedy';get('supervised-epsilon').listeners.change();
  get('supervised-replay-controller').value=row.policy;get('supervised-replay-controller').listeners.change();
  get('supervised-replay-checkpoint').value=String(row.checkpoint);get('supervised-replay-checkpoint').listeners.change();
  get('supervised-replay-panel').value=row.panel;get('supervised-replay-panel').listeners.change();
  get('supervised-replay-seed').value=String(row.seed);get('supervised-replay-seed').listeners.change();
  assert.equal(debug.supervisedSelected,row,'Every exact-target-study recording reachable');
  assert(get('supervised-step-diagnostics').innerHTML.includes('Decision for step 1'));
  debug.supervisedFrame=row.steps.length;debug.drawSupervisedWorld();
  assert(get('supervised-step-diagnostics').innerHTML.includes('Decision for step '+row.steps.length));
  assert(get('supervised-step-diagnostics').innerHTML.includes('the position after that step'));
  if(row.policy==='prior_stream')assert.equal(row.panel,'heldout');
  recordings++;
 }
 get('supervised-replay-reset').listeners.click();
 assert.equal(get('supervised-replay-step').textContent,`0 / ${debug.supervisedSelected.steps.length}`);
 get('supervised-replay-play').listeners.click();assert.equal(intervals.size,1);
 assert.equal(get('supervised-replay-play').textContent,'Ⅱ Pause');
 intervals.values().next().value();
 assert.equal(get('supervised-replay-step').textContent,`1 / ${debug.supervisedSelected.steps.length}`);
 if(intervals.size)get('supervised-replay-play').listeners.click();assert.equal(intervals.size,0);
 get('supervised-replay-scrub').listeners.input({target:{value:String(debug.supervisedSelected.steps.length)}});
 assert.equal(get('supervised-replay-step').textContent,`${debug.supervisedSelected.steps.length} / ${debug.supervisedSelected.steps.length}`);
 get('supervised-replay-play').listeners.click();const tick=intervals.values().next().value;
 for(let i=0;i<debug.supervisedSelected.steps.length;i++)tick();
 assert.equal(intervals.size,0);assert.equal(get('supervised-replay-play').textContent,'▶ Play');
 get('supervised-replay-reset').listeners.click();get('supervised-replay-play').listeners.click();debug.activateTab('competence');assert.equal(intervals.size,0);
 await get('refresh').listeners.click();assert.equal(get('supervised-content').hidden,false);
 console.log(`Exact targets: ${debug.supervised.aggregate.length} rollout aggregates, ${debug.supervised.state_aggregate.length} state aggregates, all ${recordings} recordings, ${modeCombinations} mode/policy/panel combinations, six interpretation branches, Q labels, play/pause/auto-stop/reset/scrub/tab-stop/refresh passed.`);
}else{
 assert.equal(get('supervised-content').hidden,true);assert.equal(get('supervised-empty').hidden,false);
 for(const id of ['supervised-success-chart','supervised-agreement-chart','supervised-loss-chart'])assert.equal(get(id).innerHTML,'');
 console.log('Missing exact-target data stays empty; no fabricated measurements.');
}
if(debug.competence?.aggregate?.length){
 assert.equal(get('competence-content').hidden,false);
 assert(get('competence-support-chart').innerHTML.includes('<svg'));
 assert(get('competence-fresh-chart').innerHTML.includes('<svg'));
 assert.equal(get('competence-support-cards').buttons.length,3);
 assert(!get('competence-gate-note').textContent.includes('_'),'Interpretation enum translated');
 assert(get('competence-gate-results').innerHTML.includes('Final greedy fresh maps'),'Per-seed gate results visible');
 assert(get('competence-table').innerHTML.includes('Winnable states'),'Optimality denominator visible');
 assert(debug.selected,'Default recorded trajectory selected');
 assert(get('competence-step-diagnostics').innerHTML.includes('Decision for step 1'),'First pre-action decision displayed');
 let combinations=0;
 for(const mode of ['greedy','epsilon_0_1']){
  get('competence-epsilon').value=mode;get('competence-epsilon').listeners.change();
  for(const condition of ['fixed_1','fixed_16','stream']){
   get('competence-support-cards').buttons.find(b=>b.dataset.support===condition).listeners.click();
   for(const policy of ['learner','random_actions','shortest_path']){
    get('competence-replay-controller').value=policy;get('competence-replay-controller').listeners.change();
    for(const panel of ['probe','heldout']){
     get('competence-replay-panel').value=panel;get('competence-replay-panel').listeners.change();
     assert.equal(debug.selected.policy,policy);assert.equal(debug.selected.panel,panel);assert.equal(debug.selected.condition,condition);
     if(policy==='learner')assert.equal(debug.selected.mode,mode);
     debug.frame=debug.selected.steps.length;debug.drawCompetenceWorld();
     assert(get('competence-replay-step').textContent.startsWith(String(debug.selected.steps.length)));
     combinations++;
    }
   }
  }
 }
 let recordings=0;
 for(const row of debug.competence.trajectories){
  get('competence-epsilon').value=row.policy==='learner'?row.mode:'greedy';get('competence-epsilon').listeners.change();
  get('competence-support-cards').buttons.find(b=>b.dataset.support===row.condition).listeners.click();
  get('competence-replay-controller').value=row.policy;get('competence-replay-controller').listeners.change();
  get('competence-replay-checkpoint').value=String(row.checkpoint);get('competence-replay-checkpoint').listeners.change();
  get('competence-replay-panel').value=row.panel;get('competence-replay-panel').listeners.change();
  get('competence-replay-seed').value=String(row.seed);get('competence-replay-seed').listeners.change();
  assert.equal(debug.selected,row,'Every exported trajectory reachable through selectors');
  assert(get('competence-step-diagnostics').innerHTML.includes('Decision for step 1'));
  debug.frame=row.steps.length;debug.drawCompetenceWorld();
  assert(get('competence-step-diagnostics').innerHTML.includes('Decision for step '+row.steps.length));
  assert(get('competence-step-diagnostics').innerHTML.includes('the position after that step'));
  recordings++;
 }
 assert(!get('competence-table').innerHTML.includes('NaN'));
 console.log(`All ${recordings} exported recordings reachable; first and final pre-action Q labels verified.`);
 get('competence-replay-reset').listeners.click();
 assert.equal(get('competence-replay-step').textContent,`0 / ${debug.selected.steps.length}`);
 get('competence-replay-play').listeners.click();
 assert.equal(get('competence-replay-play').textContent,'Ⅱ Pause');
 assert.equal(intervals.size,1);
 intervals.values().next().value();
 assert.equal(get('competence-replay-step').textContent,`1 / ${debug.selected.steps.length}`);
 if(intervals.size){get('competence-replay-play').listeners.click();assert.equal(intervals.size,0);}
 get('competence-replay-scrub').listeners.input({target:{value:String(debug.selected.steps.length)}});
 assert.equal(get('competence-replay-step').textContent,`${debug.selected.steps.length} / ${debug.selected.steps.length}`);
 get('competence-replay-play').listeners.click();
 const tick=intervals.values().next().value;
 for(let i=0;i<debug.selected.steps.length;i++)tick();
 assert.equal(intervals.size,0);assert.equal(get('competence-replay-play').textContent,'▶ Play');
 get('competence-replay-reset').listeners.click();get('competence-replay-play').listeners.click();debug.activateTab('adaptation');
 assert.equal(intervals.size,0,'Switching tracks stops replay');
 await get('refresh').listeners.click();
 assert.equal(get('refresh').disabled,false);assert.equal(get('competence-content').hidden,false);
 get('study-run').value='pilot_v1';await get('study-run').listeners.change();
 assert.equal(get('adaptation-content').hidden,false,'Archived adaptation remains loadable');
 assert.equal(get('competence-content').hidden,false,'Competence survives adaptation selector change');
 console.log('Play, pause, auto-stop, reset, scrub, tab-stop, refresh, and archived-adaptation controls passed.');
 console.log(`Real competence data: ${debug.competence.aggregate.length} aggregate rows; ${combinations} mode/support/controller/panel replay combinations; ending frames and diagnostic rendering passed.`);
}else{
 assert.equal(get('competence-content').hidden,true);assert.equal(get('competence-empty').hidden,false);assert.equal(get('competence-support-chart').innerHTML,'');
 console.log('Missing competence data stays empty; both historical tracks render without errors.');
}
for(const track of ['coverage','fixed','supervised','competence','adaptation','provenance']){debug.activateTab(track);for(const name of ['coverage','fixed','supervised','competence','adaptation','provenance'])assert.equal(get(name+'-panel').hidden,name!==track);}
assert.equal(get('refresh').disabled,false);
console.log('Six-track activation and data refresh controls passed. DOM/canvas stubs validate code paths only, not visual layout.');
