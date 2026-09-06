// Exercise saved-study controls with DOM/canvas stubs; does not test browser layout.
// Run from any directory: node /path/to/Q6/scripts/check_dashboard.mjs [competence-results.json] [--supervised supervised-results.json] [--missing-supervised]
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
let competencePath=null,supervisedPath=null,missingSupervised=false;
for(let i=2;i<process.argv.length;i++){
 const arg=process.argv[i];
 if(arg==='--supervised'){supervisedPath=process.argv[++i];assert(supervisedPath,'--supervised needs a real results path');}
 else if(arg==='--missing-supervised')missingSupervised=true;
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
 const override=relative==='data/competence.json'?competencePath:relative==='data/supervised.json'?supervisedPath:null;
 const target=override||full;
 if(!fs.existsSync(target))return {status:404,ok:false};
 return {status:200,ok:true,json:async()=>JSON.parse(fs.readFileSync(target,'utf8'))};
}};
const context=vm.createContext(sandbox);
await vm.runInContext(`(async()=>{${source}\n globalThis.labDebug={activateTab,renderCompetence,renderCompetenceComparison,selectCompetenceTrajectory,drawCompetenceWorld,loadData,renderSupervised,get supervised(){return supervised;},get supervisedSelected(){return supervisedTrajectory;},drawSupervisedWorld,set supervisedFrame(value){supervisedFrame=value;},get competence(){return competence;},get selected(){return competenceTrajectory;},get condition(){return competenceCondition;},set frame(value){competenceFrame=value;}};})()`,context);
const get=id=>elements.get(id),debug=sandbox.labDebug;
assert.equal(get('adaptation-content').hidden,false,'Existing adaptation data render');
assert.equal(get('provenance-content').hidden,false,'Existing provenance data render');
assert(get('success-chart').innerHTML.includes('<svg'),'Existing adaptation curve render');
assert(get('provenance-bars').innerHTML.includes('bar-row'),'Existing provenance bars render');
assert(!get('load-status').textContent.includes('Could not load'),'No data errors');
assert.equal(get('tab-supervised').attributes['aria-selected'],'true','Exact targets is the default track');
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
for(const track of ['supervised','competence','adaptation','provenance']){debug.activateTab(track);for(const name of ['supervised','competence','adaptation','provenance'])assert.equal(get(name+'-panel').hidden,name!==track);}
assert.equal(get('refresh').disabled,false);
console.log('Four-track activation and data refresh controls passed. DOM/canvas stubs validate code paths only, not visual layout.');
