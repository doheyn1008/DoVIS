// Run the actual dashboard script with a minimal DOM and deliberately ordered fetches.
// Rendering and browser compatibility remain covered by browser_smoke.py.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../mutual_lab/static/app.js', import.meta.url), 'utf8');

function run(id, seed) {
  return {
    summary: {run_id:id, seed, mode:'demo', status:'completed', started_at:'2026-01-01T00:00:00Z',
      exchanges_completed:0, threshold:.9, sustained_checks:2, directions:[], transport:{requests:0},
      agents:[{name:'A',language:'en'}, {name:'B',language:'ko'}]},
    points: [{source:'A',listener:'B',exchange:0,accuracy:.25,hits:1,n:4,invalid_answers:0,
      wilson_low:.01,wilson_high:.9}], dialogue:[],
  };
}

async function dashboard() {
  const elements = new Map();
  let comparisonPaints = 0;
  const paint = new Proxy({}, {get:(_, method)=>(...args)=>{
    if (method === 'setLineDash' && JSON.stringify(args[0]) === '[3,5]') comparisonPaints++;
  }});
  class Element {
    constructor(id='') {
      this.id=id; this.textContent=''; this.value=''; this.children=[]; this.options=[];
      this.listeners=new Map(); this.firstChild={textContent:''}; this.style={};
      this.clientWidth=640; this.clientHeight=270; this.classes=new Set();
      this.classList={add:(name)=>this.classes.add(name), remove:(name)=>this.classes.delete(name),
        toggle:(name,enabled)=>enabled ? this.classes.add(name) : this.classes.delete(name)};
    }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) {
      this.children=children;
      if (this.id === 'compare') { this.options=[...children]; this.value=children[0]?.value || ''; }
    }
    add(option) { this.options.push(option); }
    addEventListener(type, listener) { this.listeners.set(type, listener); }
    setAttribute() {}
    removeAttribute() {}
    querySelectorAll() { return Array.from({length:4},()=>new Element()); }
    getContext() { return paint; }
  }
  const element = (id)=>{
    if (!elements.has(id)) elements.set(id,new Element(id));
    return elements.get(id);
  };
  element('notice').textContent='Initial notice';
  element('seed').value='29'; element('maximum').value='10';
  element('lang-a').value='ko'; element('lang-b').value='ja';
  const details=new Map([['A',run('A',1)],['B',run('B',2)],['C',run('C',3)],['D',run('D',29)]]);
  const listing={runs:['A','B','C'].map(id=>details.get(id).summary),demo_running:false,job_error:null,demo_run_id:null};
  const held=new Map();
  const response=(body)=>({ok:true,json:async()=>body});
  const context=vm.createContext({
    document:{getElementById:element,createElement:()=>new Element(),createTextNode:(text)=>({textContent:text})},
    window:{devicePixelRatio:1},
    Option:class {constructor(text,value){this.textContent=text;this.value=value;}},
    ResizeObserver:class {observe(){}}, setInterval:()=>1,clearInterval:()=>{},
    fetch:(path,options)=>{
      const key=`${options?.method || 'GET'} ${path}`;
      const waiting=held.get(key)?.shift();
      if (waiting) return waiting;
      if (path === '/api/runs') return Promise.resolve(response(listing));
      if (path === '/api/demo') return Promise.resolve(response({status:'started',mode:'demo'}));
      const detail=details.get(decodeURIComponent(path.slice('/api/run/'.length)));
      assert.ok(detail,`Unexpected fetch: ${path}`);
      return Promise.resolve(response(detail));
    },
  });
  vm.runInContext(source,context,{filename:'app.js'});
  const pump=()=>new Promise(resolve=>setImmediate(resolve));
  await pump();
  const state=()=>vm.runInContext('state',context);
  assert.equal(state().selected,'A');
  assert.equal(state().data.summary.run_id,'A');
  return {
    element, state, details, listing, pump, paints:()=>comparisonPaints,
    hold(path,method='GET') {
      let resolve,reject;
      const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});
      const key=`${method} ${path}`;
      if (!held.has(key)) held.set(key,[]);
      held.get(key).push(promise);
      return {reply:(body)=>resolve(response(body)),fail:reject};
    },
    compare(id) {element('compare').value=id;return element('compare').listeners.get('change')();},
    select(id) {return vm.runInContext(`selectRun(${JSON.stringify(id)})`,context);},
    refresh() {return vm.runInContext('refresh()',context);},
    submit() {return element('demo-form').listeners.get('submit')({preventDefault(){}});},
  };
}

function noComparison(app) {
  assert.equal(app.state().compare,null);
  assert.equal(app.element('compare').value,'');
  assert.equal(app.element('compare-label').textContent,'');
}

test('clearing comparison immediately removes lines and rejects a late response',async()=>{
  const app=await dashboard();
  await app.compare('C');
  assert.equal(app.state().compare.summary.run_id,'C');
  const held=app.hold('/api/run/B');
  const request=app.compare('B');
  await app.compare('');
  noComparison(app);
  const paints=app.paints();
  held.reply(app.details.get('B')); await request;
  noComparison(app); assert.equal(app.paints(),paints);
});

test('changing the base run invalidates an outstanding comparison',async()=>{
  const app=await dashboard(),held=app.hold('/api/run/B');
  const request=app.compare('B');
  await app.select('C');
  held.reply(app.details.get('B')); await request;
  assert.equal(app.state().selected,'C'); noComparison(app);
});

test('a late base-A comparison stays invalid after switching A to C to A',async()=>{
  const app=await dashboard(),held=app.hold('/api/run/B');
  const request=app.compare('B');
  await app.select('C'); await app.select('A');
  held.reply(app.details.get('B')); await request;
  noComparison(app);
});

test('the latest selected comparison wins when responses arrive in reverse order',async()=>{
  const app=await dashboard(),b=app.hold('/api/run/B'),c=app.hold('/api/run/C');
  const first=app.compare('B'),second=app.compare('C');
  c.reply(app.details.get('C')); await second;
  b.reply(app.details.get('B')); await first;
  assert.equal(app.element('compare').value,'C');
  assert.equal(app.state().compare.summary.run_id,'C');
  assert.match(app.element('compare-label').textContent,/Seed 3/);
});

test('B to C to B distinguishes the two requests for B',async()=>{
  const app=await dashboard(),oldB=app.hold('/api/run/B'),c=app.hold('/api/run/C');
  const first=app.compare('B'),second=app.compare('C');
  const newB=app.hold('/api/run/B'),third=app.compare('B');
  newB.reply(run('B',200)); await third;
  c.reply(app.details.get('C')); await second;
  oldB.reply(run('B',100)); await first;
  assert.equal(app.state().compare.summary.seed,200);
  assert.match(app.element('compare-label').textContent,/Seed 200/);
});

test('a stale failed request cannot overwrite the current notice',async()=>{
  const app=await dashboard(),held=app.hold('/api/run/B');
  const request=app.compare('B');
  await app.compare('');
  const notice=app.element('notice').textContent;
  held.fail(new Error('stale failure')); await request;
  assert.equal(app.element('notice').textContent,notice);
  assert.equal(app.element('notice').classes.has('error'),false);
  noComparison(app);
});

test('an active comparison failure is still shown',async()=>{
  const app=await dashboard(),held=app.hold('/api/run/B');
  const request=app.compare('B');
  held.fail(new Error('current failure')); await request;
  assert.equal(app.element('notice').textContent,'current failure');
  assert.equal(app.element('notice').classes.has('error'),true);
});

test('starting a demo invalidates old requests before its POST completes',async()=>{
  const app=await dashboard(),b=app.hold('/api/run/B'),post=app.hold('/api/demo','POST');
  const request=app.compare('B'),submission=app.submit();
  b.reply(app.details.get('B')); await request;
  const beforePost=app.state().compare;
  post.reply({status:'started',mode:'demo'}); await submission;
  app.listing.demo_run_id='D'; app.listing.runs.unshift(app.details.get('D').summary);
  await app.refresh();
  assert.equal(beforePost,null);
  assert.equal(app.state().selected,'D'); noComparison(app);
});

test('ordinary refresh preserves a pending comparison on the same base',async()=>{
  const app=await dashboard(),held=app.hold('/api/run/B');
  const request=app.compare('B');
  await app.refresh();
  held.reply(app.details.get('B')); await request;
  assert.equal(app.element('compare').value,'B');
  assert.equal(app.state().compare.summary.run_id,'B');
});
