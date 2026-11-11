"use strict";
const $ = (id) => document.getElementById(id);
const names = {en:"English",ko:"한국어",ja:"日本語",es:"Español"};
const state = {selected:null,runs:[],data:null,compare:null,compareRequest:0,replay:0,timer:null,refreshing:false,followingDemo:false};
const percent = (value) => `${(value * 100).toFixed(1)}%`;
const initialNotice = $("notice").textContent;
async function api(path, options) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}
function text(id, value) { $(id).textContent = value; }
function announce(message, error=false) { text("notice", message); $("notice").classList.toggle("error", error); }
function runTitle(run) {
  const agents = run.agents || [];
  return `${names[agents[0]?.language] || "A"} ↔ ${names[agents[1]?.language] || "B"}`;
}
function renderList() {
  $("run-list").replaceChildren();
  text("run-count", String(state.runs.length));
  for (const run of state.runs) {
    const button = document.createElement("button");
    button.className = `run-item${run.run_id === state.selected ? " active" : ""}`;
    const heading = document.createElement("strong");
    const dot = document.createElement("span"); dot.className = `run-dot ${run.status}`;
    heading.append(dot, document.createTextNode(runTitle(run)));
    const detail = document.createElement("small");
    detail.textContent = `Seed ${run.seed} · ${run.exchanges_completed}회 · ${{demo:"데모",endpoint:"문맥 적응",neural:"누적 학습"}[run.mode] || run.mode} · ${new Date(run.started_at).toLocaleTimeString("ko-KR", {hour:"2-digit",minute:"2-digit"})}`;
    button.append(heading, detail);
    button.addEventListener("click", () => selectRun(run.run_id));
    $("run-list").append(button);
  }
  const previous = $("compare").value;
  $("compare").replaceChildren(new Option("다른 실행과 비교", ""));
  for (const run of state.runs.filter(r => r.run_id !== state.selected)) {
    $("compare").add(new Option(`${runTitle(run)} · Seed ${run.seed} · ${run.run_id.slice(9,15)}`, run.run_id));
  }
  if ([...$("compare").options].some(option => option.value === previous)) $("compare").value = previous;
}
function resetComparison() {
  state.compareRequest++; state.compare = null; $("compare").value = ""; drawChart();
}
async function selectRun(id) {
  stopPlayback(); state.followingDemo = false; state.selected = id; state.replay = 0;
  resetComparison();
  await loadSelected(); renderList();
}
async function loadSelected() {
  const selected = state.selected;
  if (!selected) return;
  const data = await api(`/api/run/${encodeURIComponent(selected)}`);
  if (state.selected !== selected) return;
  state.data = data; renderData();
}
async function refresh() {
  if (state.refreshing) return;
  state.refreshing = true;
  try {
    const result = await api("/api/runs"); state.runs = result.runs;
    $("launch").disabled = result.demo_running;
    $("launch").firstChild.textContent = result.demo_running ? "실험 진행 중 " : "실험 시작 ";
    if (result.job_error) { announce(result.job_error, true); state.followingDemo = false; }
    if (state.followingDemo && result.demo_run_id) {
      state.selected = result.demo_run_id; state.followingDemo = false; resetComparison();
    }
    if (!state.selected && state.runs.length && !state.followingDemo) {
      state.selected = state.runs[0].run_id; resetComparison();
    }
    renderList(); await loadSelected();
  } catch(error) { announce(`기록을 불러오지 못했어: ${error.message}`, true); }
  finally { state.refreshing = false; }
}
function renderData() {
  const {summary:run,points} = state.data; if (!run) return;
  text("mode", {demo:"PIPELINE DEMO",endpoint:"CONTEXT PILOT",neural:"PERSISTENT LEARNING"}[run.mode] || run.mode);
  text("run-id", run.run_id); text("status", ({completed:"COMPLETED",running:"RUNNING",failed:"FAILED",cancelled:"CANCELLED"})[run.status] || run.status);
  text("exchanges", `${run.exchanges_completed}회`);
  $("export").href = `/api/run/${encodeURIComponent(run.run_id)}/export`;
  $("export").classList.remove("disabled"); $("export").removeAttribute("aria-disabled");
  const directions = run.directions || [];
  for (let i=0;i<2;i++) {
    const direction = directions[i]; const suffix = i === 0 ? "ab" : "ba";
    text(`score-${suffix}`, direction ? percent(direction.last_accuracy) : "—");
    text(`detail-${suffix}`, direction ? `${names[direction.source_language]} 발화 → ${names[direction.listener_language]} 화자 · 보정 ${percent(direction.last_corrected_accuracy ?? (direction.last_accuracy-.25)/.75)}` : "초기 평가를 준비하고 있어.");
  }
  const complete = directions.length === 2 && directions.every(d => d.sustained_threshold_confirmed_exchange != null);
  const scale = run.threshold_scale === "corrected" ? "우연 보정" : "원정답률";
  text("threshold-time", complete ? `${Math.max(...directions.map(d => d.sustained_threshold_confirmed_exchange))}회` : run.status === "completed" ? "미도달" : ["failed","cancelled"].includes(run.status) ? "중단" : "측정 중");
  text("threshold-detail", `${scale} ${percent(run.threshold)} · ${run.sustained_checks}회 연속 평가 통과`);
  text("target-label", `${scale} 목표 ${Math.round(run.threshold*100)}%`);
  const conditionValues = ["그림 4지선다 · 곡선은 원정답률", "물체 쌍 단위", `대화 0회 · ${directions[0]?.eval_items || "—"}문항/방향`, `${scale} ${percent(run.threshold)} · ${run.sustained_checks}회 연속`];
  [...$("conditions").querySelectorAll("dd")].forEach((node,i) => node.textContent = conditionValues[i]);
  const inclusionLabels = {primary:"1차 조건 충족",positive_baseline_extension:"양수 초기 이해도 확장",uncertain:"초기 이해도 판정 보류",ineligible:"모국어 적격성 미충족",control:"대조 조건"};
  const modelNote = run.mode === "demo" ? "기호학습 데모 · 완전한 시각 인식을 가정해. 언어 사이의 실제 난이도 비교에는 사용할 수 없어." : run.mode === "neural" ? `모국어 이해 가중치로 누적 학습 · ${run.model_profile?.id || ""} · ${inclusionLabels[run.inclusion?.analysis_group] || "사전검사 중"}. 공통 곡선 구간 ${run.common_horizon}회.` : `고정 가중치 + 대화 기록으로 적응해. 초기 외국어 지식은 검증되지 않았어. 모델 요청 ${run.transport.requests}회.`;
  text("measurement-note", `${modelNote} 목표 미도달 실행도 그대로 기록해.`);
  if (run.error) announce(run.error, true);
  else announce(run.mode === "demo" ? initialNotice : run.mode === "neural" ? "제한된 자연어 영역의 누적 학습 파일럿이야. 모국어 검사와 사전검사를 통과한 실행만 해당 분석에 포함해." : "모델 파일럿 실행이야. 초기 외국어 지식, 그림 설명의 정확성, 언어 유지 여부를 별도로 확인해야 해.");
  $("evaluation-rows").replaceChildren();
  for (const point of points) {
    const tr = document.createElement("tr");
    const values = [point.exchange, `${point.source} 발화 → ${point.listener} 화자`, `${point.hits} / ${point.n}`, percent(point.accuracy), percent(point.corrected_accuracy ?? (point.accuracy-.25)/.75), `${percent(point.wilson_low)} – ${percent(point.wilson_high)}`, point.invalid_answers];
    for (const value of values) { const td=document.createElement("td");td.textContent=String(value);tr.append(td); }
    $("evaluation-rows").append(tr);
  }
  renderReplay(); drawChart();
}
function renderReplay() {
  const dialogue = state.data?.dialogue || [];
  state.replay = Math.max(0, Math.min(state.replay, dialogue.length-1));
  const event = dialogue[state.replay];
  $("scrub").max = String(Math.max(0,dialogue.length-1)); $("scrub").value=String(state.replay);
  text("replay-count", `${event?.exchange || 0} / ${dialogue.length}`);
  $("shared-image").hidden=!event; $("image-empty").hidden=!!event;
  $("messages").replaceChildren();
  if (!event) return;
  $("shared-image").src=`/assets/${encodeURIComponent(state.selected)}/${encodeURIComponent(event.image)}`;
  for(const message of event.messages) {
    const box=document.createElement("div");box.className="message";
    const meta=document.createElement("div");meta.className=`message-meta ${message.speaker === state.data.summary.agents[0].name ? "blue" : "purple"}`;
    meta.textContent=`AGENT ${message.speaker} · ${names[message.language] || message.language}`;
    const body=document.createElement("p");body.textContent=message.text;box.append(meta,body);$("messages").append(box);
  }
}
function stopPlayback(){if(state.timer)clearInterval(state.timer);state.timer=null;text("play","▶");$("play").setAttribute("aria-label","대화 재생");}
$("play").addEventListener("click",()=>{
  if(state.timer){stopPlayback();return;}
  if(!state.data?.dialogue.length)return;
  if(state.replay>=state.data.dialogue.length-1)state.replay=0;
  text("play","Ⅱ");$("play").setAttribute("aria-label","재생 일시정지");renderReplay();
  state.timer=setInterval(()=>{if(state.replay>=state.data.dialogue.length-1){stopPlayback();return;}state.replay++;renderReplay();},650);
});
$("previous").addEventListener("click",()=>{stopPlayback();state.replay--;renderReplay();});
$("next").addEventListener("click",()=>{stopPlayback();state.replay++;renderReplay();});
$("scrub").addEventListener("input",()=>{stopPlayback();state.replay=Number($("scrub").value);renderReplay();});
$("intervals").addEventListener("change",drawChart);
$("compare").addEventListener("change",async()=>{
  const id = $("compare").value;
  if (!id || !state.selected) { resetComparison(); return; }
  const request = ++state.compareRequest, base = state.selected;
  const isCurrent = () => request === state.compareRequest && base === state.selected && id === $("compare").value;
  state.compare = null; drawChart();
  try {
    const data = await api(`/api/run/${encodeURIComponent(id)}`);
    if (!isCurrent()) return;
    const baseRun = state.data?.summary, otherRun = data.summary;
    if (baseRun?.mode !== otherRun.mode || ["model_profile","native_training_profile","acquisition_profile","evaluation_profile"].some(key => baseRun?.[key] && JSON.stringify(baseRun[key]) !== JSON.stringify(otherRun[key]))) {
      announce("학습 방식이나 모델 능력 조건이 다른 실행은 같은 조건으로 비교할 수 없어.", true); return;
    }
    state.compare = data; drawChart();
  } catch(error) {
    if (isCurrent()) announce(error.message,true);
  }
});
function chartGeometry(){const width=$("curve").clientWidth,height=$("curve").clientHeight;return{width,height,left:39,right:12,top:20,bottom:30};}
function drawChart(){
  const canvas=$("curve"),g=chartGeometry();if(!g.width)return;
  const ratio=window.devicePixelRatio||1;canvas.width=g.width*ratio;canvas.height=g.height*ratio;
  const ctx=canvas.getContext("2d");ctx.scale(ratio,ratio);ctx.clearRect(0,0,g.width,g.height);
  const points=state.data?.points || [],other=state.compare?.points || [];
  $("chart-empty").classList.toggle("hidden",points.length>0);
  text("compare-label",state.compare ? `점선: Seed ${state.compare.summary.seed} 비교 실행` : "");
  const maxX=Math.max(5,...points.map(p=>p.exchange),...other.map(p=>p.exchange));
  const x=(n)=>g.left+n/maxX*(g.width-g.left-g.right),y=(n)=>g.height-g.bottom-n*(g.height-g.top-g.bottom);
  ctx.font="10px system-ui";ctx.textAlign="right";ctx.textBaseline="middle";
  for(const value of [0,.25,.5,.75,1]){ctx.strokeStyle="#edf0f6";ctx.lineWidth=1;ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(x(0),y(value));ctx.lineTo(x(maxX),y(value));ctx.stroke();ctx.fillStyle="#99a5b7";ctx.fillText(`${Math.round(value*100)}%`,g.left-9,y(value));}
  ctx.textAlign="center";for(let i=0;i<=5;i++){const value=maxX*i/5;ctx.fillStyle="#99a5b7";ctx.fillText(String(Math.round(value)),x(value),g.height-12);}
  const summary=state.data?.summary;
  const rawTarget=summary?.threshold_raw_equivalent ?? (summary?.threshold_scale === "corrected" ? .25+.75*summary.threshold : summary?.threshold ?? .9);
  for(const [value,color] of [[.25,"#cbd4df"],[rawTarget,"#7dae99"]]){ctx.strokeStyle=color;ctx.lineWidth=1;ctx.setLineDash([4,5]);ctx.beginPath();ctx.moveTo(x(0),y(value));ctx.lineTo(x(maxX),y(value));ctx.stroke();}
  function series(collection,comparison){
    const sources=[...new Set(collection.map(p=>p.source))];
    sources.forEach((source,index)=>{
      const group=collection.filter(p=>p.source===source),color=index===0?"#356ae6":"#9861dc";
      if($("intervals").checked&&!comparison&&group.length){ctx.fillStyle=index===0?"#356ae610":"#9861dc10";ctx.beginPath();group.forEach((p,i)=>i?ctx.lineTo(x(p.exchange),y(p.wilson_high)):ctx.moveTo(x(p.exchange),y(p.wilson_high)));[...group].reverse().forEach(p=>ctx.lineTo(x(p.exchange),y(p.wilson_low)));ctx.closePath();ctx.fill();}
      ctx.strokeStyle=color;ctx.globalAlpha=comparison ? 0.45 : 1;ctx.lineWidth=comparison?1.5:2.4;ctx.setLineDash(comparison?[3,5]:[]);ctx.beginPath();group.forEach((p,i)=>i?ctx.lineTo(x(p.exchange),y(p.accuracy)):ctx.moveTo(x(p.exchange),y(p.accuracy)));ctx.stroke();ctx.setLineDash([]);
      if(!comparison)for(const p of group){ctx.beginPath();ctx.fillStyle="#fff";ctx.arc(x(p.exchange),y(p.accuracy),3.1,0,2*Math.PI);ctx.fill();ctx.stroke();}ctx.globalAlpha=1;
    });
  }
  series(other,true);series(points,false);
}
$("curve").addEventListener("pointermove",event=>{
  const points=state.data?.points || [];if(!points.length)return;
  const g=chartGeometry(),rect=$("curve").getBoundingClientRect();
  const maxX=Math.max(5,...points.map(p=>p.exchange),...(state.compare?.points || []).map(p=>p.exchange));
  const value=(event.clientX-rect.left-g.left)/(g.width-g.left-g.right)*maxX;
  const closest=points.reduce((a,b)=>Math.abs(a.exchange-value)<Math.abs(b.exchange-value)?a:b);
  const current=points.filter(p=>p.exchange===closest.exchange);
  text("chart-tooltip",`${closest.exchange}회 대화\n${current.map(p=>`${p.source} → ${p.listener}  ${percent(p.accuracy)} (${p.hits}/${p.n})`).join("\n")}`);
  $("chart-tooltip").classList.remove("hidden");
  $("chart-tooltip").style.left=`${Math.max(0,Math.min(g.width-180,event.clientX-rect.left+10))}px`;
  $("chart-tooltip").style.top=`${Math.max(0,event.clientY-rect.top-65)}px`;
});
$("curve").addEventListener("pointerleave",()=>$("chart-tooltip").classList.add("hidden"));
new ResizeObserver(drawChart).observe($("chart-wrap"));
$("demo-form").addEventListener("submit",async event=>{
  event.preventDefault();$("launch").disabled=true;
  resetComparison();
  try {
    await api("/api/demo",{method:"POST",headers:{"Content-Type":"application/json","X-Mutual-Lab":"1"},
      body:JSON.stringify({seed:Number($("seed").value),max_exchanges:Number($("maximum").value),
        lang_a:$("lang-a").value,lang_b:$("lang-b").value})});
    state.selected=null;state.followingDemo=true;state.replay=0;stopPlayback();resetComparison();
    announce("새 실험을 시작했어. 측정 결과가 순서대로 표시돼.");
    await refresh();
  }
  catch(error){announce(error.message,true);$("launch").disabled=false;}
});
refresh();setInterval(refresh,2500);
