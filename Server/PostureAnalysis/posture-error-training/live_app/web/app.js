const $=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let busy=false,sequence=-1;
const names={idle:'대기',loading_model:'모델 준비',connecting:'카메라 연결 중',warming_up:'자세 수집 중',pose_not_visible:'관절 확인 필요',predicting:'실시간 추론 중',error:'오류'};
function showError(message){$('error').textContent=message||'';$('error').classList.toggle('hidden',!message)}
async function post(path,body={}){if(busy)return;busy=true;for(const id of ['start','stop','reset'])$(id).disabled=true;try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error);showError('');render(data)}catch(e){showError(e.message)}finally{busy=false;for(const id of ['start','stop','reset'])$(id).disabled=false}}
function render(s){
 $('source').disabled=s.running;$('exercise').disabled=s.running;$('record').disabled=s.running;if(s.running){$('source').value=s.source;$('exercise').value=s.exercise}
 $('status').textContent=names[s.status]||s.status;$('status').classList.toggle('active',s.status==='predicting');$('message').textContent=s.message||'';$('mode').textContent=s.mode==='video_replay'?'저장 영상 재생 테스트':s.transport==='snapshot'?'ESP32-CAM LIVE · 사진 연속 수신':'ESP32-CAM LIVE';
 $('capture-fps').textContent=s.capture_fps||0;$('processing-fps').textContent=s.processing_fps||0;$('buffer').textContent=`${s.buffer_frames} / ${s.required_frames}`;
 if(sequence!==s.frame_sequence){sequence=s.frame_sequence;$('video').src='/api/frame.jpg?seq='+sequence}
 const previous=s.status==='idle'&&s.last_result;const result=s.result||(previous?s.last_result:null);const errors=result?.detected_errors||[];
 const pending=result?.raw_detected_errors?.length&&!errors.length;
 const resultActive=s.status==='predicting'||Boolean(previous);
 const prefix=previous?'중지 전 마지막 판정':'현재 판정';
 $('live-title').textContent=resultActive?(errors.length?`${prefix}: ${errors.join(' · ')}`:pending?`${prefix}: 오류 후보 재확인 중`:`${prefix}: 선택 오류 미검출`):`${prefix}: ${names[s.status]||s.status}`;
 $('live-text').textContent=resultActive?(errors.length?(result.feedbacks||[]).join(' '):pending?'모델이 오류 후보를 감지했습니다. 최근 3회 중 2회 이상 반복 감지되면 교정 문구를 표시합니다.':'학습한 오류의 감지 기준을 넘지 않았습니다. 전체 자세가 정상이라는 보장은 아닙니다.'):(s.status==='idle'?'실시간 시작을 눌러주세요. 종목 선택 후 전신이 보이도록 운동하세요.':s.status==='warming_up'?`${s.message} · 수집이 완료되어야 오류를 분류합니다.`:s.message||'');
 $('live-feedback').classList.toggle('warning',errors.length>0);
 if(previous)$('live-text').textContent+=' (중지 전 결과이며 현재 영상 판정이 아닙니다.)';
 $('assessment').textContent=resultActive?(errors.length?`${prefix}: ${errors.join(' · ')}`:`${prefix}: 선택 오류 미검출 · 전체 정상 판정 아님`):(s.message||'시작 대기');$('assessment').classList.toggle('warning',errors.length>0);
 $('feedback').innerHTML=result?.feedbacks?.length?result.feedbacks.map(f=>`<li>${esc(f)}</li>`).join(''):`<li>${resultActive?'선택 오류 감지 기준을 넘지 않아 교정 문구가 없습니다.':'자세를 충분히 수집하면 교정 문구를 표시합니다.'}</li>`;
 $('scores').innerHTML=result?Object.entries(result.scores||{}).map(([name,value])=>`<div class="score"><div class="score-label"><span>${esc(name)}</span><b>${value===null?'학습 불가':Number(value).toFixed(3)}</b></div><div class="track"><div class="fill ${errors.includes(name)?'detected':''}" style="width:${value===null?0:Math.max(0,Math.min(100,value*100))}%"></div></div></div>`).join(''):'<p class="muted">아직 유효한 모델 판정이 없습니다.</p>';
 $('raw').textContent=JSON.stringify(result,null,2);$('log').textContent=s.log_path?'판정 로그: '+s.log_path:'';if(s.error)showError(s.error);else if(!busy)showError('');
}
$('start').onclick=()=>post('/api/start',{source:$('source').value,exercise:$('exercise').value,record:$('record').checked});$('stop').onclick=()=>post('/api/stop');$('reset').onclick=()=>post('/api/reset');
(async()=>{try{const c=await (await fetch('/api/config')).json();$('source').value=c.default_source;$('exercise').innerHTML=Object.entries(c.exercises).map(([id,name])=>`<option value="${esc(id)}">${esc(name)}</option>`).join('');$('exercise').value=c.default_exercise;render(await (await fetch('/api/state')).json())}catch(e){showError(e.message)}})();
async function poll(){try{const r=await fetch('/api/state');if(!r.ok)throw Error('서버 응답 오류');render(await r.json())}catch(e){showError('로컬 서버 연결이 끊겼습니다. VS Code 실행 상태를 확인하세요.')}finally{setTimeout(poll,250)}}poll();
