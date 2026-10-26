/** 영상 분석과 반복별 검토 라벨의 로컬 저장을 담당한다. */
const el = id => document.getElementById(id);
let timer, currentId, currentResult, reviewState;
let savedSources=[];
const exerciseNames = {squat:'스쿼트',pushup:'푸시업'};
const preferenceKey='posture-score-youtube-input';
fetch('/rules.json').then(r=>r.json()).then(x=>el('rules').textContent=JSON.stringify(x,null,2));

/** 저장된 분석 목록을 조회한다. 실패하면 안내 문구를 표시한다. */
async function loadJobs(){
  try {
    const jobs=await fetch('/jobs').then(r=>r.json()); el('savedJobs').replaceChildren();
    for(const j of jobs){const b=document.createElement('button'); b.style.margin='5px'; b.textContent=exerciseNames[j.exercise]+' · '+j.repetitions+'회 · '+j.id.slice(0,6);b.onclick=()=>openJob(j.id);el('savedJobs').append(b)}
    if(!jobs.length)el('savedJobs').textContent='아직 보관된 결과가 없습니다.';
  }catch(e){el('savedJobs').textContent='보관된 분석을 불러오지 못했습니다.'}
}

/** 작업을 선택하고 저장된 결과 또는 진행 상태를 표시한다. */
function openJob(id){clearTimeout(timer);el('result').style.display='none';history.replaceState(null,'','?job='+id);poll(id)}

/** 입력 방식을 전환한다. 선택되지 않은 입력은 분석 요청에 포함하지 않는다. */
el('sourceType').onchange=()=>{
  const youtube=el('sourceType').value==='youtube';
  el('youtubeFields').hidden=!youtube;el('fileFields').hidden=youtube;
};

/** 마지막 입력을 이 브라우저에 보관한다. 저장 차단 시에도 분석은 가능하다. */
function savePreferences(){
  try{localStorage.setItem(preferenceKey,JSON.stringify({exercise:el('exercise').value,url:el('youtubeUrl').value,start_seconds:el('startSeconds').value,duration_seconds:el('durationSeconds').value}))}catch(e){}
}

/** 저장된 운동·주소·시간(초)을 입력란에 채운다. 분석은 버튼으로 요청한다. */
function chooseSource(source){
  el('sourceType').value='youtube';el('sourceType').onchange();
  el('exercise').value=source.exercise;el('youtubeUrl').value=source.url;
  el('startSeconds').value=source.start_seconds;el('durationSeconds').value=source.duration_seconds;
  savePreferences();
}

/** 자동 저장된 링크를 불러오고 다시 보기·기록 삭제 동작을 연결한다. */
async function loadSources(initial=false){
  try{
    const response=await fetch('/sources');if(!response.ok)throw Error('저장된 유튜브 목록을 불러오지 못했습니다.');
    savedSources=await response.json();el('savedSources').replaceChildren();
    for(const source of savedSources){
      const row=document.createElement('div');row.className='row';
      const title=document.createElement('strong');title.textContent=exerciseNames[source.exercise]+' · '+source.title;
      const detail=document.createElement('p');detail.textContent=source.url+' · '+source.start_seconds+'초부터 '+source.duration_seconds+'초';
      const choose=document.createElement('button');choose.textContent='불러오기';choose.onclick=()=>{chooseSource(source);el('sourceStatus').textContent='주소와 운동 구간을 불러왔습니다. 영상 분석하기를 눌러 주세요.';};
      row.append(title,detail,choose);
      if(source.last_job_id){const open=document.createElement('button');open.textContent='결과 보기';open.onclick=()=>{chooseSource(source);openJob(source.last_job_id)};row.append(open)}
      const remove=document.createElement('button');remove.textContent='목록에서 삭제';remove.onclick=async()=>{
        remove.disabled=true;
        try{const response=await fetch('/sources/'+source.id,{method:'DELETE'});if(!response.ok)throw Error('기록을 삭제하지 못했습니다.');
          if(el('youtubeUrl').value===source.url&&el('exercise').value===source.exercise&&Number(el('startSeconds').value)===source.start_seconds&&Number(el('durationSeconds').value)===source.duration_seconds){el('youtubeUrl').value='';savePreferences()}
          await loadSources();el('sourceStatus').textContent='링크를 목록에서 삭제했습니다. 보관된 분석 영상은 유지됩니다.';
        }catch(e){el('sourceStatus').textContent=e.message;remove.disabled=false}
      };row.append(remove);el('savedSources').append(row);
    }
    if(!savedSources.length)el('savedSources').textContent='저장된 링크가 없습니다. 새 주소로 분석하면 자동 저장됩니다.';
    if(initial){
      let previous;try{previous=JSON.parse(localStorage.getItem(preferenceKey))}catch(e){}
      if(previous&&exerciseNames[previous.exercise])chooseSource(previous);
      else{const example=savedSources.find(s=>s.exercise===el('exercise').value);if(example)chooseSource(example)}
    }
  }catch(e){el('sourceStatus').textContent=e.message}
}

el('exercise').onchange=()=>{
  if(el('sourceType').value==='youtube'){
    const example=savedSources.find(source=>source.exercise===el('exercise').value);
    if(example)chooseSource(example);else{el('youtubeUrl').value='';savePreferences()}
  }
};
for(const id of ['youtubeUrl','startSeconds','durationSeconds'])el(id).addEventListener('input',savePreferences);

/** 선택한 운동과 영상 입력을 전달한다. 입력 오류·접수 실패는 화면에 표시한다. */
el('start').onclick=async()=>{
  const youtube=el('sourceType').value==='youtube';
  let endpoint,options;
  if(youtube){
    const url=el('youtubeUrl').value.trim();
    const start=Number(el('startSeconds').value),duration=Number(el('durationSeconds').value);
    if(!url){el('status').textContent='먼저 유튜브 영상 주소를 입력해 주세요.';return}
    if(el('startSeconds').value===''||!Number.isFinite(start)||start<0||start>86400||!Number.isFinite(duration)||duration<1||duration>120){el('status').textContent='시작 시간은 0~86400초, 분석 길이는 1~120초로 입력해 주세요.';return}
    endpoint='/analyze-youtube';
    options={method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url,exercise:el('exercise').value,start_seconds:start,duration_seconds:duration})};
  }else{
    const file=el('file').files[0];if(!file){el('status').textContent='먼저 영상을 선택해 주세요.';return}
    if(file.size>300*1024*1024){el('status').textContent='300MB 이하 영상을 선택해 주세요.';return}
    endpoint='/analyze?exercise='+el('exercise').value;options={method:'POST',body:file};
  }
  clearTimeout(timer);el('start').disabled=true;el('result').style.display='none';el('progress').value=0;el('status').textContent='영상을 로컬 분석기에 전달하고 있습니다.';
  try{const r=await fetch(endpoint,options);const data=await r.json();if(!r.ok)throw Error(r.status===409?'다른 영상을 분석 중입니다. 잠시 후 다시 시도하세요.':data.error||'영상 접수 실패 ('+r.status+')');if(youtube){savePreferences();loadSources()}history.replaceState(null,'','?job='+data.id);poll(data.id)}catch(e){el('status').textContent=e.message;el('start').disabled=false}
};

/** 작업 상태를 1초 간격으로 조회하고 완료되면 검토 상태를 불러온다. */
async function poll(id){
 try{const s=await fetch('/jobs/'+id+'/status').then(r=>r.json());el('progress').value=s.percent||0;
 if(s.state==='failed')throw Error(s.error);if(s.state==='unknown')throw Error('작업을 찾을 수 없습니다.');
 if(s.state!=='done'){el('status').textContent=s.message||'Heavy로 관절과 반복을 분석 중입니다. '+(s.percent||0)+'%';timer=setTimeout(()=>poll(id),1000);return}
 currentId=id;currentResult=s.result;reviewState=await fetch('/jobs/'+id+'/review.json').then(r=>r.json());show();el('start').disabled=false;loadJobs();loadSources();
 }catch(e){el('status').textContent=e.message;el('start').disabled=false}
}

/** 결과 요약과 반복별 점수ㆍ검토 입력란을 구성한다. 자동 라벨은 승인하지 않는다. */
function show(){
 const r=currentResult,id=currentId;el('exercise').value=r.exercise;el('result').style.display='block';el('video').src='/jobs/'+id+(r.source?'/score-video.mp4':'/overlay.mp4');
 el('video').muted=true;el('video').load();el('video').play().catch(()=>{});
 el('download').href='/jobs/'+id+'/result.json';el('download').download='posture-score-result.json';
 for(const [element,name] of [['allLabels','labels.csv'],['reviewedLabels','reviewed.csv']]){el(element).href='/jobs/'+id+'/'+name;el(element).download=name}
 el('status').textContent='분석 완료 · 점수는 전문가 미검증 실습 기준입니다.';
 const oldSource=el('sourceInfo');if(oldSource)oldSource.remove();
 if(r.source){const p=document.createElement('p');p.id='sourceInfo';const link=document.createElement('a');link.href=r.source.url;link.target='_blank';link.rel='noopener';link.textContent=r.source.title+' · '+r.source.channel;p.append('원본: ',link,document.createTextNode(' (원본 '+r.source.original_start_seconds+'–'+r.source.original_end_seconds+'초 구간, 관절·점수 추가)'));el('video').parentElement.append(p)}
 const reps=r.repetitions;el('average').textContent=reps.length?(reps.reduce((a,x)=>a+x.score,0)/reps.length).toFixed(1)+' / 100':'측정 불가';
 el('summary').textContent='완료 '+reps.length+'회 · 유효 프레임 '+Math.round(r.valid_ratio*100)+'% · 중단/미완료 '+r.incomplete+'개';
 const subjects=[...new Set(Object.values(reviewState.items).map(x=>x.subject_id))];el('subject').value=subjects.length===1?subjects[0]:'';
 el('reps').replaceChildren();
 if(!reps.length){const p=document.createElement('p');p.textContent='완결된 반복이 없습니다. 측면 구도·관절 가림·시작과 종료 자세를 확인하세요.';el('reps').append(p)}
 for(const rep of reps){
  const rec=reviewState.items[String(rep.number)]||{},div=document.createElement('div');div.className='row';
  const button=document.createElement('button');button.textContent=rep.number+'회 보기';button.onclick=()=>{el('video').currentTime=rep.start;el('video').play()};
  const strong=document.createElement('strong');strong.textContent='자동 초안 '+rep.score+'점';
  const p=document.createElement('p');p.textContent=rep.parts.map(x=>x.name+' '+x.score+'/'+x.maximum).join(' · ');
  const small=document.createElement('small');small.textContent=rep.start+'–'+rep.end+'초';
  const status=document.createElement('p');status.className='reviewStatus';status.textContent=({accepted:'사람 검토: 초안 승인',corrected:'사람 검토: 점수 수정',rejected:'학습 제외'})[rec.status]||'미검토 · 학습 정답 아님';
  const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=!!rec.side_confirmed;label.append(check,' 측면 촬영을 확인했습니다');
  const score=document.createElement('input');score.type='number';score.min=0;score.max=100;score.step=.1;score.value=rec.reviewed_score??rep.score;score.setAttribute('aria-label',rep.number+'회 수정 점수');
  const note=document.createElement('textarea');note.placeholder='검토 메모 (선택)';note.maxLength=1000;note.value=rec.note||'';note.setAttribute('aria-label',rep.number+'회 검토 메모');
  const actions=document.createElement('div');
  for(const [value,text] of [['accepted','초안 승인'],['corrected','수정 점수 저장'],['rejected','학습 제외']]){
   const b=document.createElement('button');b.textContent=text;b.style.margin='4px';b.onclick=async()=>{
    for(const x of actions.children)x.disabled=true;
    try{const response=await fetch('/jobs/'+id+'/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({number:rep.number,status:value,subject_id:el('subject').value.trim(),side_confirmed:check.checked,score:score.value===''?null:Number(score.value),note:note.value,revision:reviewState.revision})});const data=await response.json();if(!response.ok)throw Error(data.error||'저장 실패');reviewState=data;status.textContent='저장됨 · '+({accepted:'초안 승인',corrected:'수정 점수 '+score.value+'점',rejected:'학습 제외'})[value]}
    catch(e){status.textContent=e.message}finally{for(const x of actions.children)x.disabled=false}
   };actions.append(b)
  }
  div.append(button,strong,p,small,status,label,score,note,actions);el('reps').append(div)
 }
}
loadJobs();loadSources(true);const saved=new URLSearchParams(location.search).get('job');if(saved&&/^[a-f0-9]{32}$/.test(saved))openJob(saved);
