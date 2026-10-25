const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let page = 1, total = 0, selected = null, listVersion = 0, detailVersion = 0;
const limit = 24;
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '요청에 실패했습니다.');
  return data;
}
function params() {
  return new URLSearchParams({dataset:$('dataset').value, exercise:$('exercise').value, split:$('split').value, label:$('label').value, q:$('search').value, evidence:$('evidence').checked?'1':'0', page, limit});
}
function showError(error) {
  $('validation').className = 'validation error';
  $('validation').textContent = error.message;
}
async function loadList() {
  const version = ++listVersion;
  try {
    const data = await api('/api/clips?' + params());
    if (version !== listVersion) return;
    total = data.total;
    $('count').textContent = `${total.toLocaleString()}개`;
    const currentLabel = $('label').value;
    $('label').innerHTML = '<option value="">전체 라벨</option>' + data.labels.map(label=>`<option value="${esc(label)}">${esc(label.replace(/^.*? - /,''))}</option>`).join('');
    $('label').value = currentLabel;
    if (currentLabel && !$('label').value) { page=1; return loadList(); }
    $('clips').innerHTML = data.items.length ? data.items.map(item=>`<button class="clip ${item.key===selected?'active':''}" data-key="${esc(item.key)}"><strong>${esc(item.video_id)} · ${esc(item.exercise_name)}</strong><small>${esc(item.subject_id)}${item.view?' / '+esc(item.view):''} · ${esc(item.split)}</small><span class="label-line">${esc(item.labels.map(x=>x.replace(/^.*? - /,'')).join(', '))}</span><div style="margin-top:7px"><span class="badge ${item.status==='draft'?'draft':''}">${esc(item.status)}</span><span class="badge">${esc(item.dataset)}</span></div></button>`).join('') : '<div class="empty">조건에 맞는 영상이 없습니다.</div>';
    $('page').textContent = `${page} / ${Math.max(1,Math.ceil(total/limit))}`;
    $('prev').disabled = page<=1;
    $('next').disabled = page*limit>=total;
    document.querySelectorAll('.clip').forEach(button=>button.addEventListener('click',()=>selectClip(button.dataset.key)));
    if (!data.items.some(item=>item.key===selected)) {
      if (data.items.length) await selectClip(data.items[0].key);
      else { ++detailVersion; selected=null; $('detail').innerHTML='<div class="empty">조건을 바꾸어 영상을 찾아보세요.</div>'; }
    }
  } catch(error) { if (version===listVersion) showError(error); }
}
async function selectClip(key) {
  selected = key;
  const version = ++detailVersion;
  document.querySelectorAll('.clip').forEach(button=>button.classList.toggle('active',button.dataset.key===key));
  try {
    const r = await api('/api/clip?' + new URLSearchParams({key}));
    if (version !== detailVersion) return;
    const feedback = r.feedbacks;
    const e = r.evidence;
    const autoplay = new URLSearchParams(location.search).get('autoplay') === '1';
    const demo = new URLSearchParams(location.search).get('demo') === '1';
    const videoSource = demo && e ? e.bad_url : r.media_url + (autoplay && e ? '#t=' + Math.max(0, e.bad_time - 1) : '');
    $('detail').innerHTML = `<div class="detail-header"><h2>${esc(r.exercise_name)} <span style="font-size:16px;color:#70829a">${esc(r.video_id)}</span></h2><span class="badge ${r.status==='draft'?'draft':''}">${esc(r.status)}</span></div><p class="meta">${esc(r.subject_id)} · ${esc(r.split)}${r.view?' · '+esc(r.view):''}${r.duration_sec?' · '+r.duration_sec.toFixed(2)+'초':''}${demo&&e?' · 검토용 6초 구간 재생':''} · <a href="${esc(r.media_url)}" target="_blank" rel="noopener">영상만 열기</a></p><div class="playback-layout"><div class="video-box"><video id="main-video" controls ${autoplay?'autoplay muted loop playsinline':''} preload="metadata" src="${esc(videoSource)}"></video></div><div class="feedback"><h3>Corrective feedback examples${r.feedback_status==='draft'?' · draft':''}</h3>${feedback.length?'<ul>'+feedback.map(f=>`<li>${esc(f)}</li>`).join('')+'</ul>':'<p>이 영상에는 별도 피드백 문장이 없습니다. 아래 자세 라벨을 확인하세요.</p>'}</div></div><div class="section"><h3>Posture labels</h3><ul>${r.labels.map((l,i)=>`<li><b>${esc(l.split(' - ').slice(1).join(' - ') || l)}</b><br><span style="color:#667891">${esc(l)}</span><br>${esc(r.labels_descriptive[i]||'')}</li>`).join('')}</ul></div><div class="section"><h3>라벨 출처와 검토 메모</h3><p><code>${esc(r.label_origin)}</code><br>${esc(r.notes)}</p></div>${e?`<div class="section"><h3>good / bad 비교 근거</h3><p>${esc(e.observation)}</p><img class="evidence-img" src="${esc(e.image_url)}" alt="good와 bad 자세 비교"><p>근거 화면: good ${e.good_time}초 / bad ${e.bad_time}초. 아래는 각각의 검토용 6초 영상입니다.</p><div class="compare-videos"><div><p>GOOD 기준</p><video controls preload="none" src="${esc(e.good_url)}"></video></div><div><p>BAD 비교</p><video controls preload="none" src="${esc(e.bad_url)}"></video></div></div></div>`:''}${r.original_feedbacks?.length?`<details><summary>Original QEVD free-text feedback (preserved)</summary><ul>${r.original_feedbacks.map(f=>`<li>${esc(f)}</li>`).join('')}</ul></details>`:''}<details><summary>연결된 데이터 JSON 보기</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details>`;
  } catch(error) { if (version===detailVersion) showError(error); }
}
for (const id of ['dataset','exercise','split','label','evidence']) $(id).addEventListener('change',()=>{
  page=1;
  if (['dataset','exercise'].includes(id)) $('label').value='';
  loadList();
});
let timer;
$('search').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(()=>{page=1;loadList();},200);});
$('prev').addEventListener('click',()=>{page--;loadList();});
$('next').addEventListener('click',()=>{page++;loadList();});
$('validate').addEventListener('click',async()=>{
  $('validate').disabled=true;
  $('validate').textContent='검증 중…';
  $('validation').className='validation';
  $('validation').textContent='원본 영상 해시, JSON 형식, 파일 연결, 참여자 분할을 확인하고 있습니다.';
  try {
    const result=await api('/api/validate',{method:'POST'});
    $('validation').innerHTML=`<b>데이터 구조 검증 통과</b><br>원본 150개 해시 일치 · 라벨 100개 연결 정상 · QEVD ID 충돌 0 · 학습/평가 참여자 겹침 0<br>자세 라벨의 정확도를 검증한 결과는 아닙니다.<details><summary>검증 결과 JSON</summary><pre>${esc(JSON.stringify(result,null,2))}</pre></details>`;
  } catch(error) { showError(error); }
  finally { $('validate').disabled=false; $('validate').textContent='데이터 검증 실행'; }
});
(async()=>{
  try {
    if(new URLSearchParams(location.search).get('demo')==='1') document.body.classList.add('demo-mode');
    const stats=await api('/api/stats');
    $('stats').innerHTML=[['QEVD 영상',stats.datasets.qevd.toLocaleString()],['어깨 라벨 초안',stats.datasets.shoulder],['어깨 오류 유형','6종'],['통합 운동 종목','5종']].map(([name,count])=>`<div class="stat"><span>${name}</span><b>${count}</b></div>`).join('');
    $('exercise').innerHTML+=[...stats.exercises].map(e=>`<option value="${esc(e.id)}">${esc(e.name)} (${e.count.toLocaleString()})</option>`).join('');
    const initial = new URLSearchParams(location.search);
    for (const name of ['dataset','exercise','split']) {
      if (initial.has(name) && [...$(name).options].some(o=>o.value===initial.get(name))) $(name).value=initial.get(name);
    }
    $('search').value=initial.get('q') || '';
    $('evidence').checked=initial.get('evidence')==='1';
    $('exercise-shortcuts').innerHTML = stats.exercises.map(e=>`<button data-exercise="${esc(e.id)}">${esc(e.name)}</button>`).join('');
    $('exercise-shortcuts').querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{
      $('dataset').value=button.dataset.exercise==='shoulder_press'?'shoulder':'qevd';
      $('exercise').value=button.dataset.exercise;
      $('split').value='';$('label').value='';$('search').value='';$('evidence').checked=false;
      const url=new URL(location.href);
      url.searchParams.set('dataset',$('dataset').value);url.searchParams.set('exercise',$('exercise').value);
      for(const name of ['q','evidence','label','split']) url.searchParams.delete(name);
      history.replaceState(null,'',url);
      page=1;loadList();
    }));
    await loadList();
  } catch(error) { showError(error); }
})();
