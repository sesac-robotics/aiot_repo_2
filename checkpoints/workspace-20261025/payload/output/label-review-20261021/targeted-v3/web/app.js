/* 역할: 저장 영상의 단일 반복 구간을 모델 분석 API에 전달하고 항목 예측·관절 영상을 표시한다.
   입력: 운동 코드·허용 목록 ID·시작/종료 초. 출력: 항목별 배점 점수와 보류 사유.
   실패: 잘못된 구간·분석 중복·관절 부족·서버 오류는 상태 영역에 표시한다. */
const $ = id => document.getElementById(id);
let sources = [];
const itemNames = {trunk_line:'몸통·골반 정렬', descent_range:'하강 범위', return_extension:'상단 복귀', support:'손·발 지지'};
function updateSources() {
  const selected = sources.filter(s => s.exercise === $('exercise').value);
  $('source').replaceChildren(...selected.map(s => new Option(s.title, s.id)));
  if (!selected.length) $('status').textContent = '이 운동의 저장 영상이 없습니다.';
  updateSegment();
}
function updateSegment() {
  const s = sources.find(s => s.id === $('source').value);
  if (!s) return;
  $('start').value = s.start.toFixed(2);
  $('end').value = s.end.toFixed(2);
  $('ack').checked = false;
}
function showResult(id, result) {
  const source = sources.find(s => s.source_job === result.source_job);
  if (source) { $('exercise').value = source.exercise; updateSources(); $('source').value = source.id; }
  $('start').value = result.segment.start.toFixed(2); $('end').value = result.segment.end.toFixed(2);
  $('diagnostics').checked = result.items.some(i => i.diagnostic_grade != null);
  $('items').replaceChildren();
  for (const item of result.items) {
    const tr = document.createElement('tr');
    const values = [item.name_ko, item.predicted_points === null ? '산출 보류' : `${item.predicted_points.toFixed(1)} / ${item.weight}점 항목`, item.diagnostic_grade == null ? '—' : `${item.diagnostic_grade.toFixed(2)} / 2 (채점 보류)`, item.status === 'experimental_prediction' ? '학습 모델의 실험 예측' : item.reasons.join(' · ')];
    for (const value of values) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
    $('items').append(tr);
  }
  $('video').src = `/model/jobs/${id}/overlay.mp4`;
  $('download').href = `/model/jobs/${id}/result.json`;
  $('result').hidden = false;
}
$('exercise').addEventListener('change', updateSources);
$('source').addEventListener('change', updateSegment);
$('analyze').addEventListener('click', async () => {
  if (!$('ack').checked) { $('status').textContent = '한 번의 동작 구간 확인란을 선택하세요.'; return; }
  $('analyze').disabled = true; $('result').hidden = true;
  try {
    $('status').textContent = 'RTX Heavy로 관절을 추출하고 모델 항목 예측·표시 영상을 만들고 있습니다.';
    const response = await fetch('/model/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({source_id:$('source').value, start:Number($('start').value), end:Number($('end').value), acknowledged:true, diagnostics:$('diagnostics').checked})});
    const request = await response.json();
    if (!response.ok) throw new Error(request.error || '분석 요청 실패');
    while (true) {
      await new Promise(resolve => setTimeout(resolve, 1200));
      const response = await fetch(`/model/status?id=${request.id}`);
      if (!response.ok) throw new Error('진행 상태 확인 실패');
      const state = await response.json();
      if (state.state === 'failed' || state.state === 'stale') throw new Error(state.error);
      if (state.state === 'done') {
        showResult(request.id, state.result);
        $('status').textContent = '항목 분석 완료 · 종합 점수는 산출하지 않았습니다.';
        history.replaceState({}, '', `/model/?job=${request.id}`); break;
      }
    }
  } catch (error) { $('status').textContent = error.message; }
  finally { $('analyze').disabled = false; }
});
(async () => {
  try {
    sources = await (await fetch('/model/catalog')).json(); updateSources();
    const report = await (await fetch('/model/metrics')).json();
    for (const [exercise, value] of Object.entries(report.results)) {
      const title = document.createElement('h3'); title.textContent = exercise === 'pushups' ? '푸시업' : '스쿼트'; $('metrics').append(title);
      for (const [id, item] of Object.entries(value.items)) {
        const p = document.createElement('p');
        p.textContent = `${itemNames[id] || id}: ` + (item.web_experiment_enabled ? `실험 가능 · 검증 ${item.validation_samples}개 · 평균 등급 오차 ${item.validation_MAE_grade.toFixed(2)} / 2` : '보완 중 · ' + (item.hold_reasons || ['학습 등급 자료 부족']).join(' · '));
        $('metrics').append(p);
      }
    }
    const id = new URLSearchParams(location.search).get('job');
    if (id && /^[a-f0-9]{32}$/.test(id)) {
      const response = await fetch(`/model/status?id=${id}`); const state = await response.json();
      if (state.state === 'stale') $('status').textContent = state.error;
      if (response.ok && state.state === 'done') {showResult(id, state.result); $('status').textContent = '보관한 모델 분석 결과를 불러왔습니다.';}
    }
  } catch(error) {$('status').textContent = '화면 로드 실패: ' + error.message;}
})();
