/* 모델 결과는 텍스트로 표시하며 HTML로 실행하지 않습니다. */
const $ = id => document.getElementById(id);
const labels = {
  stopped: '측정 종료',
  idle: '측정 대기',
  loading: '모델 준비 중',
  warming_up: '연속 관측 수집 중',
  predicting: '교정 결과',
  collecting: '반복 관측 중',
  pose_not_visible: '관절 관측 부족',
  not_requested: '선택하지 않은 평가',
  unavailable: '실행 불가',
  stale: '최신 관측 대기',
  camera_unavailable: '카메라 연결 대기',
  connecting: '카메라 연결 중',
  running: '측정 중',
  failed: '연결 처리 실패'
};

function text(parent, value, tag = 'p') {
  const e = document.createElement(tag);
  e.textContent = value;
  parent.append(e);
  return e;
}

function renderCorrection(data) {
  const box = $('correction');
  box.replaceChildren();
  text(box, labels[data.status] || data.status || '대기');
  if (data.error) text(box, data.error);
  if (data.message) text(box, data.message);
  if (data.buffer_frames !== undefined) text(box,
    `관측 ${data.buffer_frames}/${data.required_frames} 표본`);
  for (const line of data.result?.feedbacks || []) text(box, line);
  if (data.result && !(data.result.detected_errors || []).length) text(box, '선택 오류 미검출');
}

function renderScore(data) {
  const box = $('score');
  box.replaceChildren();
  text(box, labels[data.status] || data.status || '대기');
  if (data.error) text(box, data.error);
  if (data.movement) text(box, `${data.movement.phase} · 완료 ${data.movement.count}회`);
  if (data.live_score_100) {
    text(box, '진행 중 · 잠정 항목', 'h3');
    for (const item of data.live_score_100.items || []) text(box,
      `${item.title||item.id}: AI ${item.ai_points ?? '미평가'} · 규칙 ${item.rule_points ?? '미평가'}`);
  }
  for (const rep of data.repetitions || []) {
    text(box, `${rep.number}회`, 'h3');
    if (rep.error) {
      text(box, `평가 보류: ${rep.error}`);
      continue;
    }
    const table = document.createElement('table');
    const head = document.createElement('tr');
    for (const title of ['항목', 'AI', '규칙']) text(head, title, 'th');
    table.append(head);
    for (const item of rep.score_100?.items || []) {
      const row = document.createElement('tr');
      text(row, item.title || item.id, 'td');
      text(row, item.ai_points == null ? '미평가' : String(item.ai_points), 'td');
      text(row, item.rule_points == null ? '미평가' : String(item.rule_points), 'td');
      table.append(row);
    }
    box.append(table);
    for (const method of ['ai', 'rule']) {
      const total = rep.score_100?.[method];
      if (total) text(box,
        `${method==='ai'?'AI':'규칙'}: ${total.earned_subtotal ?? total.earned ?? total.points ?? '미평가'} / ${total.assessed_weight ?? total.maximum ?? '미확인'} (관측 항목)`
        );
    }
  }
}
async function post(path, payload) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  const state = await response.json();
  if (!response.ok) throw Error(state.error);
  return state;
}
$('options').addEventListener('submit', async event => {
  event.preventDefault();
  try {
    await post('/api/start', Object.fromEntries(new FormData(event.target)));
    $('notice').textContent = '측정을 시작합니다.';
  } catch (error) {
    $('notice').textContent = error.message;
  }
});
$('stop').onclick = async () => {
  try {
    await post('/api/stop', {});
  } catch (error) {
    $('notice').textContent = error.message;
  }
};
fetch('/api/config').then(r => r.json()).then(config => {
  if (config.fixed_mode) {
    const select = document.querySelector('[name=mode]');
    select.value = config.fixed_mode;
    select.disabled = true; // 비활성 필드는 FormData에서 제외되어 서버 기본값을 사용합니다.
  }
});
async function poll() {
  try {
    const state = await fetch('/api/state').then(r => r.json());
    $('notice').textContent = labels[state.status] || state.status;
    $('camera-status').textContent =
      `${state.camera||''} ${state.capture_fps?.toFixed(1)||0} fps ${state.camera_error||''}`;
    renderCorrection(state.correction || {});
    renderScore(state.score || {});
    $('detail').textContent = JSON.stringify(state, null, 2);
    $('preview').hidden = !state.running || state.status !== 'running';
    if (!$('preview').hidden) $('preview').src = `/api/frame.jpg?t=${Date.now()}`;
  } catch (error) {
    $('notice').textContent = '서버 연결을 확인하십시오.';
  } finally {
    setTimeout(poll, 400);
  }
}
poll();
