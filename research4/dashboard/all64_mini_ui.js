(function () {
  'use strict';
  const el = id => document.getElementById('mini-all64-' + id);
  const set = (id, value) => { const node = el(id); if (node) node.textContent = value; };
  const gib = value => Number.isFinite(value) ? (value / 1073741824).toFixed(1) + ' GiB' : 'â€”';
  const num = value => Number.isFinite(value) ? value.toFixed(4) : 'â€”';

  window.renderAll64Mini = function (data) {
    const mini = data && data.mini;
    const age = mini && Number.isFinite(mini.observed_epoch) ? Date.now() / 1000 - mini.observed_epoch : Infinity;
    const online = mini && mini.schema === 'research4.all64-mini-dashboard.v1' && mini.online === true && age >= -5 && age <= 35;
    const archived = mini && mini.schema === 'research4.all64-mini-dashboard.v1' && mini.archive_only === true;
    const stage = archived ? mini.last_recorded : online ? (mini.active || mini.last_recorded) : null;
    const active = !!(online && mini.active && stage === mini.active && (stage.status === 'running' || stage.status === 'finalizing'));
    const status = el('status');
    if (status) { status.textContent = archived ? 'Archived Mini record' : !online ? 'Mini unavailable' : active ? (stage.status === 'finalizing' ? 'Finalizing' : 'Running') : stage ? 'Recorded Â· ' + stage.status : 'No current run'; status.classList.toggle('wait', !active); }
    set('title', stage ? 'Mac mini Â· rank ' + stage.rank + ' Â· ' + ({qualification:'qualification', trajectory:'training', dev:'development evaluation'}[stage.phase] || 'stage') : 'Mac mini Â· all 64 layers');
    set('owner', stage ? 'Owner ' + stage.owner.slice(0, 8) + ' Â· ' + (active ? 'live process verified' : 'saved record') : 'Waiting for verified Mini telemetry');
    let completed = null, total = null, label = 'No recorded progress';
    if (stage && stage.phase === 'qualification') { completed = stage.completed_cases; total = stage.total_cases; label = 'Saved qualification cases Â· not training'; }
    if (stage && stage.phase === 'trajectory') { completed = stage.completed_steps; total = stage.total_steps; label = 'Saved optimizer updates'; }
    if (stage && stage.phase === 'dev') label = 'Development evaluation';
    set('count', Number.isInteger(completed) && Number.isInteger(total) ? completed + ' / ' + total : 'â€”');
    set('progress-label', label);
    const fill = el('fill'), track = el('track');
    const percent = Number.isInteger(completed) && Number.isInteger(total) && total > 0 ? Math.max(0, Math.min(100, completed / total * 100)) : 0;
    if (fill) fill.style.width = percent + '%';
    if (track) { track.setAttribute('aria-valuenow', String(Number.isInteger(completed) ? completed : 0)); track.setAttribute('aria-valuemax', String(total || 1)); }
    set('epoch', stage && stage.phase === 'trajectory' && Number.isInteger(stage.completed_steps) ? (stage.completed_steps ? Math.min(3, Math.ceil(stage.completed_steps / 136)) : 0) + ' / 3' : stage && stage.phase === 'dev' ? stage.epoch + ' / 3' : 'â€”');
    set('speed', stage && Number.isFinite(stage.tokens_per_second) ? stage.tokens_per_second.toFixed(2) + ' tok/s' : 'â€”');
    set('loss', stage ? num(stage.latest_loss) : 'â€”');
    const memory = online && mini.resources && mini.resources.memory;
    set('memory', memory && Number.isFinite(memory.total_bytes) && Number.isFinite(memory.available_bytes) ? gib(memory.total_bytes - memory.available_bytes) + ' / ' + gib(memory.total_bytes) : 'â€”');
    const guard = stage && stage.guard;
    set('guard', guard && guard.fresh && guard.target_verified ? ((guard.valid === true && guard.stop_required === false ? 'Normal Â· ' : 'Stop / invalid Â· ') + gib(guard.headroom_estimate_bytes)) : 'Not verified');
    const stopped = stage && stage.status === 'stopped' ? ' Last recorded stop: ' + (stage.stop_reason || 'See the saved release record.') : '';
    const priorGuard = stage && stage.status === 'stopped' && guard && guard.pressure ? ' Last guard pressure: ' + guard.pressure + '.' : '';
    set('note', archived ? 'Saved training and development results. Live Mini resource metrics are disabled.' : !online ? 'Mini telemetry is unavailable or stale; no progress is inferred.' : stage ? (stage.note || '') + stopped + priorGuard + ' Guard headroom is an estimate, not guaranteed allocatable memory. Saved qualification cases are not HPO training updates.' : 'No matching Mini run is currently recorded.');
    const body = el('dev-body');
    if (body) {
      while (body.firstChild) body.removeChild(body.firstChild);
      const rows = (online || archived) && Array.isArray(mini.dev) ? mini.dev : [];
      for (const row of rows) {
        const tr = document.createElement('tr');
        for (const text of ['R' + row.rank, String(row.epoch), row.status === 'recorded' ? num(row.primary) : row.status === 'ambiguous' ? 'Ambiguous' : 'Pending', row.status === 'recorded' ? num(row.pooled_action) : 'â€”', row.status === 'recorded' ? num(row.full_suffix) : 'â€”']) {
          const td = document.createElement('td'); td.textContent = text; tr.appendChild(td);
        }
        body.appendChild(tr);
      }
    }
  };
})();
