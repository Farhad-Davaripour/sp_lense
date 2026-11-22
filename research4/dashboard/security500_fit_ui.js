/* Exact security500 metadata only. No model actions, timers, requests or discovery. */
(() => {
  const plan = '748288eb723074fa752322fded67c5298c31cc09d3f6e6c0a5e65cda765bc0e3';
  const get = id => document.getElementById(id);
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const gib = value => finite(value) && value >= 0 ? (value / 1024 ** 3).toFixed(1) + ' GiB' : '—';
  const stamp = value => {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'Not available' :
      new Date(date.getTime() - 6 * 3600 * 1000).toISOString().slice(0, 19).replace('T', ' ') + ' ABT';
  };
  function stage(title, status, detail, index) {
    const item = document.createElement('li'); item.className = 'fit-stage ' + status;
    const top = document.createElement('div'); top.className = 'fit-stage-top';
    const mark = document.createElement('span'); mark.className = 'fit-stage-index';
    mark.textContent = status === 'completed' ? '✓' : String(index + 1);
    const heading = document.createElement('div'); const name = document.createElement('h3'); name.textContent = title;
    const label = document.createElement('span'); label.className = 'fit-stage-state';
    label.textContent = ({completed:'Complete', active:'In progress', pending:'Pending', stopped:'Stopped', blocked:'Needs review', unavailable:'Not verified'})[status];
    heading.append(name, label); top.append(mark, heading);
    const text = document.createElement('p'); text.textContent = detail;
    item.append(top, text); return item;
  }
  window.renderSecurity500Fit = data => {
    if (!get('security500-current')) return;
    const value = data?.mac?.security500_fit;
    const prepared = value?.schema === 'research4.agent-security500-dashboard-fit.v1' &&
      value.plan_sha256 === plan && value.preparation_verified === true;
    const available = prepared && value.status === 'available' && Number.isInteger(value.optimizer_updates) &&
      value.optimizer_updates >= 0 && value.optimizer_updates <= 500;
    const count = available ? value.optimizer_updates : null;
    const complete = available && value.training_status === 'complete' && count === 500 && value.dev_completed === 100 && value.completion_saved_metadata_verified === true;
    const stopped = available && value.training_status === 'stopped';
    const paused = available && value.training_status === 'paused';
    const live = available && value.live_owner_verified === true && value.stale !== true && data?.mac?.online !== false;
    const labels = !prepared ? 'Awaiting verified metadata' : !available ? 'Data approved · awaiting run identity' :
      complete ? 'Training and DEV complete' : stopped ? 'Training stopped' : paused ? 'Training paused' :
      value.training_status === 'terminal_cleanup_unverified' ? 'Release not verified' : value.stale ? 'Saved progress · may be out of date' :
      live ? count ? 'Training · verified updates' : 'Loading · no updates yet' : 'Saved progress · owner unconfirmed';
    get('security500-status').textContent = labels;
    get('security500-count').textContent = count === null ? '— / 500 verified updates' : count.toLocaleString() + ' / 500 verified updates';
    get('security500-state').textContent = count === null ? 'Preparation does not count as optimizer progress.' :
      complete ? '500 training records and 100 development records completed.' : 'Only contiguous finite saved Adam updates count.';
    get('security500-fill').style.width = count === null ? '0%' : (100 * count / 500) + '%';
    if (count === null) get('security500-track').removeAttribute('aria-valuenow');
    else get('security500-track').setAttribute('aria-valuenow', String(count));
    get('security500-loss').textContent = available && finite(value.last_training_loss) ? value.last_training_loss.toFixed(4) : '—';
    get('security500-speed').textContent = available && finite(value.last_training_target_tokens_per_compute_second) ? value.last_training_target_tokens_per_compute_second.toFixed(2) : '—';
    get('security500-peak').textContent = available ? gib(value.recorded_mlx_peak_bytes) : '—';
    const reserve = available ? value.native_reserve : null;
    get('security500-reserve').textContent = reserve ? gib(reserve.bytes) : '—';
    get('security500-reserve-note').textContent = reserve ? 'Last native estimate · ' + Math.round(reserve.age_seconds) + 's old' : 'No recent verified native sample';
    const checkpoints = available ? value.checkpoints : null;
    get('security500-checkpoints').textContent = checkpoints && Number.isInteger(checkpoints.count) ? checkpoints.count + ' / 6' : '— / 6';
    get('security500-dev').textContent = available && value.dev_completed === 100 && finite(value.dev_pooled_target_NLL) ? value.dev_pooled_target_NLL.toFixed(5) : 'Pending';
    const stages = [
      ['Approve data and fit plan', prepared ? 'completed' : 'unavailable', '500 training records (484 unique inputs) · 100 DEV records (95 unique inputs) · shared Last8/rank16.'],
      ['Fine-tune shared adapter', count === 500 ? 'completed' : stopped ? 'stopped' : paused ? 'blocked' : live ? 'active' : 'pending', count === null ? 'Awaiting the exact run and saved update journal.' : count.toLocaleString() + ' / 500 verified optimizer updates.'],
      ['Validate on 100 DEV records', available && value.dev_completed === 100 ? 'completed' : available && value.phase === 'DEV_evaluating' ? 'active' : 'pending', available && value.dev_completed === 100 ? 'DEV token loss reported separately; it is not Game accuracy.' : 'No validation result is claimed before a complete saved receipt.'],
      ['Verify saved completion and release', complete ? 'completed' : 'pending', complete ? 'Saved completion and owned release verified.' : 'Final checkpoint and owned release checks follow training and DEV.'],
      ['Audit checkpoints and validation', complete && value.final_saved_audit_verified === true ? 'completed' : 'pending', complete && value.final_saved_audit_verified === true ? 'All six checkpoints, optimizer state, frozen base and 100 DEV records verified. Game results are separate.' : 'Waiting for the hash-verified final saved-artifact audit.']
    ];
    const list = get('security500-stages'); list.replaceChildren();
    stages.forEach((item, index) => list.append(stage(...item, index)));
    get('security500-note').textContent = 'Training speed counts supervised target tokens divided by gradient and Adam compute time. Native reserve is an estimate. Published checkpoint counts use metadata. These readings do not establish inference speed, security improvement or exploitation outcomes.';
    get('security500-stamp').textContent = available ? 'Last saved progress: ' + stamp(value.recorded_utc) + ' · Process liveness is checked separately.' : 'Approved plan is bound to this pilot; no progress is inferred from another run.';
  };
})();
