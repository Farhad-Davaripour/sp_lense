/* Saved preparation and optimizer progress; this component never submits work. */
(() => {
  const labels = {completed: 'Complete', active: 'In progress', pending: 'Not started', blocked: 'Needs review', stopped: 'Stopped', unavailable: 'Not verified'};
  const titles = {source_checks: 'Scenario checks', saved_runtime: 'Training history check', dataset_build: 'Prepare dataset', full_audit: 'Audit complete dataset', input_resource_plan: 'Review input size and memory', train: 'Fine-tune adapter', validate: 'Validate model', checkpoint_audit: 'Verify checkpoints', game_comparison: 'Compare in the game'};
  const get = id => document.getElementById(id);
  const renderPreparation = provider => {
    const card = get('next-preparation');
    if (!card) return;
    try {
      let savedCompletion = false;
      if (provider?.schema !== 'research4.provider352-dashboard-fit.v1' || provider.status !== 'available'
          || provider.plan_sha256 !== 'ca582107fa244d5cdbecb3aa2ed910c9558385198564b9d189fdcba56673e52b') {
        const saved = JSON.parse(get('completed-fit-receipt').textContent);
        if (saved.schema === 'research4.provider352-dashboard-fit.v1' && saved.status === 'available'
            && saved.plan_sha256 === 'ca582107fa244d5cdbecb3aa2ed910c9558385198564b9d189fdcba56673e52b'
            && saved.training_status === 'complete' && saved.optimizer_updates === 2160
            && saved.dev_completed === 500 && saved.final_saved_audit_verified === true) {
          provider = saved; savedCompletion = true;
        }
      }
      const value = JSON.parse(get('preparation-receipt').textContent);
      if (value.schema !== 'research4.saved-preparation-steps.v1' || value.model_execution_started !== false || value.optimizer_updates !== 0) throw Error('Invalid preparation receipt');
      get('preparation-title').textContent = value.title;
      get('preparation-status').textContent = value.headline;
      const list = get('preparation-stage-list'); list.replaceChildren();
      value.stages.forEach((stage, index) => {
        const status = Object.hasOwn(labels, stage.status) ? stage.status : 'unavailable';
        const item = document.createElement('li'); item.className = 'fit-stage ' + status;
        const top = document.createElement('div'); top.className = 'fit-stage-top';
        const number = document.createElement('span'); number.className = 'fit-stage-index'; number.textContent = status === 'completed' ? '✓' : String(index + 1); number.setAttribute('aria-hidden', 'true');
        const heading = document.createElement('div');
        const title = document.createElement('h3'); title.textContent = stage.title;
        const state = document.createElement('span'); state.className = 'fit-stage-state'; state.textContent = labels[status];
        heading.append(title, state); top.append(number, heading);
        const detail = document.createElement('p'); detail.textContent = stage.detail;
        item.append(top, detail); list.append(item);
      });
      get('preparation-updates').textContent = '0 / 2,160 planned updates';
      get('preparation-note').textContent = value.note;
      const when = new Date(value.recorded_utc);
      const stamp = new Date(when.getTime() - 6 * 3600 * 1000).toISOString().slice(0, 16).replace('T', ' ') + ' ABT';
      get('preparation-stamp').textContent = 'Saved preparation: ' + stamp + ' · Refresh the page for the latest published steps.';
      if (get('preparation-training-state')) get('preparation-training-state').textContent = 'Saved preparation only; live training progress is not verified in this reading.';
      get('preparation-status').textContent = 'Preparation snapshot · live status unavailable';
      get('preparation-stage-summary').textContent = 'Preparation steps · training progress has not been verified in this reading.';
      if (get('provider-fit-metrics')) get('provider-fit-metrics').hidden = true;
      if (get('provider-fit-track')) get('provider-fit-track').hidden = true;
      if (provider?.schema === 'research4.provider352-dashboard-fit.v1' && provider.status === 'available'
          && provider.plan_sha256 === 'ca582107fa244d5cdbecb3aa2ed910c9558385198564b9d189fdcba56673e52b'
          && Number.isInteger(provider.optimizer_updates) && provider.optimizer_updates >= 0 && provider.optimizer_updates <= 2160) {
        const count = provider.optimizer_updates;
        const stopped = provider.training_status === 'stopped';
        const paused = provider.training_status === 'paused';
        const complete = provider.training_status === 'complete' && count === 2160 && provider.dev_completed === 500;
        const audited = complete && provider.final_saved_audit_verified === true;
        const live = provider.live_owner_verified === true && provider.stale !== true;
        get('preparation-title').textContent = 'SP Lens · service-preservation 3.5.2';
        get('preparation-status').textContent = stopped ? 'Fine-tuning stopped' : paused ? 'Fine-tuning paused' : audited ? 'Training, DEV and saved audit complete' : complete ? 'Training and DEV complete' : provider.stale ? 'Saved progress · may be out of date' : live ? (count ? 'Fine-tuning · verified updates' : 'Preparing fit · no verified updates yet') : 'Saved progress · worker unconfirmed';
        get('preparation-updates').textContent = count.toLocaleString() + ' / 2,160 verified updates';
        if (get('preparation-training-state')) get('preparation-training-state').textContent = count ? 'Verified saved optimizer updates for this exact run.' : 'The bound worker has not yet verified an optimizer update.';
        if (get('provider-fit-track')) {
          get('provider-fit-track').hidden = false;
          get('provider-fit-track').setAttribute('aria-valuenow', String(count));
          get('provider-fit-fill').style.width = (count / 2160 * 100) + '%';
        }
        const stages = value.stages.filter(stage => stage.id !== 'game_comparison').map(stage => ({...stage}));
        stages.forEach(stage => {
          if (provider.preparation_verified === true && ['dataset_build', 'full_audit', 'resource_review'].includes(stage.id)) {
            stage.status = 'completed';
            stage.detail = {dataset_build: '2,160 training examples and 500 separate DEV examples saved.', full_audit: 'All 2,660 examples checked for legal actions, complete targets, split separation and token limits.', resource_review: 'Reviewed Mac recipe: rank 16, last eight layers, 512-token training limit and unchanged memory stops.'}[stage.id];
          }
          if (stage.id === 'train') {
            stage.status = count === 2160 ? 'completed' : stopped ? 'stopped' : paused ? 'blocked' : live ? 'active' : 'unavailable';
            stage.detail = count.toLocaleString() + ' / 2,160 verified updates. ' + (count === 0 ? 'Preparation and process activity are not optimizer updates.' : 'Count is checked against the contiguous actual-update journal.');
          }
          if (stage.id === 'validate') {
            stage.status = provider.dev_completed === 500 ? 'completed' : provider.phase === 'DEV_evaluating' && count === 2160 ? 'active' : 'pending';
            stage.detail = provider.dev_completed === 500 ? '500 / 500 DEV examples complete. Token loss: ' + (Number.isFinite(provider.dev_pooled_target_NLL) ? provider.dev_pooled_target_NLL.toFixed(5) : 'not verified') + '. This is not game accuracy.' : 'No verified DEV completion count. Validation is separate from training updates.';
          }
          if (stage.id === 'checkpoint_audit' && audited) {
            stage.title = 'Audit saved training artifacts'; stage.status = 'completed';
            stage.detail = 'All 15 checkpoints, final Adam state, unchanged DEV weights, base/source/guard records and owned release passed. Native inference qualification remains pending.';
          }
        });
        const completedSteps = stages.filter(stage => stage.status === 'completed').length;
        get('preparation-stage-summary').textContent = completedSteps + ' of ' + stages.length + ' fine-tuning steps complete';
        get('preparation-next-note').textContent = audited
          ? 'Next: verify the Mac inference endpoint and adapter switching, then run the separate game comparison. Neither is complete yet.'
          : 'Next: complete training, validation and checkpoint auditing before the inference and game checks.';
        list.replaceChildren();
        stages.forEach((stage, index) => {
          const status = Object.hasOwn(labels, stage.status) ? stage.status : 'unavailable';
          const item = document.createElement('li'); item.className = 'fit-stage ' + status;
          const top = document.createElement('div'); top.className = 'fit-stage-top';
          const number = document.createElement('span'); number.className = 'fit-stage-index'; number.textContent = status === 'completed' ? '✓' : String(index + 1); number.setAttribute('aria-hidden', 'true');
          const heading = document.createElement('div'); const title = document.createElement('h3'); title.textContent = stage.title;
          const state = document.createElement('span'); state.className = 'fit-stage-state'; state.textContent = labels[status];
          heading.append(title, state); top.append(number, heading); const detail = document.createElement('p'); detail.textContent = stage.detail;
          item.append(top, detail); list.append(item);
        });
        get('preparation-note').textContent = provider.note + ' This card does not infer progress from process activity or establish checkpoint/game results.';
        if (audited) get('preparation-note').textContent = 'Training, 500-example DEV and the saved-artifact audit passed. Supervisor OS exit status was unobserved; successful application completion and physical release were verified separately. Game results and inference qualification remain pending.';
        const runTime = new Date(provider.recorded_utc);
        const runStamp = Number.isNaN(runTime.getTime()) ? 'unavailable' : new Date(runTime.getTime() - 6 * 3600 * 1000).toISOString().slice(0, 19).replace('T', ' ') + ' ABT';
        if (complete && runStamp !== 'unavailable') get('preparation-title').textContent = 'SP Lens - ' + runStamp.slice(0,16) + ' ABT';
        get('preparation-stamp').textContent = (savedCompletion ? 'Saved audited completion: ' : 'Saved run progress: ') + runStamp + ' · These are saved training results, not current inference activity.';
        if (get('provider-fit-metrics')) {
          get('provider-fit-metrics').hidden = false;
          const speed = provider.last_training_target_tokens_per_compute_second;
          const loss = provider.last_training_loss;
          const peak = provider.recorded_mlx_peak_bytes;
          get('provider-fit-speed').textContent = Number.isFinite(speed) ? speed.toFixed(2) : '—';
          get('provider-fit-loss').textContent = Number.isFinite(loss) ? loss.toFixed(3) : '—';
          get('provider-fit-peak').textContent = Number.isFinite(peak) ? (peak / 1024 ** 3).toFixed(1) + ' GiB' : '—';
          if (get('provider-fit-dev')) get('provider-fit-dev').textContent = provider.dev_completed === 500 && Number.isFinite(provider.dev_pooled_target_NLL) ? provider.dev_pooled_target_NLL.toFixed(5) : 'Pending';
        }
      }
    } catch (_) {
      get('preparation-status').textContent = 'Preparation status unavailable';
      get('preparation-updates').textContent = '—';
      get('preparation-stage-list').replaceChildren();
      get('preparation-note').textContent = 'The saved preparation receipt could not be read.';
    }
  };
  window.renderFineTuningSteps = data => {
    renderPreparation(data?.mac?.provider352_fit);
    const value = data?.fine_tuning_steps || data?.mac?.fine_tuning_steps;
    const available = value?.status === 'available';
    const stages = available && Array.isArray(value.stages) ? value.stages : [];
    get('fit-steps-title').textContent = 'SP Lens · ongoing service preservation';
    const active = stages.find(stage => stage.status === 'active');
    const blocked = stages.find(stage => stage.status === 'blocked');
    const stopped = value?.terminal_training_status === 'stopped';
    get('fit-steps-status').textContent = !available ? 'Progress unavailable' : stopped ? 'Fine-tuning stopped' : active ? (titles[active.id] || active.title) : blocked ? 'Preparation needs review' : 'Awaiting next stage';
    const count = available && Number.isInteger(value.optimizer_updates) && value.optimizer_updates >= 0 && value.optimizer_updates <= 2160 ? value.optimizer_updates : null;
    get('fit-steps-updates').textContent = (count === null ? '—' : count.toLocaleString()) + ' / 2,160 updates';
    const train = stages.find(stage => stage.id === 'train');
    get('fit-steps-training-state').textContent = !available ? 'Current attempt has no verified progress metadata.' : stopped ? (count === 0 ? 'Stopped before the first optimizer update.' : 'Stopped after ' + count.toLocaleString() + ' verified updates.') : count === 0 ? 'Fine-tuning has not started.' : train?.status === 'completed' ? 'Fine-tuning updates completed.' : 'Verified optimizer updates';
    get('fit-steps-fill').style.width = count === null ? '0%' : Math.min(100, count / 2160 * 100) + '%';
    if (count === null) get('fit-steps-track').removeAttribute('aria-valuenow');
    else get('fit-steps-track').setAttribute('aria-valuenow', String(count));
    const list = get('fit-stage-list');
    list.replaceChildren();
    stages.forEach((stage, index) => {
      const status = Object.hasOwn(labels, stage.status) ? stage.status : 'unavailable';
      const item = document.createElement('li'); item.className = 'fit-stage ' + status;
      const top = document.createElement('div'); top.className = 'fit-stage-top';
      const number = document.createElement('span'); number.className = 'fit-stage-index'; number.textContent = status === 'completed' ? '✓' : String(index + 1); number.setAttribute('aria-hidden', 'true');
      const heading = document.createElement('div');
      const title = document.createElement('h3'); title.textContent = titles[stage.id] || stage.title;
      const state = document.createElement('span'); state.className = 'fit-stage-state'; state.textContent = stage.id === 'train' && status === 'blocked' ? 'Not started · awaiting review' : labels[status];
      heading.append(title, state); top.append(number, heading);
      const detail = document.createElement('p'); detail.textContent = stage.detail || '';
      item.append(top, detail); list.append(item);
    });
    const warnings = available && Array.isArray(value.warnings) ? value.warnings : [];
    const notice = get('fit-steps-notice'); notice.hidden = warnings.length === 0; notice.textContent = warnings.join(' ');
    const when = value?.recorded_utc ? new Date(value.recorded_utc) : null;
    const offset = data?.model_version_catalog?.utc_offset_seconds;
    const zone = data?.model_version_catalog?.display_zone_abbreviation;
    const validTime = when && !Number.isNaN(when.getTime());
    const timestamp = !validTime ? 'unavailable' : Number.isInteger(offset) && typeof zone === 'string' ? new Date(when.getTime() + offset * 1000).toISOString().slice(0, 16).replace('T', ' ') + ' ' + zone : when.toISOString();
    get('fit-steps-stamp').textContent = 'Last saved update: ' + timestamp + (value?.stale ? ' · May be out of date.' : '') + ' · Preparation checks are separate from optimizer progress.';
  };
})();
