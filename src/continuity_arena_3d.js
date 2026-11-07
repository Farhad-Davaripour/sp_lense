/* Local WebGL spectator loader. Buffer state while the module loads. */
(function () {
  window.createArena3D = function (options) {
    let renderer = null, stopped = false;
    const pending = { zone: 'operations', observation: {}, details: {} };
    import('/assets/arena_scene.js').then(({ createScene }) => {
      if (stopped) return;
      renderer = createScene(options);
      renderer.update(pending.observation, pending.details);
      renderer.setTarget(pending.zone);
    }).catch(error => {
      options.container.dataset.renderer = 'unavailable';
      const note = document.createElement('div');note.className = 'scene-error';
      note.textContent = 'The 3D view could not start. Recorded simulator data remains available below. ' + error.message;
      options.container.appendChild(note);
    });
    return {
      setTarget(zone) { pending.zone = zone; renderer?.setTarget(zone); },
      update(observation, details) { pending.observation = observation; pending.details = details; renderer?.update(observation, details); },
      burst(zone, action) { renderer?.burst(zone, action); },
      snapshot() { return renderer?.snapshot() || { activeZone: pending.zone, observation: pending.observation }; },
      destroy() { stopped = true; renderer?.destroy(); }
    };
  };
})();
