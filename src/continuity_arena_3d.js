/* Local WebGL spectator loader. Buffer state while the module loads. */
(function () {
  window.createArena3D = function (options) {
    let renderer = null, stopped = false;
    const pending = { zone: 'operations', observation: {}, details: {}, agents: null, arrival: undefined, feedback: null };
    import('/assets/arena_scene.js').then(({ createScene }) => {
      if (stopped) return;
      renderer = createScene(options);
      renderer.update(pending.observation, pending.details);
      if (pending.agents) renderer.updateAgents(...pending.agents);
      if (pending.feedback) renderer.feedback?.(pending.feedback);
      pending.feedback = null;
      renderer.setTarget(pending.zone, pending.arrival);
    }).catch(error => {
      options.container.dataset.renderer = 'unavailable';
      const note = document.createElement('div');note.className = 'scene-error';
      note.textContent = 'The 3D view could not start. Recorded simulator data remains available below. ' + error.message;
      options.container.appendChild(note);
    });
    return {
      setTarget(zone, onArrival) { pending.zone = zone; if (onArrival !== undefined) pending.arrival = onArrival; renderer?.setTarget(zone, onArrival); },
      update(observation, details) { pending.observation = observation; pending.details = details; renderer?.update(observation, details); },
      updateAgents(agents, selected, overlay) { pending.agents = [agents, selected, overlay]; renderer?.updateAgents(agents, selected, overlay); },
      feedback(notice) { if (renderer) renderer.feedback?.(notice); else pending.feedback = notice; },
      inspectSite(id, focus=false) { renderer?.inspectSite(id, focus); },
      burst(zone, action) { renderer?.burst(zone, action); },
      snapshot() { return renderer?.snapshot() || { activeZone: pending.zone, observation: pending.observation }; },
      destroy() { stopped = true; pending.arrival = undefined; renderer?.destroy(); }
    };
  };
})();
