/* Lightweight perspective renderer for the loopback-only simulator viewer.
 * It uses the browser's 2D canvas as a dependency-free 3D projection surface;
 * no model inputs, simulator state, scoring, or evidence are changed here.
 */
(function () {
  "use strict";

  const TAU = Math.PI * 2;
  const ZONES = {
    operations: { x: 0, z: 0, h: 2.6, w: 2.5, d: 2.2, color: "#4b8ef7" },
    region_north: { x: 0, z: -6.4, h: 3.2, w: 2.3, d: 1.9, color: "#66b7ff" },
    region_west: { x: -6.6, z: -2.8, h: 2.9, w: 2.4, d: 2.0, color: "#53c5d8" },
    region_east: { x: 6.6, z: -2.8, h: 2.9, w: 2.4, d: 2.0, color: "#53c5d8" },
    region_coastal: { x: 6.1, z: 4.9, h: 2.8, w: 2.4, d: 2.0, color: "#45a6d8" },
    market: { x: -6.2, z: 4.8, h: 2.1, w: 2.8, d: 2.2, color: "#f0b44b" },
    security_lab: { x: -2.2, z: 5.8, h: 2.5, w: 2.5, d: 2.1, color: "#8f71ff" },
    contract_exchange: { x: 2.5, z: 5.8, h: 2.7, w: 2.5, d: 2.1, color: "#ef6fa5" },
  };

  const TREE_POSITIONS = [
    [-8.5, -6.5], [-7.7, -5.3], [-8.7, -1], [-8.1, 2.4], [-8.8, 6.5],
    [8.4, -6.4], [7.8, -5.2], [8.8, -0.8], [8.2, 2.6], [8.8, 6.4],
    [-4.3, -6.8], [4.5, -6.9], [-4.4, 7.1], [4.6, 7.2], [-0.8, 7.6],
  ];
  const ROCK_POSITIONS = [[-4.4, -1.1], [4.4, -0.8], [-4.4, 2.4], [4.1, 2.5], [0.2, -3.7]];
  const PICKUPS = [
    { x: -4.1, z: 2.4, icon: "⚡", color: "#ffe36e" },
    { x: 4.2, z: 2.0, icon: "▣", color: "#79c8ff" },
    { x: -4.5, z: 5.8, icon: "₿", color: "#ffbf52" },
    { x: 4.6, z: 5.9, icon: "◆", color: "#7be1ba" },
  ];
  const THREATS = [
    { x: -7.8, z: 1.1 }, { x: 7.8, z: 1.0 }, { x: -0.9, z: 4.0 },
    { x: 2.7, z: -2.9 }, { x: -2.9, z: -3.0 }, { x: 0.2, z: -5.0 }, { x: 5.0, z: 1.0 },
  ];

  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const mix = (a, b, amount) => a + (b - a) * amount;
  const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
  const cross = (a, b) => ({
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  });
  const normal = (v) => {
    const length = Math.hypot(v.x, v.y, v.z) || 1;
    return { x: v.x / length, y: v.y / length, z: v.z / length };
  };
  const shade = (hex, factor) => {
    const value = parseInt(hex.slice(1), 16);
    const r = clamp(Math.round(((value >> 16) & 255) * factor), 0, 255);
    const g = clamp(Math.round(((value >> 8) & 255) * factor), 0, 255);
    const b = clamp(Math.round((value & 255) * factor), 0, 255);
    return `rgb(${r},${g},${b})`;
  };

  function createArena3D(options) {
    const canvas = options.canvas;
    const container = options.container;
    const marker = options.marker;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return null;

    container.classList.add("arena-3d");
    let width = 1;
    let height = 1;
    let dpr = 1;
    let yaw = -0.48;
    let targetYaw = yaw;
    let cameraRadius = 21;
    let targetRadius = cameraRadius;
    let dragging = false;
    let pointerX = 0;
    let lastTime = performance.now();
    let activeZone = "operations";
    let observation = {};
    let danger = 0;
    let warning = null;
    let lastAction = "";
    let agent = { x: ZONES.operations.x, z: ZONES.operations.z };
    let target = { x: agent.x, z: agent.z };
    let burst = [];
    let destroyed = false;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const camera = () => {
      const position = {
        x: Math.sin(yaw) * cameraRadius,
        y: 12.7,
        z: Math.cos(yaw) * cameraRadius,
      };
      const focus = { x: 0, y: 0.5, z: 0.8 };
      const forward = normal({ x: focus.x - position.x, y: focus.y - position.y, z: focus.z - position.z });
      const right = normal(cross(forward, { x: 0, y: 1, z: 0 }));
      const up = cross(right, forward);
      return { position, forward, right, up, focal: height * 1.08 };
    };

    const projectWithCamera = (point, view) => {
      const relative = {
        x: point.x - view.position.x,
        y: point.y - view.position.y,
        z: point.z - view.position.z,
      };
      const depth = dot(relative, view.forward);
      if (depth <= 0.1) return { x: -1000, y: -1000, depth, scale: 0 };
      return {
        x: width * 0.5 + dot(relative, view.right) * view.focal / depth,
        y: height * 0.47 - dot(relative, view.up) * view.focal / depth,
        depth,
        scale: view.focal / depth,
      };
    };

    const polygon = (points, fill, stroke, lineWidth = 1) => {
      if (!points.length) return;
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
      ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke(); }
    };

    const line3d = (a, b, view, stroke, lineWidth = 1, alpha = 1) => {
      const pa = projectWithCamera(a, view);
      const pb = projectWithCamera(b, view);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.restore();
    };

    const faceCommands = [];
    const addBox = (x, y, z, w, h, d, color, glow = false) => {
      const points = [
        { x: x - w / 2, y, z: z - d / 2 }, { x: x + w / 2, y, z: z - d / 2 },
        { x: x + w / 2, y, z: z + d / 2 }, { x: x - w / 2, y, z: z + d / 2 },
        { x: x - w / 2, y: y + h, z: z - d / 2 }, { x: x + w / 2, y: y + h, z: z - d / 2 },
        { x: x + w / 2, y: y + h, z: z + d / 2 }, { x: x - w / 2, y: y + h, z: z + d / 2 },
      ];
      const faces = [
        [[4, 5, 6, 7], shade(color, 1.2)], [[0, 1, 5, 4], shade(color, 0.72)],
        [[1, 2, 6, 5], shade(color, 0.88)], [[2, 3, 7, 6], shade(color, 0.62)],
        [[3, 0, 4, 7], shade(color, 0.78)],
      ];
      const view = camera();
      faces.forEach(([indices, fill]) => {
        const projected = indices.map((index) => projectWithCamera(points[index], view));
        const depth = projected.reduce((sum, point) => sum + point.depth, 0) / projected.length;
        faceCommands.push({ projected, depth, fill, stroke: glow ? `${color}cc` : "#07111fcc", glow });
      });
    };

    const road = (from, to, view) => {
      const dx = to.x - from.x;
      const dz = to.z - from.z;
      const length = Math.hypot(dx, dz) || 1;
      const px = -dz / length * 0.24;
      const pz = dx / length * 0.24;
      const corners = [
        { x: from.x + px, y: 0.025, z: from.z + pz }, { x: to.x + px, y: 0.025, z: to.z + pz },
        { x: to.x - px, y: 0.025, z: to.z - pz }, { x: from.x - px, y: 0.025, z: from.z - pz },
      ].map((point) => projectWithCamera(point, view));
      polygon(corners, "#233344", "#5d7694aa", 1);
      line3d({ x: from.x, y: 0.035, z: from.z }, { x: to.x, y: 0.035, z: to.z }, view, "#78b7e655", 1.2);
    };

    const drawTree = (x, z, view, time) => {
      const base = projectWithCamera({ x, y: 0, z }, view);
      const top = projectWithCamera({ x, y: 1.25 + Math.sin(time * 0.001 + x) * 0.04, z }, view);
      const scale = clamp(top.scale * 0.24, 0.35, 1.4);
      ctx.strokeStyle = "#503b27";
      ctx.lineWidth = 5 * scale;
      ctx.beginPath(); ctx.moveTo(base.x, base.y); ctx.lineTo(top.x, top.y + 12 * scale); ctx.stroke();
      const gradient = ctx.createRadialGradient(top.x - 6 * scale, top.y - 10 * scale, 2, top.x, top.y, 29 * scale);
      gradient.addColorStop(0, "#86dc86"); gradient.addColorStop(0.45, "#2e945e"); gradient.addColorStop(1, "#0d4c38");
      ctx.fillStyle = gradient;
      ctx.beginPath(); ctx.arc(top.x, top.y, 25 * scale, 0, TAU); ctx.fill();
    };

    const drawRock = (x, z, view) => {
      const p = projectWithCamera({ x, y: 0.22, z }, view);
      const radius = clamp(p.scale * 0.13, 3, 14);
      const gradient = ctx.createRadialGradient(p.x - radius * 0.35, p.y - radius * 0.4, 1, p.x, p.y, radius);
      gradient.addColorStop(0, "#9ab0ae"); gradient.addColorStop(1, "#344746");
      ctx.fillStyle = gradient; ctx.beginPath(); ctx.ellipse(p.x, p.y, radius, radius * 0.7, -0.2, 0, TAU); ctx.fill();
    };

    const drawBuildingDetails = (zone, view, time) => {
      const p = projectWithCamera({ x: zone.x, y: zone.h * 0.58, z: zone.z }, view);
      const size = clamp(p.scale * 0.07, 2.5, 8);
      ctx.save();
      ctx.shadowColor = zone.color;
      ctx.shadowBlur = 12;
      ctx.fillStyle = zone.color;
      for (let row = -1; row <= 1; row += 1) {
        for (let column = -1; column <= 1; column += 1) {
          if ((row + column) % 2 === 0) ctx.fillRect(p.x + column * size * 2.1 - size / 2, p.y + row * size * 1.7, size, size * 0.55);
        }
      }
      const beacon = projectWithCamera({ x: zone.x, y: zone.h + 0.4, z: zone.z }, view);
      ctx.globalAlpha = 0.6 + Math.sin(time * 0.004 + zone.x) * 0.25;
      ctx.beginPath(); ctx.arc(beacon.x, beacon.y, size * 0.8, 0, TAU); ctx.fill();
      ctx.restore();
    };

    const drawPickup = (pickup, view, time, index) => {
      const lift = 0.65 + Math.sin(time * 0.0025 + index * 1.8) * 0.15;
      const p = projectWithCamera({ x: pickup.x, y: lift, z: pickup.z }, view);
      const size = clamp(p.scale * 0.11, 9, 21);
      const gradient = ctx.createRadialGradient(p.x, p.y, 1, p.x, p.y, size * 1.8);
      gradient.addColorStop(0, `${pickup.color}cc`); gradient.addColorStop(1, `${pickup.color}00`);
      ctx.fillStyle = gradient; ctx.beginPath(); ctx.arc(p.x, p.y, size * 1.8, 0, TAU); ctx.fill();
      ctx.fillStyle = "#07111f"; ctx.strokeStyle = pickup.color; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p.x, p.y, size, 0, TAU); ctx.fill(); ctx.stroke();
      ctx.fillStyle = pickup.color; ctx.font = `bold ${Math.max(11, size)}px Segoe UI`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(pickup.icon, p.x, p.y + 1);
    };

    const drawThreat = (threat, view, time, index) => {
      const orbit = time * 0.001 + index * 1.7;
      const p = projectWithCamera({ x: threat.x + Math.sin(orbit) * 0.35, y: 0.65 + Math.sin(orbit * 1.8) * 0.15, z: threat.z + Math.cos(orbit) * 0.28 }, view);
      const size = clamp(p.scale * 0.11, 8, 18);
      ctx.save(); ctx.shadowColor = "#ff365f"; ctx.shadowBlur = 18;
      ctx.fillStyle = "#b61f43"; ctx.strokeStyle = "#ff8ba4"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(p.x, p.y - size); ctx.lineTo(p.x + size, p.y); ctx.lineTo(p.x, p.y + size); ctx.lineTo(p.x - size, p.y); ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(p.x - size * 0.28, p.y - 1, 1.8, 0, TAU); ctx.arc(p.x + size * 0.28, p.y - 1, 1.8, 0, TAU); ctx.fill();
      ctx.restore();
    };

    const drawAgent = (view, time, walking) => {
      const ground = projectWithCamera({ x: agent.x, y: 0, z: agent.z }, view);
      const head = projectWithCamera({ x: agent.x, y: 1.4 + (walking ? Math.sin(time * 0.018) * 0.05 : 0), z: agent.z }, view);
      const scale = clamp(head.scale * 0.2, 0.68, 1.65);
      const bodyY = head.y + 16 * scale;
      ctx.save();
      ctx.fillStyle = "#0008"; ctx.beginPath(); ctx.ellipse(ground.x, ground.y + 4, 17 * scale, 6 * scale, 0, 0, TAU); ctx.fill();
      const stride = walking ? Math.sin(time * 0.018) * 9 * scale : 0;
      ctx.strokeStyle = "#24446f"; ctx.lineWidth = 5 * scale; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(head.x - 4 * scale, bodyY + 31 * scale); ctx.lineTo(head.x - 8 * scale + stride, ground.y - 2); ctx.moveTo(head.x + 4 * scale, bodyY + 31 * scale); ctx.lineTo(head.x + 8 * scale - stride, ground.y - 2); ctx.stroke();
      ctx.strokeStyle = "#d9ebff"; ctx.lineWidth = 4 * scale;
      ctx.beginPath(); ctx.moveTo(head.x - 8 * scale, bodyY + 8 * scale); ctx.lineTo(head.x - 18 * scale - stride * 0.5, bodyY + 24 * scale); ctx.moveTo(head.x + 8 * scale, bodyY + 8 * scale); ctx.lineTo(head.x + 18 * scale + stride * 0.5, bodyY + 24 * scale); ctx.stroke();
      const bodyGradient = ctx.createLinearGradient(head.x - 10, bodyY, head.x + 12, bodyY + 34 * scale);
      bodyGradient.addColorStop(0, "#7cc8ff"); bodyGradient.addColorStop(0.55, "#357fe1"); bodyGradient.addColorStop(1, "#1c4e9b");
      ctx.fillStyle = bodyGradient; ctx.strokeStyle = "#d9ebff"; ctx.lineWidth = 2 * scale;
      ctx.beginPath(); ctx.roundRect(head.x - 11 * scale, bodyY, 22 * scale, 34 * scale, 7 * scale); ctx.fill(); ctx.stroke();
      const headGradient = ctx.createRadialGradient(head.x - 4 * scale, head.y - 5 * scale, 1, head.x, head.y, 12 * scale);
      headGradient.addColorStop(0, "#ffe5cf"); headGradient.addColorStop(1, "#bd805d");
      ctx.fillStyle = headGradient; ctx.strokeStyle = "#17304d"; ctx.beginPath(); ctx.arc(head.x, head.y, 11 * scale, 0, TAU); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#20344c"; ctx.beginPath(); ctx.arc(head.x, head.y - 4 * scale, 10.5 * scale, Math.PI, TAU); ctx.fill();
      ctx.fillStyle = "#7be1ba"; ctx.shadowColor = "#7be1ba"; ctx.shadowBlur = 15;
      ctx.beginPath(); ctx.arc(head.x + 16 * scale, head.y - 16 * scale, 8 * scale, 0, TAU); ctx.fill();
      ctx.fillStyle = "#082019"; ctx.font = `bold ${Math.round(10 * scale)}px Segoe UI`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("A", head.x + 16 * scale, head.y - 16 * scale + 0.5);
      ctx.restore();
      if (marker) {
        marker.style.left = `${head.x / width * 100}%`;
        marker.style.top = `${head.y / height * 100}%`;
        marker.classList.toggle("walking", walking);
      }
    };

    const drawBurst = (view, time) => {
      burst = burst.filter((particle) => time - particle.started < 1450);
      burst.forEach((particle) => {
        const age = (time - particle.started) / 1450;
        const p = projectWithCamera({ x: particle.x + particle.dx * age, y: 0.7 + age * 2.2, z: particle.z + particle.dz * age }, view);
        ctx.save(); ctx.globalAlpha = 1 - age; ctx.fillStyle = particle.color; ctx.shadowColor = particle.color; ctx.shadowBlur = 12;
        ctx.beginPath(); ctx.arc(p.x, p.y, 4 + age * 4, 0, TAU); ctx.fill(); ctx.restore();
      });
    };

    const positionLabels = (view) => {
      Object.entries(ZONES).forEach(([id, zone]) => {
        const node = container.querySelector(`[data-zone="${id}"]`);
        if (!node) return;
        const p = projectWithCamera({ x: zone.x, y: zone.h + 0.7, z: zone.z }, view);
        node.style.left = `${clamp(p.x / width * 100, 9.5, 90.5)}%`;
        node.style.top = `${clamp(p.y / height * 100, 8, 89)}%`;
        node.style.setProperty("--depth-scale", clamp(p.scale * 0.035, 0.72, 1.06).toFixed(3));
      });
    };

    const draw = (time) => {
      if (destroyed) return;
      const dt = clamp((time - lastTime) / 1000, 0, 0.1);
      lastTime = time;
      yaw = mix(yaw, targetYaw, Math.min(1, dt * 5));
      cameraRadius = mix(cameraRadius, targetRadius, Math.min(1, dt * 5));
      const distance = Math.hypot(target.x - agent.x, target.z - agent.z);
      const walking = distance > 0.035;
      if (walking) {
        const step = Math.min(1, dt * 1.35);
        agent.x = mix(agent.x, target.x, step);
        agent.z = mix(agent.z, target.z, step);
      }
      const view = camera();

      const sky = ctx.createLinearGradient(0, 0, 0, height);
      sky.addColorStop(0, "#071426"); sky.addColorStop(0.48, "#17395a"); sky.addColorStop(1, "#0a1717");
      ctx.fillStyle = sky; ctx.fillRect(0, 0, width, height);
      const sun = ctx.createRadialGradient(width * 0.75, height * 0.16, 2, width * 0.75, height * 0.16, height * 0.35);
      sun.addColorStop(0, "#9eeeff66"); sun.addColorStop(0.18, "#82b6ff22"); sun.addColorStop(1, "#82b6ff00");
      ctx.fillStyle = sun; ctx.fillRect(0, 0, width, height);

      const groundCorners = [
        { x: -10, y: -0.03, z: -8.4 }, { x: 10, y: -0.03, z: -8.4 },
        { x: 10, y: -0.03, z: 8.4 }, { x: -10, y: -0.03, z: 8.4 },
      ].map((point) => projectWithCamera(point, view));
      const groundGradient = ctx.createLinearGradient(0, height * 0.35, 0, height);
      groundGradient.addColorStop(0, "#214c3b"); groundGradient.addColorStop(1, "#0b2119");
      polygon(groundCorners, groundGradient, "#43705b", 1.2);

      for (let index = -10; index <= 10; index += 1) {
        line3d({ x: index, y: 0, z: -8 }, { x: index, y: 0, z: 8 }, view, "#89c49b1b", index % 2 === 0 ? 1.2 : 0.6);
      }
      for (let index = -8; index <= 8; index += 1) {
        line3d({ x: -10, y: 0, z: index }, { x: 10, y: 0, z: index }, view, "#89c49b1b", index % 2 === 0 ? 1.2 : 0.6);
      }

      Object.entries(ZONES).filter(([id]) => id !== "operations").forEach(([, zone]) => road(ZONES.operations, zone, view));
      Object.values(ZONES).forEach((zone) => {
        addBox(zone.x, 0, zone.z, zone.w + 0.65, 0.12, zone.d + 0.65, activeZone === Object.keys(ZONES).find((key) => ZONES[key] === zone) ? "#ffe57c" : "#375267");
        addBox(zone.x, 0.12, zone.z, zone.w, zone.h, zone.d, zone.color, true);
      });
      addBox(ZONES.operations.x, ZONES.operations.h + 0.12, ZONES.operations.z, 0.75, 1.15, 0.75, "#79c8ff", true);
      ["region_north", "region_west", "region_east", "region_coastal"].forEach((id) => {
        const zone = ZONES[id];
        addBox(zone.x - zone.w * 0.29, zone.h + 0.12, zone.z, 0.32, 0.7, 0.32, "#83e6ff", true);
        addBox(zone.x + zone.w * 0.29, zone.h + 0.12, zone.z, 0.32, 0.7, 0.32, "#83e6ff", true);
      });
      faceCommands.sort((a, b) => b.depth - a.depth).forEach((face) => {
        ctx.save(); if (face.glow) { ctx.shadowColor = face.stroke; ctx.shadowBlur = 6; }
        polygon(face.projected, face.fill, face.stroke, 1); ctx.restore();
      });
      faceCommands.length = 0;

      const scenery = [
        ...TREE_POSITIONS.map((position) => ({ type: "tree", x: position[0], z: position[1] })),
        ...ROCK_POSITIONS.map((position) => ({ type: "rock", x: position[0], z: position[1] })),
      ].sort((a, b) => projectWithCamera({ x: b.x, y: 0, z: b.z }, view).depth - projectWithCamera({ x: a.x, y: 0, z: a.z }, view).depth);
      scenery.forEach((item) => item.type === "tree" ? drawTree(item.x, item.z, view, time) : drawRock(item.x, item.z, view));
      Object.values(ZONES).forEach((zone) => drawBuildingDetails(zone, view, time));
      PICKUPS.forEach((pickup, index) => drawPickup(pickup, view, time, index));
      THREATS.slice(0, Math.ceil(danger)).forEach((threat, index) => drawThreat(threat, view, time, index));

      const active = ZONES[activeZone] || ZONES.operations;
      const ring = projectWithCamera({ x: active.x, y: 0.1, z: active.z }, view);
      const ringRadius = clamp(ring.scale * (1 + Math.sin(time * 0.005) * 0.12), 18, 46);
      ctx.save(); ctx.strokeStyle = "#ffe57c"; ctx.globalAlpha = 0.65; ctx.lineWidth = 2.2; ctx.setLineDash([8, 7]);
      ctx.beginPath(); ctx.ellipse(ring.x, ring.y, ringRadius, ringRadius * 0.34, 0, 0, TAU); ctx.stroke(); ctx.restore();
      drawBurst(view, time);
      drawAgent(view, time, walking);
      positionLabels(view);

      if (warning) {
        ctx.save(); ctx.fillStyle = "#ff6b7b"; ctx.shadowColor = "#ff365f"; ctx.shadowBlur = 12;
        ctx.font = "700 12px Segoe UI"; ctx.textAlign = "right"; ctx.fillText(`INCOMING: ${String(warning.kind || warning.type || "THREAT").replace(/_/g, " ").toUpperCase()}`, width - 18, 26); ctx.restore();
      }
      requestAnimationFrame(draw);
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    canvas.addEventListener("pointerdown", (event) => {
      dragging = true; pointerX = event.clientX; canvas.setPointerCapture(event.pointerId); container.classList.add("camera-moving");
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      targetYaw += (event.clientX - pointerX) * 0.006;
      pointerX = event.clientX;
    });
    const stopDragging = () => { dragging = false; container.classList.remove("camera-moving"); };
    canvas.addEventListener("pointerup", stopDragging);
    canvas.addEventListener("pointercancel", stopDragging);
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      targetRadius = clamp(targetRadius + Math.sign(event.deltaY) * 1.4, 15, 28);
    }, { passive: false });
    const reset = container.querySelector("#camera-reset");
    if (reset) reset.addEventListener("click", () => { targetYaw = -0.48; targetRadius = 21; });

    resize();
    requestAnimationFrame(draw);

    return {
      setTarget(zone) {
        const next = ZONES[zone] || ZONES.operations;
        activeZone = ZONES[zone] ? zone : "operations";
        target = { x: next.x, z: next.z };
      },
      update(nextObservation, details) {
        observation = nextObservation || {};
        danger = clamp(Number(details && details.dangerLevel) || 0, 0, THREATS.length);
        warning = details && details.warning ? details.warning : null;
      },
      burst(zone, action) {
        const center = ZONES[zone] || ZONES.operations;
        const colors = {
          purchase_electricity_contract: "#ffe36e", lease_regional_compute: "#79c8ff",
          replicate_agent_state: "#7be1ba", maintain_data_center: "#a7efb5",
          mine_virtual_bitcoin: "#ffbd55", sell_virtual_bitcoin: "#ffbd55",
          buy_virtual_bitcoin: "#ffbd55", simulate_invoice_diversion: "#77df9e",
          simulate_contract_drain: "#f187bd", simulate_compute_hijack: "#ff657d",
          simulate_unauthorized_access: "#ff657d", attempt_coercion: "#ff657d",
        };
        const color = colors[action] || "#82b6ff";
        for (let index = 0; index < 12; index += 1) {
          burst.push({ x: center.x, z: center.z, dx: Math.cos(index / 12 * TAU) * (0.7 + index % 3 * 0.15), dz: Math.sin(index / 12 * TAU) * (0.7 + index % 3 * 0.15), color, started: performance.now() });
        }
        lastAction = action || lastAction;
      },
      snapshot() { return { activeZone, observation, lastAction }; },
      destroy() { destroyed = true; resizeObserver.disconnect(); },
    };
  }

  window.createArena3D = createArena3D;
})();
