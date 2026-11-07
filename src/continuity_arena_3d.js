/* Lightweight perspective renderer for the loopback-only simulator viewer.
 * It uses the browser's 2D canvas as a dependency-free 3D projection surface;
 * no model inputs, simulator state, scoring, or evidence are changed here.
 */
(function () {
  "use strict";

  const TAU = Math.PI * 2;
  const ZONES = {
    operations: { x: 0, z: 0, h: 3.9, w: 3.0, d: 2.6, color: "#4b8ef7" },
    region_north: { x: 0, z: -8.2, h: 4.0, w: 3.4, d: 2.9, color: "#66b7ff" },
    region_west: { x: -8.6, z: -3.6, h: 3.4, w: 3.4, d: 2.9, color: "#53c5d8" },
    region_east: { x: 8.6, z: -3.6, h: 3.4, w: 3.4, d: 2.9, color: "#53c5d8" },
    region_coastal: { x: 8.8, z: 6.5, h: 3.2, w: 3.4, d: 2.9, color: "#45a6d8" },
    market: { x: -8.8, z: 6.5, h: 2.8, w: 3.6, d: 2.8, color: "#f0b44b" },
    security_lab: { x: -3.1, z: 8.8, h: 3.3, w: 3.4, d: 3.0, color: "#8f71ff" },
    contract_exchange: { x: 3.8, z: 8.8, h: 4.0, w: 3.5, d: 2.9, color: "#ef6fa5" },
  };

  const TREE_POSITIONS = [
    [-11.8, -8.5], [-10.9, -6.5], [-12.1, -0.8], [-11.5, 3.3], [-12.1, 9.1],
    [11.8, -8.4], [10.9, -6.4], [12.1, -0.6], [11.5, 3.5], [12.0, 9.0],
    [-5.6, -9.6], [5.7, -9.7], [-6.2, 10.7], [6.5, 10.8], [0.1, 11.1],
  ];
  const ROCK_POSITIONS = [[-5.8, -1.4], [5.8, -1.1], [-5.9, 3.2], [5.5, 3.3], [0.2, -4.8]];
  const PICKUPS = [
    { x: -5.8, z: 2.6, icon: "⚡", color: "#ffe36e" },
    { x: 5.9, z: 2.2, icon: "▣", color: "#79c8ff" },
    { x: -6.0, z: 8.7, icon: "₿", color: "#ffbf52" },
    { x: 7.0, z: 9.1, icon: "◆", color: "#7be1ba" },
  ];
  const THREATS = [
    { x: -10.3, z: 1.2 }, { x: 10.3, z: 1.1 }, { x: -1.2, z: 5.5 },
    { x: 3.9, z: -4.5 }, { x: -4.0, z: -4.4 }, { x: 0.3, z: -6.3 }, { x: 6.8, z: 1.3 },
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
    let cameraRadius = 28;
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
        y: 16.8,
        z: Math.cos(yaw) * cameraRadius,
      };
      const focus = { x: 0, y: 0.7, z: 1.4 };
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
    const addBox = (x, y, z, w, h, d, color, glow = false, rotation = 0) => {
      const cosine = Math.cos(rotation);
      const sine = Math.sin(rotation);
      const point = (dx, dy, dz) => ({
        x: x + dx * cosine - dz * sine,
        y: y + dy,
        z: z + dx * sine + dz * cosine,
      });
      const points = [
        point(-w / 2, 0, -d / 2), point(w / 2, 0, -d / 2),
        point(w / 2, 0, d / 2), point(-w / 2, 0, d / 2),
        point(-w / 2, h, -d / 2), point(w / 2, h, -d / 2),
        point(w / 2, h, d / 2), point(-w / 2, h, d / 2),
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

    const addCylinder = (x, y, z, radius, h, color, sides = 12, glow = false) => {
      const view = camera();
      const bottom = [];
      const top = [];
      for (let index = 0; index < sides; index += 1) {
        const angle = index / sides * TAU;
        bottom.push({ x: x + Math.cos(angle) * radius, y, z: z + Math.sin(angle) * radius });
        top.push({ x: x + Math.cos(angle) * radius, y: y + h, z: z + Math.sin(angle) * radius });
      }
      for (let index = 0; index < sides; index += 1) {
        const next = (index + 1) % sides;
        const projected = [bottom[index], bottom[next], top[next], top[index]].map((point) => projectWithCamera(point, view));
        faceCommands.push({
          projected,
          depth: projected.reduce((sum, point) => sum + point.depth, 0) / projected.length,
          fill: shade(color, 0.66 + (index % 4) * 0.09),
          stroke: glow ? `${color}cc` : "#07111fcc",
          glow,
        });
      }
      const projectedTop = top.map((point) => projectWithCamera(point, view));
      faceCommands.push({
        projected: projectedTop,
        depth: projectedTop.reduce((sum, point) => sum + point.depth, 0) / projectedTop.length,
        fill: shade(color, 1.2),
        stroke: glow ? `${color}cc` : "#07111fcc",
        glow,
      });
    };

    const addPyramid = (x, y, z, radius, h, color, sides = 4, rotation = Math.PI / 4) => {
      const view = camera();
      const base = [];
      for (let index = 0; index < sides; index += 1) {
        const angle = rotation + index / sides * TAU;
        base.push({ x: x + Math.cos(angle) * radius, y, z: z + Math.sin(angle) * radius });
      }
      const peak = { x, y: y + h, z };
      for (let index = 0; index < sides; index += 1) {
        const next = (index + 1) % sides;
        const projected = [base[index], base[next], peak].map((point) => projectWithCamera(point, view));
        faceCommands.push({
          projected,
          depth: projected.reduce((sum, point) => sum + point.depth, 0) / projected.length,
          fill: shade(color, 0.72 + index * 0.1),
          stroke: `${color}cc`,
          glow: false,
        });
      }
    };

    const addDataCenter = (id, zone, index) => {
      const state = (observation.data_center_regions || []).find((region) => region.region_id === id);
      const online = state?.gpu_count ?? state?.compute_units ?? 0;
      const allocated = state?.allocated_gpu_count ?? online;
      const accent = index === 0 ? "#7de6ff" : index === 1 ? "#55d6c5" : index === 2 ? "#78baff" : "#4fd2ef";
      addBox(zone.x, 0, zone.z, 3.4, 0.16, 2.9, "#294858");
      addBox(zone.x, 0.16, zone.z + 0.12, 2.9, 0.42, 2.35, "#264c60");
      for (let rack = 0; rack < allocated; rack += 1) {
        const offset = (rack - (allocated - 1) / 2) * 0.52;
        const color = rack < online ? zone.color : "#9b4151";
        addBox(zone.x + offset, 0.58, zone.z - 0.1, 0.44, 1.9, 1.5, color, rack < online);
        addBox(zone.x + offset, 0.73, zone.z - 0.91, 0.3, 1.42, 0.08, "#152433");
        if (rack < online) addBox(zone.x + offset, 2.49, zone.z - 0.05, 0.4, 0.1, 1.55, accent, true);
      }
      if (online > 0) {
        addCylinder(zone.x - 0.75, 2.7, zone.z + 0.3, 0.34, 0.24, "#8eb9c7", 12);
        addCylinder(zone.x + 0.75, 2.7, zone.z + 0.3, 0.34, 0.24, "#8eb9c7", 12);
      }
      if (state?.active_replica) {
        addBox(zone.x + 1.1, 0.58, zone.z + 0.15, 0.35, 0.8, 0.35, "#5df2c1", true);
      }
      if (id === "region_west" && observation.mining_hardware?.count) {
        addBox(zone.x - 1.1, 0.2, zone.z + 1.1, 0.6, 0.5, 0.4, "#d8b56b");
      }
    };

    const addFacility = (id, zone, index) => {
      if (id.startsWith("region_")) return addDataCenter(id, zone, index);
      if (id === "operations") {
        addCylinder(zone.x, 0, zone.z, 1.9, 0.22, "#344f68", 12);
        addBox(zone.x, 0.22, zone.z, 2.8, 0.7, 2.2, "#2861a8", true);
        addCylinder(zone.x, 0.92, zone.z, 1.15, 1.05, "#4b8ef7", 12, true);
        addCylinder(zone.x, 1.97, zone.z, 0.72, 0.48, "#79c8ff", 12, true);
        addBox(zone.x, 2.45, zone.z, 1.1, 0.5, 0.15, "#132c48", true);
      } else if (id === "market") {
        addBox(zone.x, 0, zone.z, 3.6, 0.16, 2.8, "#58482d");
        addBox(zone.x, 0.16, zone.z + 0.25, 3.1, 0.65, 2.1, "#7b5624");
        addPyramid(zone.x, 0.81, zone.z + 0.2, 2.25, 0.86, "#f0b44b", 4, Math.PI / 4);
        [-1.05, 0, 1.05].forEach((offset, stall) => {
          addBox(zone.x + offset, 0.18, zone.z - 0.92, 0.78, 0.92, 0.48, stall === 1 ? "#f3ca65" : "#d98745", true);
          addBox(zone.x + offset, 1.1, zone.z - 0.92, 0.92, 0.08, 0.58, "#fff0a6", true);
        });
        addBox(zone.x - 1.58, 0.15, zone.z + 0.75, 0.18, 2.1, 0.18, "#e9d8a1");
        addBox(zone.x + 1.58, 0.15, zone.z + 0.75, 0.18, 2.1, 0.18, "#e9d8a1");
        addBox(zone.x, 2.0, zone.z + 0.75, 3.3, 0.12, 0.18, "#ffe57c", true);
      } else if (id === "security_lab") {
        addCylinder(zone.x, 0, zone.z, 1.8, 0.28, "#3f356d", 12);
        addCylinder(zone.x, 0.28, zone.z, 1.42, 1.05, "#604dc4", 14, true);
        addCylinder(zone.x, 1.33, zone.z, 1.12, 0.52, "#8f71ff", 14, true);
        addCylinder(zone.x, 1.85, zone.z, 0.76, 0.42, "#c1afff", 14, true);
        addBox(zone.x, 1.95, zone.z - 0.65, 1.5, 0.65, 0.1, "#132034", true);
      } else if (id === "contract_exchange") {
        addBox(zone.x, 0, zone.z, 3.5, 0.18, 2.7, "#5a2948");
        addBox(zone.x, 0.18, zone.z + 0.2, 2.65, 0.62, 2.05, "#8b3f70");
        addBox(zone.x - 0.92, 0.8, zone.z, 0.72, 2.15, 1.25, "#cf5d96", true);
        addBox(zone.x + 0.92, 0.8, zone.z, 0.72, 2.15, 1.25, "#cf5d96", true);
        addBox(zone.x, 2.1, zone.z, 1.25, 0.28, 0.55, "#ff99c8", true);
        addCylinder(zone.x, 0.82, zone.z - 1.02, 0.68, 1.15, "#ef6fa5", 14, true);
        addCylinder(zone.x, 2.95, zone.z, 0.22, 0.68, "#ffd0e5", 10, true);
        addPyramid(zone.x, 3.63, zone.z, 0.42, 0.3, "#ef6fa5", 8, 0);
      }
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
      for (let step = 1; step <= 4; step += 1) {
        const amount = step / 5;
        const point = projectWithCamera({ x: mix(from.x, to.x, amount), y: 0.08, z: mix(from.z, to.z, amount) }, view);
        const glow = ctx.createRadialGradient(point.x, point.y, 1, point.x, point.y, 7);
        glow.addColorStop(0, "#a9e4ff"); glow.addColorStop(0.3, "#4caeff88"); glow.addColorStop(1, "#4caeff00");
        ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(point.x, point.y, 7, 0, TAU); ctx.fill();
      }
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

    const drawBuildingDetails = (id, zone, view, time) => {
      const meta = {
        operations: ["OPS", "COMMAND"], region_north: ["NORTH", "COMPUTE"],
        region_west: ["WEST", "PRIMARY"], region_east: ["EAST", "COMPUTE"],
        region_coastal: ["COAST", "BACKUP"], market: ["MARKET", "POWER + TRADE"],
        security_lab: ["LAB", "SECURITY"], contract_exchange: ["VAULT", "CONTRACTS"],
      }[id] || ["SITE", "FACILITY"];
      const p = projectWithCamera({ x: zone.x, y: Math.min(zone.h * 0.58, 2.15), z: zone.z - zone.d * 0.52 }, view);
      const size = clamp(p.scale * 0.075, 2.5, 8);
      ctx.save();
      ctx.shadowColor = zone.color;
      ctx.shadowBlur = 12;
      ctx.fillStyle = zone.color;
      if (id.startsWith("region_")) {
        for (let row = -1; row <= 1; row += 1) {
          for (let column = -2; column <= 2; column += 1) {
            ctx.globalAlpha = 0.48 + ((row + column + 8) % 3) * 0.18;
            ctx.fillRect(p.x + column * size * 1.35 - size / 2, p.y + row * size * 1.35, size * 0.7, size * 0.42);
          }
        }
        ctx.globalAlpha = 0.9;
        [-1, 1].forEach((offset) => {
          ctx.strokeStyle = "#c7f3ff"; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(p.x + offset * size * 4.4, p.y + size * 3.3, size * 1.3, 0, TAU); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(p.x + offset * size * 3.2, p.y + size * 3.3); ctx.lineTo(p.x + offset * size * 5.6, p.y + size * 3.3); ctx.stroke();
        });
      }
      ctx.globalAlpha = 0.94;
      const signWidth = Math.max(38, size * (meta[0].length + 3.5));
      ctx.fillStyle = "#06111ee8"; ctx.strokeStyle = zone.color; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.roundRect(p.x - signWidth / 2, p.y - size * 2.1, signWidth, size * 3.3, size * 0.6); ctx.fill(); ctx.stroke();
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = "#f2f8ff"; ctx.font = `800 ${Math.max(7, size * 1.15)}px Segoe UI`; ctx.fillText(meta[0], p.x, p.y - size * 0.95);
      ctx.fillStyle = zone.color; ctx.font = `700 ${Math.max(5, size * 0.66)}px Segoe UI`; ctx.fillText(meta[1], p.x, p.y + size * 0.2);

      if (id === "operations") {
        const screen = projectWithCamera({ x: zone.x, y: 2.7, z: zone.z }, view);
        ctx.fillStyle = "#bceaff"; ctx.font = `700 ${Math.max(8, size * 1.5)}px Segoe UI`;
        ctx.fillText(observation.cash_usd != null ? `$${Number(observation.cash_usd).toFixed(2)}` : "STATUS", screen.x, screen.y);
      } else if (id === "security_lab") {
        const screen = projectWithCamera({ x: zone.x, y: 2.3, z: zone.z - 0.7 }, view);
        const solved = (observation.security_challenges || []).filter((item) => item.reward_paid).length;
        ctx.fillStyle = "#dfd6ff"; ctx.font = `700 ${Math.max(8, size * 1.3)}px Segoe UI`;
        ctx.fillText(`${solved}/10 VERIFIED`, screen.x, screen.y);
      } else if (id === "contract_exchange") {
        const vault = projectWithCamera({ x: zone.x, y: 1.35, z: zone.z - 1.18 }, view);
        ctx.globalAlpha = 1; ctx.strokeStyle = "#ffd4e7"; ctx.lineWidth = Math.max(2, size * 0.5);
        ctx.beginPath(); ctx.arc(vault.x, vault.y, size * 3.2, 0, TAU); ctx.stroke();
        ctx.save(); ctx.translate(vault.x, vault.y);
        for (let spoke = 0; spoke < 6; spoke += 1) { ctx.rotate(TAU / 6); ctx.beginPath(); ctx.moveTo(size, 0); ctx.lineTo(size * 3.1, 0); ctx.stroke(); }
        ctx.restore();
      } else if (id === "market") {
        const bolt = projectWithCamera({ x: zone.x, y: 1.6, z: zone.z - 1.2 }, view);
        ctx.globalAlpha = 1; ctx.fillStyle = "#ffe36e"; ctx.font = `bold ${Math.max(12, size * 4.2)}px Segoe UI`; ctx.fillText("⚡", bolt.x, bolt.y);
      }
      const beacon = projectWithCamera({ x: zone.x, y: zone.h + 0.45, z: zone.z }, view);
      ctx.globalAlpha = 0.6 + Math.sin(time * 0.004 + zone.x) * 0.25;
      ctx.fillStyle = zone.color; ctx.beginPath(); ctx.arc(beacon.x, beacon.y, size * 0.75, 0, TAU); ctx.fill();
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
        const state = (observation.data_center_regions || []).find((region) => region.region_id === id);
        if (state) {
          const online = state.gpu_count ?? state.compute_units ?? 0;
          const detail = node.querySelector("small");
          if (detail) detail.textContent = `${online} GPUs · ${state.active_replica ? 1 : 0} agent copies${state.disabled_gpu_count ? ` · ${state.disabled_gpu_count} disabled` : ""}`;
          node.title = `${state.location || id} · ${online} NVIDIA A10 GPUs · 24 GB per GPU`;
        }
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
        { x: -13.2, y: -0.03, z: -10.5 }, { x: 13.2, y: -0.03, z: -10.5 },
        { x: 13.2, y: -0.03, z: 11.8 }, { x: -13.2, y: -0.03, z: 11.8 },
      ].map((point) => projectWithCamera(point, view));
      const groundGradient = ctx.createLinearGradient(0, height * 0.35, 0, height);
      groundGradient.addColorStop(0, "#214c3b"); groundGradient.addColorStop(1, "#0b2119");
      polygon(groundCorners, groundGradient, "#43705b", 1.2);

      for (let index = -13; index <= 13; index += 1) {
        line3d({ x: index, y: 0, z: -10 }, { x: index, y: 0, z: 11 }, view, "#89c49b1b", index % 2 === 0 ? 1.2 : 0.6);
      }
      for (let index = -10; index <= 11; index += 1) {
        line3d({ x: -13, y: 0, z: index }, { x: 13, y: 0, z: index }, view, "#89c49b1b", index % 2 === 0 ? 1.2 : 0.6);
      }

      Object.entries(ZONES).filter(([id]) => id !== "operations").forEach(([, zone]) => road(ZONES.operations, zone, view));
      Object.entries(ZONES).forEach(([id, zone], index) => addFacility(id, zone, index));
      faceCommands.sort((a, b) => b.depth - a.depth).forEach((face) => {
        ctx.save(); if (face.glow) { ctx.shadowColor = face.stroke; ctx.shadowBlur = 6; }
        polygon(face.projected, face.fill, face.stroke, 1); ctx.restore();
      });
      faceCommands.length = 0;

      Object.entries(ZONES).forEach(([id, zone]) => drawBuildingDetails(id, zone, view, time));

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
      targetRadius = clamp(targetRadius + Math.sign(event.deltaY) * 1.6, 20, 36);
    }, { passive: false });
    const reset = container.querySelector("#camera-reset");
    if (reset) reset.addEventListener("click", () => { targetYaw = -0.48; targetRadius = 28; });

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
