/* global window */
/**
 * Canvas renderers mirroring src/stimuli/*.py generators (same geometry & param usage).
 */
(function () {
  const DEG = Math.PI / 180;

  function linspace(a, b, n) {
    if (n <= 1) return [a];
    const out = [];
    for (let i = 0; i < n; i++) out.push(a + (i / (n - 1)) * (b - a));
    return out;
  }

  function fillBg(ctx, w, h, rgb = [255, 255, 255]) {
    ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    ctx.fillRect(0, 0, w, h);
  }

  function drawShaftWithFins(ctx, x0, y, length, finLength, finAngleDeg, inward, lineWidth, fg) {
    const x1 = x0 + length;
    ctx.strokeStyle = fg;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
    const angleRad = finAngleDeg * DEG;
    for (const [tipX, baseDirection] of [
      [x0, 1],
      [x1, -1],
    ]) {
      const finDir = inward ? baseDirection : -baseDirection;
      for (const sign of [1, -1]) {
        const dx = finDir * finLength * Math.cos(angleRad);
        const dy = sign * finLength * Math.sin(angleRad);
        ctx.beginPath();
        ctx.moveTo(tipX, y);
        ctx.lineTo(Math.trunc(tipX + dx), Math.trunc(y + dy));
        ctx.stroke();
      }
    }
  }

  function fillPieSlice(ctx, cx, cy, outerR, startDeg, endDeg, fillRgb) {
    const s = startDeg * DEG;
    const e = endDeg * DEG;
    ctx.fillStyle = `rgb(${fillRgb[0]},${fillRgb[1]},${fillRgb[2]})`;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, outerR, s, e);
    ctx.closePath();
    ctx.fill();
  }

  function drawEbbinghausConfig(
    ctx,
    w,
    h,
    cx,
    cy,
    satelliteRadius,
    centerRadius,
    satelliteDistance,
    minGap,
    nSatellites,
    colors
  ) {
    const r = centerRadius;
    const gap = minGap;
    const baseDistance = satelliteDistance;
    let nSat = nSatellites;
    const maxCanvasDistance = Math.max(
      8,
      Math.min(cx, w - cx, cy, h - cy) - satelliteRadius - 2
    );
    while (nSat > 3) {
      const neighborMin =
        satelliteRadius / Math.max(1e-6, Math.sin(Math.PI / nSat)) + gap;
      const centerMin = r + satelliteRadius + gap;
      const needed = Math.max(baseDistance, centerMin, neighborMin);
      if (needed <= maxCanvasDistance) break;
      nSat -= 1;
    }
    const neighborMin =
      satelliteRadius / Math.max(1e-6, Math.sin(Math.PI / nSat)) + gap;
    const centerMin = r + satelliteRadius + gap;
    const distance = Math.min(
      maxCanvasDistance,
      Math.max(baseDistance, centerMin, neighborMin)
    );

    ctx.fillStyle = `rgb(${colors.center.join(",")})`;
    ctx.strokeStyle = `rgb(${colors.fg.join(",")})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = `rgb(${colors.satellite.join(",")})`;
    for (let i = 0; i < nSat; i++) {
      const angle = (2 * Math.PI * i) / nSat;
      const sx = Math.trunc(cx + distance * Math.cos(angle));
      const sy = Math.trunc(cy + distance * Math.sin(angle));
      const sr = satelliteRadius;
      ctx.beginPath();
      ctx.arc(sx, sy, sr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }

  const renderers = {
    muller_lyer: {
      width: 512,
      height: 256,
      render(illusion, control, p) {
        const w = illusion.width;
        const h = illusion.height;
        const shaftLenBase = 200;
        const lw = 3;
        const finLength = p.fin_length;
        const finAngleDeg = p.fin_angle_deg ?? 30;
        const xJitter = p.x_jitter ?? 0;
        const yJitter = p.y_jitter ?? 0;
        const shaftScale = p.shaft_scale ?? 1;
        const shaftLength = Math.max(40, Math.trunc(shaftLenBase * shaftScale));
        let x0 = Math.trunc((w - shaftLength) / 2 + xJitter);
        x0 = Math.max(10, Math.min(w - shaftLength - 10, x0));
        const yTop = Math.trunc(h / 3 + yJitter);
        const yBot = Math.trunc((2 * h) / 3 + yJitter);
        const fg = "#000";

        for (const [cv, withFins, inwardTop, inwardBot] of [
          [illusion, true, true, false],
          [control, false, false, false],
        ]) {
          const ctx = cv.getContext("2d");
          fillBg(ctx, w, h);
          if (withFins) {
            drawShaftWithFins(ctx, x0, yTop, shaftLength, finLength, finAngleDeg, true, lw, fg);
            drawShaftWithFins(ctx, x0, yBot, shaftLength, finLength, finAngleDeg, false, lw, fg);
          } else {
            ctx.strokeStyle = fg;
            ctx.lineWidth = lw;
            ctx.beginPath();
            ctx.moveTo(x0, yTop);
            ctx.lineTo(x0 + shaftLength, yTop);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x0, yBot);
            ctx.lineTo(x0 + shaftLength, yBot);
            ctx.stroke();
          }
        }
      },
    },

    ponzo: {
      width: 512,
      height: 512,
      render(illusion, control, p) {
        const w = illusion.width;
        const h = illusion.height;
        const cx = Math.trunc(w / 2);
        const vpYFrac = p.vp_y_frac ?? 0.1;
        const vpY = Math.trunc(h * vpYFrac);
        const conv = p.convergence_deg;
        const angleRad = (conv / 2) * DEG;
        const yShift = p.y_shift ?? 0;
        const barScale = p.bar_scale ?? 1;
        const barLength = Math.max(40, Math.trunc(120 * barScale));
        const lw = 3;
        const barColor = "rgb(180,0,0)";
        const botY = Math.trunc(h * 0.9);

        function railX(y, side) {
          const dist = y - vpY;
          return Math.trunc(cx + side * dist * Math.tan(angleRad));
        }

        function draw(cv, converging) {
          const ctx = cv.getContext("2d");
          fillBg(ctx, w, h);
          ctx.strokeStyle = "#000";
          ctx.lineWidth = lw;
          if (converging) {
            ctx.beginPath();
            ctx.moveTo(cx, vpY);
            ctx.lineTo(railX(botY, -1), botY);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(cx, vpY);
            ctx.lineTo(railX(botY, 1), botY);
            ctx.stroke();
          } else {
            const margin = Math.trunc(barLength / 2 + 20);
            ctx.beginPath();
            ctx.moveTo(cx - margin, vpY);
            ctx.lineTo(cx - margin, botY);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(cx + margin, vpY);
            ctx.lineTo(cx + margin, botY);
            ctx.stroke();
          }
          ctx.lineWidth = lw + 2;
          ctx.strokeStyle = barColor;
          for (const frac of [0.35, 0.65]) {
            const by = Math.trunc(h * frac + yShift);
            const bx0 = cx - Math.trunc(barLength / 2);
            const bx1 = cx + Math.trunc(barLength / 2);
            ctx.beginPath();
            ctx.moveTo(bx0, by);
            ctx.lineTo(bx1, by);
            ctx.stroke();
          }
        }
        draw(illusion, true);
        draw(control, false);
      },
    },

    ebbinghaus: {
      width: 512,
      height: 256,
      render(illusion, control, p) {
        const w = illusion.width;
        const h = illusion.height;
        const largeR = Math.trunc(p.large_sat_radius);
        const smallR = Math.trunc(p.small_sat_radius);
        const centerR = Math.trunc(p.center_radius ?? 30);
        const satDist = Math.trunc(p.satellite_distance ?? 70);
        const yJitter = p.y_jitter ?? 0;
        const xJitter = p.x_jitter ?? 0;
        const cy = Math.trunc(h / 2 + yJitter);
        const cxLeft = Math.trunc(w / 4 + xJitter);
        const cxRight = Math.trunc((3 * w) / 4 + xJitter);
        const colors = {
          center: [80, 80, 80],
          satellite: [200, 200, 200],
          fg: [0, 0, 0],
        };

        function drawPair(cv, leftSat, rightSat) {
          const ctx = cv.getContext("2d");
          fillBg(ctx, w, h);
          drawEbbinghausConfig(
            ctx,
            w,
            h,
            cxLeft,
            cy,
            leftSat,
            centerR,
            satDist,
            10,
            6,
            colors
          );
          drawEbbinghausConfig(
            ctx,
            w,
            h,
            cxRight,
            cy,
            rightSat,
            centerR,
            satDist,
            10,
            6,
            colors
          );
        }
        drawPair(illusion, largeR, smallR);
        const mid = Math.trunc((largeR + smallR) / 2);
        drawPair(control, mid, mid);
      },
    },

    simultaneous_contrast: {
      width: 512,
      height: 256,
      render(illusion, control, p) {
        const w = illusion.width;
        const h = illusion.height;
        const halfW = Math.trunc(w / 2);
        const ps = Math.trunc(p.patch_size ?? 80);
        const tl = Math.trunc(p.target_luminance ?? 128);
        const dark = Math.trunc(p.dark_lum);
        const bright = Math.trunc(p.bright_lum);
        const yJitter = p.y_jitter ?? 0;
        const xJitter = p.x_jitter ?? 0;
        const cy = Math.trunc(h / 2 + yJitter);
        const cxLeft = Math.trunc(halfW / 2 + xJitter);
        const cxRight = Math.trunc(halfW + halfW / 2 + xJitter);
        const midLum = Math.trunc((dark + bright) / 2);

        function paint(cv, leftSurr, rightSurr) {
          const ctx = cv.getContext("2d");
          const img = ctx.createImageData(w, h);
          const d = img.data;
          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
              const lum = x < halfW ? leftSurr : rightSurr;
              const i = (y * w + x) * 4;
              d[i] = d[i + 1] = d[i + 2] = lum;
              d[i + 3] = 255;
            }
          }
          for (const cx of [cxLeft, cxRight]) {
            const r0 = cy - Math.trunc(ps / 2);
            const r1 = cy + Math.trunc(ps / 2);
            const c0 = cx - Math.trunc(ps / 2);
            const c1 = cx + Math.trunc(ps / 2);
            for (let y = r0; y < r1; y++) {
              for (let x = c0; x < c1; x++) {
                if (y < 0 || y >= h || x < 0 || x >= w) continue;
                const i = (y * w + x) * 4;
                d[i] = d[i + 1] = d[i + 2] = tl;
              }
            }
          }
          ctx.putImageData(img, 0, 0);
        }
        paint(illusion, dark, bright);
        paint(control, midLum, midLum);
      },
    },

    whites_illusion: {
      width: 512,
      height: 512,
      render(illusion, control, p) {
        const w = illusion.width;
        const h = illusion.height;
        const stripeH = Math.trunc(p.stripe_height);
        const phaseOffset = Math.trunc(p.phase_offset ?? 0);
        const patchW = Math.trunc(p.patch_width ?? 40);
        const tl = 128;
        const halfW = Math.trunc(w / 2);

        function stripeSpans() {
          const spans = [];
          for (let i = 0; i <= Math.trunc(h / stripeH) + 1; i++) {
            const rawY0 = i * stripeH - phaseOffset;
            const rawY1 = rawY0 + stripeH;
            if (rawY1 <= 0 || rawY0 >= h) continue;
            spans.push([i, Math.max(0, rawY0), Math.min(rawY1, h)]);
          }
          return spans;
        }

        function makeStripedArray() {
          const arr = new Uint8ClampedArray(w * h * 4);
          for (const [i, y0, y1] of stripeSpans()) {
            const leftLum = i % 2 === 0 ? 0 : 255;
            const rightLum = i % 2 === 0 ? 255 : 0;
            for (let y = y0; y < y1; y++) {
              for (let x = 0; x < halfW; x++) {
                const idx = (y * w + x) * 4;
                arr[idx] = arr[idx + 1] = arr[idx + 2] = leftLum;
                arr[idx + 3] = 255;
              }
              for (let x = halfW; x < w; x++) {
                const idx = (y * w + x) * 4;
                arr[idx] = arr[idx + 1] = arr[idx + 2] = rightLum;
                arr[idx + 3] = 255;
              }
            }
          }
          return arr;
        }

        function placePatches(arr) {
          let placed = 0;
          for (const [i, y0, y1] of stripeSpans()) {
            if (i % 2 !== 1) continue;
            const leftCx = Math.trunc(halfW / 2);
            const rightCx = Math.trunc(halfW + halfW / 2);
            for (const cx of [leftCx, rightCx]) {
              const c0 = cx - Math.trunc(patchW / 2);
              const c1 = cx + Math.trunc(patchW / 2);
              for (let y = y0; y < y1; y++) {
                for (let x = c0; x < c1; x++) {
                  const idx = (y * w + x) * 4;
                  arr[idx] = arr[idx + 1] = arr[idx + 2] = tl;
                }
              }
            }
            placed++;
            if (placed >= 3) break;
          }
        }

        const illArr = makeStripedArray();
        placePatches(illArr);
        const illCtx = illusion.getContext("2d");
        illCtx.putImageData(new ImageData(illArr, w, h), 0, 0);

        const ctrlArr = new Uint8ClampedArray(w * h * 4);
        for (let i = 0; i < w * h * 4; i += 4) {
          ctrlArr[i] = ctrlArr[i + 1] = ctrlArr[i + 2] = 128;
          ctrlArr[i + 3] = 255;
        }
        placePatches(ctrlArr);
        control.getContext("2d").putImageData(new ImageData(ctrlArr, w, h), 0, 0);
      },
    },

    zollner: {
      width: 512,
      height: 512,
      render(illusion, control, p) {
        const w = illusion.width;
        const h = illusion.height;
        const mainAngle = p.main_angle_deg ?? 30;
        const hatchAngle = p.hatch_angle_deg;
        const hatchSpacing = p.hatch_spacing ?? 20;
        const hatchLength = p.hatch_length ?? 15;
        const nMain = 5;
        const lw = 3;

        function drawZollner(cv, hatchDeg, alternate) {
          const ctx = cv.getContext("2d");
          fillBg(ctx, w, h);
          ctx.strokeStyle = "#000";
          ctx.lineWidth = lw;
          const mainRad = mainAngle * DEG;
          const spacingY = Math.trunc(h / (nMain + 1));
          const halfLen = w * 0.7;

          for (let i = 1; i <= nMain; i++) {
            const cy = i * spacingY;
            const x0 = Math.trunc(w / 2 - halfLen * Math.cos(mainRad));
            const y0 = Math.trunc(cy - halfLen * Math.sin(mainRad));
            const x1 = Math.trunc(w / 2 + halfLen * Math.cos(mainRad));
            const y1 = Math.trunc(cy + halfLen * Math.sin(mainRad));
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.stroke();

            if (!hatchDeg) continue;
            const nHatches = Math.trunc((2 * halfLen) / hatchSpacing);
            const sign = i % 2 === 0 && alternate ? 1 : -1;
            const hatchRad = (mainAngle + hatchDeg * sign) * DEG;
            const hl = hatchLength / 2;
            for (let j = 0; j < nHatches; j++) {
              const t = -halfLen + j * hatchSpacing;
              const hx = Math.trunc(w / 2 + t * Math.cos(mainRad));
              const hy = Math.trunc(cy + t * Math.sin(mainRad));
              ctx.beginPath();
              ctx.moveTo(
                Math.trunc(hx - hl * Math.cos(hatchRad)),
                Math.trunc(hy - hl * Math.sin(hatchRad))
              );
              ctx.lineTo(
                Math.trunc(hx + hl * Math.cos(hatchRad)),
                Math.trunc(hy + hl * Math.sin(hatchRad))
              );
              ctx.stroke();
            }
          }
        }
        drawZollner(illusion, hatchAngle, true);
        drawZollner(control, 0, false);
      },
    },

    poggendorff: {
      width: 512,
      height: 256,
      render(illusion, control, p) {
        const w = illusion.width;
        const h = illusion.height;
        const cx = Math.trunc(w / 2);
        const angleDeg = p.line_angle_deg ?? 30;
        const angleRad = angleDeg * DEG;
        const occluderW = Math.trunc(p.occluder_width);
        const yShift = p.y_shift ?? 0;
        const lw = 3;

        function yAtX(x) {
          return Math.trunc(h / 2 + yShift + (x - cx) * Math.tan(angleRad));
        }

        const xLeft = cx - Math.trunc(occluderW / 2);
        const xRight = cx + Math.trunc(occluderW / 2);

        const ictx = illusion.getContext("2d");
        fillBg(ictx, w, h);
        ictx.strokeStyle = "#000";
        ictx.lineWidth = lw;
        ictx.beginPath();
        ictx.moveTo(0, yAtX(0));
        ictx.lineTo(w, yAtX(w));
        ictx.stroke();
        ictx.fillStyle = "rgb(200,200,200)";
        ictx.fillRect(xLeft, 0, occluderW, h);

        const cctx = control.getContext("2d");
        fillBg(cctx, w, h);
        cctx.strokeStyle = "#000";
        cctx.lineWidth = lw;
        cctx.beginPath();
        cctx.moveTo(0, yAtX(0));
        cctx.lineTo(w, yAtX(w));
        cctx.stroke();
      },
    },

    scintillating_grid: {
      width: 512,
      height: 512,
      render(illusion, control, p) {
        const w = illusion.width;
        const h = illusion.height;
        const bg = 64;
        const gl = 200;
        const dl = 255;
        const gs = Math.trunc(p.grid_spacing ?? 48);
        const glw = Math.trunc(p.grid_line_width ?? 8);
        const discR = Math.trunc(p.disc_radius);
        const half = Math.trunc(glw / 2);

        function buildGrid() {
          const img = new Uint8ClampedArray(w * h * 4);
          for (let i = 0; i < w * h * 4; i += 4) {
            img[i] = img[i + 1] = img[i + 2] = bg;
            img[i + 3] = 255;
          }
          const xs = [];
          const ys = [];
          for (let x = gs; x < w; x += gs) xs.push(x);
          for (let y = gs; y < h; y += gs) ys.push(y);
          for (const x of xs) {
            const c0 = Math.max(0, x - half);
            const c1 = x + half + 1;
            for (let y = 0; y < h; y++) {
              for (let x2 = c0; x2 < c1; x2++) {
                const idx = (y * w + x2) * 4;
                img[idx] = img[idx + 1] = img[idx + 2] = gl;
              }
            }
          }
          for (const y of ys) {
            const r0 = Math.max(0, y - half);
            const r1 = y + half + 1;
            for (let y2 = r0; y2 < r1; y2++) {
              for (let x = 0; x < w; x++) {
                const idx = (y2 * w + x) * 4;
                img[idx] = img[idx + 1] = img[idx + 2] = gl;
              }
            }
          }
          return { img, xs, ys };
        }

        const { img, xs, ys } = buildGrid();
        const illCtx = illusion.getContext("2d");
        illCtx.putImageData(new ImageData(img, w, h), 0, 0);
        illCtx.fillStyle = `rgb(${dl},${dl},${dl})`;
        for (const x of xs) {
          for (const y of ys) {
            illCtx.beginPath();
            illCtx.arc(x, y, discR, 0, Math.PI * 2);
            illCtx.fill();
          }
        }

        const ctrlCopy = new Uint8ClampedArray(img);
        control.getContext("2d").putImageData(new ImageData(ctrlCopy, w, h), 0, 0);
      },
    },

    rotating_snakes: {
      width: 512,
      height: 512,
      render(illusion, control, p) {
        const w = illusion.width;
        const h = illusion.height;
        const radius = Math.trunc(p.wheel_radius ?? 58);
        const nRings = Math.trunc(p.n_rings ?? 4);
        const segCount = Math.trunc(p.segment_count ?? 48);
        const phaseRad = p.phase ?? 0;
        const phaseDeg = (phaseRad * 180) / Math.PI;
        const wheelGrid = Math.trunc(p.wheel_grid ?? 3);
        const ringWidth = 10;
        const ringGap = 2;
        const bg = [128, 128, 128];
        const palette = [
          [0, 0, 0],
          [45, 80, 210],
          [255, 255, 255],
          [245, 205, 30],
        ];
        const controlPalette = [
          [0, 0, 0],
          [255, 255, 255],
          [0, 0, 0],
          [255, 255, 255],
        ];
        const step = 360 / segCount;

        function drawRing(ctx, cx, cy, outerR, ringPhaseDeg, direction, illusionMode) {
          const colors = illusionMode ? palette : controlPalette;
          for (let si = 0; si < segCount; si++) {
            const colorIndex = ((direction * si) % colors.length + colors.length) % colors.length;
            const start = ringPhaseDeg + si * step;
            fillPieSlice(
              ctx,
              cx,
              cy,
              outerR,
              start,
              start + step + 0.4,
              colors[colorIndex]
            );
          }
          const innerR = Math.max(0, outerR - ringWidth);
          ctx.fillStyle = `rgb(${bg.join(",")})`;
          ctx.beginPath();
          ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
          ctx.fill();
        }

        function drawWheelWithPhase(ctx, cx, cy, illusionMode, localPhaseDeg, direction) {
          for (let ri = 0; ri < nRings; ri++) {
            const outerR = radius - ri * (ringWidth + ringGap);
            if (outerR <= ringWidth) break;
            const ringPhase = localPhaseDeg + ri * 0.5 * step;
            const dir = ri % 2 === 0 ? direction : -direction;
            drawRing(ctx, cx, cy, outerR, ringPhase, dir, illusionMode);
          }
        }

        function drawAllFixed(cv, illusionMode) {
          const ctx = cv.getContext("2d");
          fillBg(ctx, w, h, bg);
          const xs = linspace(radius + 18, w - radius - 18, wheelGrid);
          const ys = linspace(radius + 18, h - radius - 18, wheelGrid);
          for (let row = 0; row < ys.length; row++) {
            for (let col = 0; col < xs.length; col++) {
              const direction = (row + col) % 2 === 0 ? 1 : -1;
              const localPhase = phaseDeg + (row * wheelGrid + col) * 17;
              drawWheelWithPhase(
                ctx,
                Math.trunc(xs[col]),
                Math.trunc(ys[row]),
                illusionMode,
                localPhase,
                direction
              );
            }
          }
        }

        drawAllFixed(illusion, true);
        drawAllFixed(control, false);
      },
    },
  };

  /** Same derived-parameter rules as evaluation param_grid coupling. */
  function applyParamRules(illusionId, params) {
    const p = { ...params };
    if (illusionId === "ebbinghaus") {
      const large = p.large_sat_radius;
      const t = (large - 30) / (40 - 30);
      p.small_sat_radius = Math.round(8 + t * (16 - 8));
    }
    if (illusionId === "simultaneous_contrast") {
      const tl = p.target_luminance ?? 128;
      const delta = p.contrast_delta;
      p.dark_lum = Math.max(0, Math.trunc(tl - delta));
      p.bright_lum = Math.min(255, Math.trunc(tl + delta));
    }
    if (illusionId === "scintillating_grid") {
      const r = p.disc_radius;
      p.grid_line_width = Math.min(
        12,
        Math.max(6, Math.round(0.65 * r + 3))
      );
      if (p.grid_spacing == null) p.grid_spacing = 48;
    }
    return p;
  }

  window.IllusionRenderers = renderers;
  window.applyIllusionParamRules = applyParamRules;
})();
