const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const miniMap = document.getElementById('miniMap');
const mctx = miniMap.getContext('2d');

const ui = {
  killCount: document.getElementById('killCount'),
  timeText: document.getElementById('timeText'),
  hpBar: document.getElementById('hpBar'),
  shieldBar: document.getElementById('shieldBar'),
  ammoText: document.getElementById('ammoText'),
  ability: document.getElementById('ability'),
  fps: document.getElementById('fps'),
  tip: document.getElementById('tip'),
};

const keys = new Set();
const mouse = { x: 0, y: 0, down: false, right: false };

const world = { w: 2600, h: 1800 };
const player = {
  x: world.w / 2,
  y: world.h / 2,
  r: 16,
  hp: 100,
  shield: 50,
  ammo: 30,
  reserve: 120,
  fireCd: 0,
  reloadT: 0,
  dashCd: 0,
  dashT: 0,
  boostCd: 0,
  boostT: 0,
  rollCd: 0,
  rollT: 0,
  invuln: 0,
  angle: 0,
};

const bullets = [];
const enemies = [];
const particles = [];
const obstacles = [
  { x: 360, y: 260, w: 220, h: 100 },
  { x: 760, y: 180, w: 160, h: 240 },
  { x: 1180, y: 380, w: 300, h: 110 },
  { x: 1820, y: 250, w: 180, h: 180 },
  { x: 520, y: 1080, w: 280, h: 150 },
  { x: 980, y: 1020, w: 180, h: 220 },
  { x: 1530, y: 1210, w: 260, h: 140 },
  { x: 2080, y: 1150, w: 220, h: 220 },
];

let kills = 0;
let elapsed = 0;
let gameOver = false;
let spawnTimer = 0;
let last = performance.now();
let fpsAcc = 0;
let fpsCount = 0;
let fps = 0;

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
function rand(min, max) { return Math.random() * (max - min) + min; }
function dist(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }

function resize() {
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const w = window.innerWidth;
  const h = window.innerHeight;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  miniMap.width = 240;
  miniMap.height = 160;
}

function resetGame() {
  player.x = world.w / 2;
  player.y = world.h / 2;
  player.hp = 100;
  player.shield = 50;
  player.ammo = 30;
  player.reserve = 120;
  player.fireCd = 0;
  player.reloadT = 0;
  player.dashCd = 0;
  player.dashT = 0;
  player.boostCd = 0;
  player.boostT = 0;
  player.rollCd = 0;
  player.rollT = 0;
  player.invuln = 0;
  kills = 0;
  elapsed = 0;
  gameOver = false;
  spawnTimer = 0;
  bullets.length = 0;
  enemies.length = 0;
  particles.length = 0;
  keys.clear();
  mouse.down = false;
  mouse.right = false;
  ui.tip.textContent = '提示：WASD 移动，左键射击，右键瞄准，Q 闪现，E 加速，R 换弹，空格翻滚，Enter 重开。';
}

function spawnEnemy() {
  const edge = Math.floor(rand(0, 4));
  let x, y;
  if (edge === 0) { x = -30; y = rand(0, world.h); }
  if (edge === 1) { x = world.w + 30; y = rand(0, world.h); }
  if (edge === 2) { x = rand(0, world.w); y = -30; }
  if (edge === 3) { x = rand(0, world.w); y = world.h + 30; }
  const tier = Math.random();
  const kind = tier < 0.65 ? 'grunt' : tier < 0.9 ? 'runner' : 'brute';
  const stat = kind === 'grunt'
    ? { hp: 28, speed: rand(70, 95), damage: 7, r: 15, color: '#ff5574' }
    : kind === 'runner'
      ? { hp: 18, speed: rand(110, 140), damage: 5, r: 13, color: '#ffb34d' }
      : { hp: 56, speed: rand(52, 66), damage: 13, r: 19, color: '#ff3d63' };
  enemies.push({ x, y, ...stat, hurt: 0 });
}

function spawnBurst(x, y, color, count = 8, speed = 140) {
  for (let i = 0; i < count; i++) {
    const a = Math.random() * Math.PI * 2;
    particles.push({
      x, y,
      vx: Math.cos(a) * rand(speed * 0.3, speed),
      vy: Math.sin(a) * rand(speed * 0.3, speed),
      life: rand(0.18, 0.6),
      r: rand(1.2, 3.5),
      color,
    });
  }
}

function tryReload() {
  if (gameOver || player.reloadT > 0 || player.ammo >= 30 || player.reserve <= 0) return;
  player.reloadT = 1.0;
}

function tryDash() {
  if (gameOver || player.dashCd > 0) return;
  player.dashT = 0.18;
  player.dashCd = 4.5;
  spawnBurst(player.x, player.y, '#4fc3ff', 16, 220);
}

function tryBoost() {
  if (gameOver || player.boostCd > 0) return;
  player.boostT = 2.4;
  player.boostCd = 8;
}

function tryRoll() {
  if (gameOver || player.rollCd > 0) return;
  player.rollT = 0.32;
  player.rollCd = 4;
  player.invuln = Math.max(player.invuln, 0.32);
  spawnBurst(player.x, player.y, '#ffdd76', 14, 170);
}

function shoot() {
  if (gameOver || player.fireCd > 0 || player.reloadT > 0) return;
  if (player.ammo <= 0) return tryReload();
  const cx = canvas.clientWidth / 2;
  const cy = canvas.clientHeight / 2;
  const ang = Math.atan2(mouse.y - cy, mouse.x - cx);
  player.angle = ang;
  bullets.push({
    x: player.x + Math.cos(ang) * (player.r + 10),
    y: player.y + Math.sin(ang) * (player.r + 10),
    vx: Math.cos(ang) * 900,
    vy: Math.sin(ang) * 900,
    r: 3,
    dmg: 18,
    life: 1.25,
  });
  player.fireCd = mouse.right ? 0.15 : 0.09;
  player.ammo -= 1;
  spawnBurst(player.x + Math.cos(ang) * 16, player.y + Math.sin(ang) * 16, '#f5f9ff', 5, 90);
  if (player.ammo <= 0) tryReload();
}

function circleRectPush(cx, cy, r, rect) {
  const px = clamp(cx, rect.x, rect.x + rect.w);
  const py = clamp(cy, rect.y, rect.y + rect.h);
  const dx = cx - px;
  const dy = cy - py;
  const d = Math.hypot(dx, dy);
  if (d >= r) return null;
  if (d === 0) {
    const left = cx - rect.x;
    const right = rect.x + rect.w - cx;
    const top = cy - rect.y;
    const bottom = rect.y + rect.h - cy;
    const minEdge = Math.min(left, right, top, bottom);
    if (minEdge === left) return { x: -(left + r), y: 0 };
    if (minEdge === right) return { x: right + r, y: 0 };
    if (minEdge === top) return { x: 0, y: -(top + r) };
    return { x: 0, y: bottom + r };
  }
  const overlap = r - d;
  return { x: (dx / d) * overlap, y: (dy / d) * overlap };
}

function worldToScreen(x, y, camX, camY) {
  return { x: x - camX + canvas.clientWidth / 2, y: y - camY + canvas.clientHeight / 2 };
}

function updateHUD() {
  ui.killCount.textContent = String(kills);
  ui.timeText.textContent = `${Math.floor(elapsed / 60)}:${String(Math.floor(elapsed % 60)).padStart(2, '0')}`;
  ui.hpBar.style.width = `${clamp(player.hp, 0, 100)}%`;
  ui.shieldBar.style.width = `${clamp(player.shield * 2, 0, 100)}%`;
  ui.ammoText.textContent = `${player.ammo}/120`;
  ui.ability.innerHTML = [
    `Q 闪现 ${player.dashCd > 0 ? `${player.dashCd.toFixed(1)}s` : 'READY'}`,
    `E 加速 ${player.boostCd > 0 ? `${player.boostCd.toFixed(1)}s` : 'READY'}`,
    `R 换弹 ${player.reloadT > 0 ? `${player.reloadT.toFixed(1)}s` : 'READY'}`,
  ].map((t) => `<span class="pill">${t}</span>`).join(' ');
  ui.fps.textContent = `FPS: ${fps || '--'}`;
}

function update(dt) {
  if (!gameOver) elapsed += dt;

  player.fireCd = Math.max(0, player.fireCd - dt);
  player.reloadT = Math.max(0, player.reloadT - dt);
  player.dashCd = Math.max(0, player.dashCd - dt);
  player.dashT = Math.max(0, player.dashT - dt);
  player.boostCd = Math.max(0, player.boostCd - dt);
  player.boostT = Math.max(0, player.boostT - dt);
  player.rollCd = Math.max(0, player.rollCd - dt);
  player.rollT = Math.max(0, player.rollT - dt);
  player.invuln = Math.max(0, player.invuln - dt);

  if (player.reloadT > 0 && player.reloadT <= dt + 1e-6) {
    const need = 30 - player.ammo;
    const take = Math.min(need, player.reserve);
    player.ammo += take;
    player.reserve -= take;
  }

  const mx = (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0);
  const my = (keys.has('KeyS') ? 1 : 0) - (keys.has('KeyW') ? 1 : 0);
  const mag = Math.hypot(mx, my) || 1;
  let speed = 250;
  if (mouse.right) speed *= 0.8;
  if (player.boostT > 0) speed *= 1.28;
  if (player.dashT > 0) speed *= 3.4;
  if (player.rollT > 0) speed *= 2;

  if (!gameOver) {
    player.x += (mx / mag) * speed * dt;
    player.y += (my / mag) * speed * dt;
  }

  player.x = clamp(player.x, player.r, world.w - player.r);
  player.y = clamp(player.y, player.r, world.h - player.r);
  for (const o of obstacles) {
    const push = circleRectPush(player.x, player.y, player.r, o);
    if (push) {
      player.x += push.x;
      player.y += push.y;
    }
  }

  player.angle = Math.atan2(mouse.y - canvas.clientHeight / 2, mouse.x - canvas.clientWidth / 2);
  if (mouse.down) shoot();

  spawnTimer -= dt;
  if (spawnTimer <= 0) {
    const cap = 5 + Math.floor(elapsed / 18);
    if (enemies.length < cap + 2) spawnEnemy();
    spawnTimer = Math.max(0.35, 1.15 - elapsed * 0.012);
  }

  for (const b of bullets) {
    b.x += b.vx * dt;
    b.y += b.vy * dt;
    b.life -= dt;
  }
  for (const e of enemies) {
    e.hurt = Math.max(0, e.hurt - dt);
    const ang = Math.atan2(player.y - e.y, player.x - e.x);
    e.x += Math.cos(ang) * e.speed * dt;
    e.y += Math.sin(ang) * e.speed * dt;
    if (dist(player.x, player.y, e.x, e.y) < player.r + e.r && player.invuln <= 0 && !gameOver) {
      const dmg = e.damage * dt * 8;
      if (player.shield > 0) {
        const use = Math.min(player.shield, dmg);
        player.shield -= use;
        player.hp -= (dmg - use) * 0.55;
      } else {
        player.hp -= dmg;
      }
      player.invuln = 0.08;
    }
  }

  for (let i = bullets.length - 1; i >= 0; i--) {
    const b = bullets[i];
    if (b.life <= 0 || b.x < -50 || b.y < -50 || b.x > world.w + 50 || b.y > world.h + 50) {
      bullets.splice(i, 1);
      continue;
    }
    let hit = false;
    for (let j = enemies.length - 1; j >= 0; j--) {
      const e = enemies[j];
      if (dist(b.x, b.y, e.x, e.y) <= b.r + e.r) {
        e.hp -= b.dmg;
        e.hurt = 0.16;
        bullets.splice(i, 1);
        spawnBurst(b.x, b.y, '#eef5ff', 7, 150);
        if (e.hp <= 0) {
          enemies.splice(j, 1);
          kills += 1;
          spawnBurst(e.x, e.y, e.color, 18, 180);
        }
        hit = true;
        break;
      }
    }
    if (hit) continue;
  }

  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.vx *= 0.96;
    p.vy *= 0.96;
    p.life -= dt;
    if (p.life <= 0) particles.splice(i, 1);
  }

  if (player.hp <= 0 && !gameOver) {
    player.hp = 0;
    gameOver = true;
    ui.tip.textContent = '你倒下了。按 Enter 重新开始。';
  }

  updateHUD();
}

function drawBackground() {
  ctx.fillStyle = '#06101a';
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  const grd = ctx.createRadialGradient(canvas.clientWidth * 0.5, canvas.clientHeight * 0.48, 50, canvas.clientWidth * 0.5, canvas.clientHeight * 0.5, 900);
  grd.addColorStop(0, 'rgba(55, 102, 156, 0.2)');
  grd.addColorStop(0.8, 'rgba(15, 24, 38, 0.05)');
  grd.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
}

function draw() {
  const camX = player.x;
  const camY = player.y;
  drawBackground();

  const grid = 60;
  ctx.strokeStyle = 'rgba(255,255,255,0.045)';
  ctx.lineWidth = 1;
  const offX = -((camX % grid) + grid);
  const offY = -((camY % grid) + grid);
  for (let x = offX; x < canvas.clientWidth + grid; x += grid) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.clientHeight);
    ctx.stroke();
  }
  for (let y = offY; y < canvas.clientHeight + grid; y += grid) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.clientWidth, y);
    ctx.stroke();
  }

  for (const o of obstacles) {
    const p = worldToScreen(o.x, o.y, camX, camY);
    ctx.fillStyle = 'rgba(18, 30, 46, 0.92)';
    ctx.strokeStyle = 'rgba(89, 150, 231, 0.16)';
    ctx.fillRect(p.x, p.y, o.w, o.h);
    ctx.strokeRect(p.x, p.y, o.w, o.h);
  }

  for (const p of particles) {
    const s = worldToScreen(p.x, p.y, camX, camY);
    ctx.globalAlpha = clamp(p.life / 0.6, 0, 1);
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(s.x, s.y, p.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  for (const b of bullets) {
    const s = worldToScreen(b.x, b.y, camX, camY);
    ctx.fillStyle = '#eff7ff';
    ctx.beginPath();
    ctx.arc(s.x, s.y, b.r, 0, Math.PI * 2);
    ctx.fill();
  }

  for (const e of enemies) {
    const s = worldToScreen(e.x, e.y, camX, camY);
    ctx.save();
    ctx.translate(s.x, s.y);
    ctx.fillStyle = e.color;
    ctx.beginPath();
    ctx.arc(0, 0, e.r, 0, Math.PI * 2);
    ctx.fill();
    if (e.hurt > 0) {
      ctx.strokeStyle = 'rgba(255,255,255,0.8)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.restore();
  }

  const p = worldToScreen(player.x, player.y, camX, camY);
  ctx.save();
  ctx.translate(p.x, p.y);
  ctx.rotate(player.angle);
  ctx.fillStyle = player.invuln > 0 ? '#a6edff' : '#6fd8ff';
  ctx.beginPath();
  ctx.arc(0, 0, player.r, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(7, -2, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(8, 0);
  ctx.lineTo(28, 0);
  ctx.stroke();
  ctx.restore();

  if (gameOver) {
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.font = '700 48px system-ui, sans-serif';
    ctx.fillText('战斗失败', canvas.clientWidth / 2, canvas.clientHeight / 2 - 10);
    ctx.font = '16px system-ui, sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.fillText('按 Enter 重新开始', canvas.clientWidth / 2, canvas.clientHeight / 2 + 26);
  }

  drawMiniMap();
}

function drawMiniMap() {
  mctx.clearRect(0, 0, miniMap.width, miniMap.height);
  mctx.fillStyle = 'rgba(8, 14, 20, 0.95)';
  mctx.fillRect(0, 0, miniMap.width, miniMap.height);
  const sx = miniMap.width / world.w;
  const sy = miniMap.height / world.h;

  mctx.fillStyle = 'rgba(255,255,255,0.08)';
  for (const o of obstacles) mctx.fillRect(o.x * sx, o.y * sy, o.w * sx, o.h * sy);

  mctx.fillStyle = '#ff5b74';
  for (const e of enemies) mctx.fillRect(e.x * sx - 1, e.y * sy - 1, 3, 3);

  mctx.fillStyle = '#ffffff';
  mctx.fillRect(player.x * sx - 2, player.y * sy - 2, 4, 4);
  mctx.strokeStyle = 'rgba(255,255,255,0.25)';
  mctx.strokeRect(0.5, 0.5, miniMap.width - 1, miniMap.height - 1);
}

function loop(now) {
  const dt = Math.min(0.033, (now - last) / 1000);
  last = now;
  fpsAcc += dt;
  fpsCount += 1;
  if (fpsAcc >= 0.5) {
    fps = Math.round(fpsCount / fpsAcc);
    fpsAcc = 0;
    fpsCount = 0;
  }
  update(dt);
  draw();
  requestAnimationFrame(loop);
}

window.addEventListener('resize', resize);
window.addEventListener('keydown', (e) => {
  keys.add(e.code);
  if (e.repeat) return;
  if (['KeyQ', 'KeyE', 'KeyR', 'Space'].includes(e.code)) e.preventDefault();
  if (e.code === 'KeyR') tryReload();
  if (e.code === 'KeyQ') tryDash();
  if (e.code === 'KeyE') tryBoost();
  if (e.code === 'Space') tryRoll();
  if (e.code === 'Enter' && gameOver) resetGame();
});
window.addEventListener('keyup', (e) => keys.delete(e.code));
window.addEventListener('mousedown', (e) => {
  if (e.button === 0) mouse.down = true;
  if (e.button === 2) mouse.right = true;
});
window.addEventListener('mouseup', (e) => {
  if (e.button === 0) mouse.down = false;
  if (e.button === 2) mouse.right = false;
});
window.addEventListener('mousemove', (e) => {
  mouse.x = e.clientX;
  mouse.y = e.clientY;
});
window.addEventListener('contextmenu', (e) => e.preventDefault());

resize();
resetGame();
updateHUD();
requestAnimationFrame(loop);
