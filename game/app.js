(()=>{
  const canvas = document.getElementById('gameCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  const grid = { x: 120, y: 80, cols: 9, rows: 5, cw: 80, ch: 80 };
  const gx = (c) => grid.x + c * grid.cw + grid.cw / 2;
  const gy = (r) => grid.y + r * grid.ch + grid.ch / 2;
  const gardenWidth = grid.cols * grid.cw;

  // 游戏难度参数
  const ZOMBIE_SPEED_BASE = 8;
  const ZOMBIE_SPEED_PER_WAVE = 0.06;
  const ZOMBIE_HP_BASE = 120;
  const ZOMBIE_HP_PER_WAVE = 6;
  const WAVE_SPAWN_BASE = 2;
  const WAVE_SPAWN_GROWTH = 0.22;
  const WAVE_INTERVAL_BASE = 2.6;
  const WAVE_INTERVAL_DROP = 0.03;
  const WAVE_INTERVAL_MIN = 1.7;

  const plantCfg = {
    sunflower:  { key: 'sunflower',  name: '向日葵',   cost: 50,  color: '#f4bf1e', icon: '🌻', hp: 70,  cooldown: 8,  produce: 25, shootCd: 0 },
    peashooter: { key: 'peashooter', name: '豌豆射手', cost: 100, color: '#4caf50', icon: '🌱', hp: 90,  cooldown: 0.7, speed: 360, dmg: 25 },
    frost:      { key: 'frost',      name: '寒冰射手', cost: 150, color: '#4fc3f7', icon: '❄️', hp: 85,  cooldown: 1.0, speed: 330, dmg: 18, slow: 0.55, slowTime: 1.5 },
    cherry:     { key: 'cherry',     name: '樱桃炸弹', cost: 175, color: '#ff7043', icon: '🍒', hp: 40,  fuse: 0.8, damage: 230, radius: 130 },
    wallnut:    { key: 'wallnut',    name: '坚果',     cost: 50,  color: '#8b5a2b', icon: '🪵', hp: 300, cooldown: 0 },
  };

  const ui = {
    sun: document.getElementById('sun'),
    lives: document.getElementById('lives'),
    wave: document.getElementById('wave'),
    score: document.getElementById('score'),
    hint: document.getElementById('hint'),
    start: document.getElementById('btnStart'),
    pause: document.getElementById('btnPause'),
    reset: document.getElementById('btnReset'),
    shovel: document.getElementById('btnShovel'),
    plants: [...document.querySelectorAll('[data-plant]')],
  };

  let plants = [];      // {type,c,r,x,y,hp,cd,dead,fuse}
  let zombies = [];     // {x,y,row,hp,speed,baseSpeed,atk,dead,slow,slowLeft}
  let bullets = [];     // {x,y,row,speed,dmg,slow,slowTime,dead}
  let suns = [];        // {x,y,vy,r,v,life,dead}
  let explosions = [];  // {x,y,radius,maxRadius,life,dead}

  let running = false;
  let gameOver = false;
  let selected = null;     // data-plant key
  let shovelMode = false;

  let sun = 150;
  let lives = 5;
  let wave = 0;
  let score = 0;

  let lastTs = 0;
  let spawnCD = 0;
  let spawnRemain = 0;
  let waveDelay = 2.2;
  let waveInterval = 0;
  let skySunTimer = 0;
  let skySunGap = 8;

  const setHint = (txt) => { ui.hint.textContent = txt; };
  const sync = () => {
    ui.sun.textContent = Math.floor(sun);
    ui.lives.textContent = lives;
    ui.wave.textContent = wave;
    ui.score.textContent = score;
  };

  const setActive = () => {
    ui.plants.forEach((btn) => btn.classList.toggle('active', !shovelMode && btn.dataset.plant === selected));
    ui.shovel.classList.toggle('active', shovelMode);
  };

  const world = (e) => {
    const rect = canvas.getBoundingClientRect();
    const sx = W / rect.width;
    const sy = H / rect.height;
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
  };

  const inGarden = (p) => p.x >= grid.x && p.x <= grid.x + gardenWidth && p.y >= grid.y && p.y <= grid.y + grid.rows * grid.ch;
  const cellOf = (p) => ({ c: Math.floor((p.x - grid.x) / grid.cw), r: Math.floor((p.y - grid.y) / grid.ch) });
  const occupied = (c, r) => plants.find((pl) => pl.c === c && pl.r === r && !pl.dead);

  const selectPlant = (type) => {
    selected = type;
    shovelMode = false;
    setActive();
    setHint(`已选择：${plantCfg[selected]?.name || ''}`);
  };

  const gameReset = () => {
    plants = [];
    zombies = [];
    bullets = [];
    suns = [];
    explosions = [];

    running = false;
    gameOver = false;
    selected = null;
    shovelMode = false;
    sun = 150;
    lives = 5;
    wave = 0;
    score = 0;
    spawnCD = 0;
    spawnRemain = 0;
    waveDelay = 2.0;
    waveInterval = 0;
    skySunTimer = 0;
    skySunGap = 8;

    setActive();
    sync();
    setHint('准备开始：选择植物后点击草坪种植。点击“开始”后生效。');
    draw();
  };

  const spawnSkySun = () => {
    suns.push({
      x: grid.x + 50 + Math.random() * (gardenWidth - 100),
      y: grid.y + 20 + Math.random() * 100,
      vy: 12,
      r: 16,
      v: 25,
      life: 12,
      dead: false,
    });
  };

  const spawnZombie = () => {
    const r = (Math.random() * grid.rows) | 0;
    const hp = ZOMBIE_HP_BASE + wave * ZOMBIE_HP_PER_WAVE;
    const speed = ZOMBIE_SPEED_BASE + wave * ZOMBIE_SPEED_PER_WAVE;
    zombies.push({
      x: W - 30,
      y: gy(r) - 2,
      row: r,
      hp,
      baseSpeed: speed,
      speed,
      atk: 0,
      slowLeft: 0,
      dead: false,
    });
  };

  const frontPlant = (z) => {
    let target = null;
    let min = 1e9;
    for (const p of plants) {
      if ((p.row ?? p.r) !== z.row || p.dead) continue;
      const d = z.x - p.x;
      if (d > 0 && d < min) {
        min = d;
        target = p;
      }
    }
    return target;
  };

  const hasZombieInFront = (plant) => {
    const row = plant.row ?? plant.r;
    return zombies.some((z) => !z.dead && z.row === row && z.x > plant.x);
  };

  const hitTestBulletZombie = (bullet, zombie) => {
    return Math.abs(bullet.x - zombie.x) <= 20 && Math.abs(bullet.y - zombie.y) <= 20;
  };

  const explodeCherry = (plant) => {
    const cfg = plantCfg.cherry;
    const hit = [];
    explosions.push({ x: plant.x, y: plant.y, radius: 0, maxRadius: cfg.radius, life: 0.32, dead: false });

    for (const z of zombies) {
      if (z.dead) continue;
      const dx = z.x - plant.x;
      const dy = z.y - plant.y;
      if (Math.hypot(dx, dy) <= cfg.radius) {
        z.hp -= cfg.damage;
        hit.push(z);
      }
    }
    for (const z of hit) {
      score += 12;
      if (z.hp <= 0) {
        z.dead = true;
        score += 16;
      }
    }
    if (hit.length) setHint(`樱桃炸弹爆炸，击中 ${hit.length} 只僵尸！`);
  };

  function update(dt) {
    if (!running || gameOver) return;

    // 天空阳光
    skySunTimer += dt;
    if (skySunTimer >= skySunGap) {
      spawnSkySun();
      skySunTimer = 0;
      skySunGap = 6 + Math.random() * 5;
    }

    // 波次生成
    if (waveDelay > 0) {
      waveDelay -= dt;
    } else if (spawnRemain <= 0 && zombies.length === 0) {
      wave++;
      spawnRemain = 2 + wave;
      waveInterval = Math.max(1.0, 1.7 - wave * 0.07);
      spawnCD = 0;
      setHint(`第 ${wave} 波开始了，出现 ${spawnRemain} 只僵尸`);
      sync();
    } else if (spawnRemain > 0) {
      spawnCD -= dt;
      if (spawnCD <= 0) {
        spawnZombie();
        spawnRemain -= 1;
        spawnCD = waveInterval;
      }
    }

    for (const p of plants) {
      if (p.dead) continue;
      const cfg = plantCfg[p.type];
      p.cd -= dt;

      if (p.type === 'sunflower' && p.cd <= 0) {
        suns.push({ x: p.x, y: p.y, vy: 0, r: 16, v: cfg.produce, life: 999, dead: false });
        p.cd = cfg.cooldown;
      }

      if (p.type === 'peashooter' && p.cd <= 0 && hasZombieInFront(p)) {
        bullets.push({
          x: p.x + 20,
          y: p.y,
          row: p.row ?? p.r,
          speed: cfg.speed,
          dmg: cfg.dmg,
          slow: 0,
          slowTime: 0,
          dead: false,
        });
        p.cd = cfg.cooldown;
      }

      if (p.type === 'frost' && p.cd <= 0 && hasZombieInFront(p)) {
        bullets.push({
          x: p.x + 20,
          y: p.y,
          row: p.row ?? p.r,
          speed: cfg.speed,
          dmg: cfg.dmg,
          slow: cfg.slow,
          slowTime: cfg.slowTime,
          dead: false,
        });
        p.cd = cfg.cooldown;
      }

      if (p.type === 'cherry') {
        p.cd -= dt;
        if (p.cd <= 0) {
          explodeCherry(p);
          p.dead = true;
        }
      }
    }

    for (const b of bullets) {
      if (b.dead) continue;
      b.x += b.speed * dt;
      if (b.x > W + 20) {
        b.dead = true;
        continue;
      }
      for (const z of zombies) {
        if (z.dead || b.dead || z.row !== b.row) continue;
        if (z.x - b.x <= 18 && z.x - b.x >= -4) {
          z.hp -= b.dmg;
          if (b.slow > 0) {
            z.slowLeft = Math.max(z.slowLeft, b.slowTime || 0);
          }
          b.dead = true;
          score += 4;
          if (z.hp <= 0) {
            z.dead = true;
            score += 12;
          }
          break;
        }
      }
    }

    for (const z of zombies) {
      if (z.dead) continue;

      if (z.slowLeft > 0) {
        z.slowLeft = Math.max(0, z.slowLeft - dt);
      }
      const target = frontPlant(z);
      if (target) {
        if (z.x - target.x <= 24) {
          z.atk -= dt;
          if (z.atk <= 0) {
            target.hp -= 18;
            z.atk = 0.75;
          }
        }
      } else {
        z.x -= z.baseSpeed * (z.slowLeft > 0 ? 0.52 : 1) * dt;
        if (z.x < grid.x - 10) {
          lives -= 1;
          z.dead = true;
          sync();
          if (lives <= 0) {
            gameOver = true;
            running = false;
            setHint('僵尸闯入了房子，游戏结束！');
          }
        }
      }
    }

    for (const s of suns) {
      if (!s.vy) continue;
      s.y += s.vy * dt;
      s.life -= dt;
      if (s.life <= 0) s.dead = true;
    }

    for (const e of explosions) {
      e.life -= dt;
      e.radius += 240 * dt;
      if (e.life <= 0) e.dead = true;
    }

    plants = plants.filter((p) => !p.dead && p.hp > 0);
    bullets = bullets.filter((b) => !b.dead);
    zombies = zombies.filter((z) => !z.dead);
    suns = suns.filter((s) => !s.dead);
    explosions = explosions.filter((e) => !e.dead);

    sync();
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, '#72d8ff');
    g.addColorStop(1, '#8adf7b');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = '#6ed45f';
    ctx.fillRect(grid.x, grid.y, gardenWidth, grid.rows * grid.ch);

    for (let r = 0; r < grid.rows; r++) {
      for (let c = 0; c < grid.cols; c++) {
        ctx.fillStyle = ((r + c) % 2) ? '#63c456' : '#58b64a';
        ctx.fillRect(grid.x + c * grid.cw, grid.y + r * grid.ch, grid.cw, grid.ch);
      }
    }

    ctx.strokeStyle = 'rgba(0,0,0,.25)';
    for (let r = 0; r <= grid.rows; r++) {
      ctx.beginPath();
      ctx.moveTo(grid.x, grid.y + r * grid.ch);
      ctx.lineTo(grid.x + gardenWidth, grid.y + r * grid.ch);
      ctx.stroke();
    }
    for (let c = 0; c <= grid.cols; c++) {
      ctx.beginPath();
      ctx.moveTo(grid.x + c * grid.cw, grid.y);
      ctx.lineTo(grid.x + c * grid.cw, grid.y + grid.rows * grid.ch);
      ctx.stroke();
    }

    for (const s of suns) {
      ctx.fillStyle = '#f4e65d';
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#8a5d10';
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('+' + s.v, s.x, s.y + 4);
    }

    for (const p of plants) {
      const cfg = plantCfg[p.type];
      const hpRate = Math.max(0, p.hp / cfg.hp);
      ctx.fillStyle = cfg.color;
      ctx.fillRect(p.x - 28, p.y - 28, 56, 56);
      ctx.fillStyle = '#fff';
      ctx.font = '24px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(cfg.icon, p.x, p.y + 8);
      ctx.fillStyle = 'rgba(255,255,255,.82)';
      ctx.fillRect(p.x - 28, p.y + 29, 56 * hpRate, 4);
      ctx.strokeStyle = 'rgba(0,0,0,.35)';
      ctx.strokeRect(p.x - 28, p.y + 29, 56, 4);
      if (p.type === 'cherry') {
        ctx.fillStyle = '#ffeb3b';
        ctx.font = '12px Arial';
        ctx.fillText(Math.max(0, p.cd.toFixed(1)) + 's', p.x, p.y - 36);
      }
    }

    for (const b of bullets) {
      ctx.fillStyle = b.slow > 0 ? '#9ce6ff' : '#ffd65c';
      ctx.beginPath();
      ctx.arc(b.x, b.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    for (const z of zombies) {
      ctx.fillStyle = '#5f6368';
      ctx.fillRect(z.x - 16, z.y - 34, 32, 60);
      ctx.fillStyle = '#fff';
      ctx.font = '20px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('🧟', z.x, z.y + 5);
      const maxHp = 130 + wave * 18;
      const hpRate = Math.max(0, z.hp / maxHp);
      ctx.fillStyle = 'red';
      ctx.fillRect(z.x - 18, z.y - 40, 36, 4);
      ctx.fillStyle = '#9f0';
      ctx.fillRect(z.x - 18, z.y - 40, 36 * hpRate, 4);
      if (z.slowLeft > 0) {
        ctx.fillStyle = '#8ff';
        ctx.fillText('I', z.x + 14, z.y - 44);
      }
    }

    for (const e of explosions) {
      ctx.fillStyle = 'rgba(255,153,0,' + (Math.max(0, e.life / 0.32) * 0.55) + ')';
      ctx.beginPath();
      ctx.arc(e.x, e.y, Math.min(e.radius, e.maxRadius), 0, Math.PI * 2);
      ctx.fill();
    }

    if (!running && !gameOver) {
      ctx.fillStyle = 'rgba(0,0,0,.34)';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#fff';
      ctx.font = '36px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('暂停中 / 点击开始', W / 2, H / 2 - 12);
    }

    if (gameOver) {
      ctx.fillStyle = 'rgba(0,0,0,.58)';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#ff8080';
      ctx.font = '64px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('游戏结束', W / 2, H / 2 - 14);
      ctx.font = '28px Arial';
      ctx.fillText('得分 ' + score + '，点击“重置”后重玩', W / 2, H / 2 + 30);
    }
  }

  canvas.addEventListener('click', (e) => {
    const p = world(e);

    // 先尝试拾取阳光，任何状态都允许
    for (let i = suns.length - 1; i >= 0; i--) {
      if (Math.hypot(suns[i].x - p.x, suns[i].y - p.y) <= suns[i].r + 3) {
        sun += suns[i].v;
        suns.splice(i, 1);
        sync();
        setHint('拾取阳光 +' + suns[i]?.v);
        return;
      }
    }

    if (!running || gameOver || !inGarden(p)) return;
    const { c, r } = cellOf(p);
    if (c < 0 || c >= grid.cols || r < 0 || r >= grid.rows) return;

    const idxPlant = occupied(c, r);

    if (shovelMode) {
      if (!idxPlant) {
        setHint('该格子没有植物可移除');
        return;
      }
      idxPlant.dead = true;
      setHint('已铲除该植物');
      sync();
      return;
    }

    if (!selected) {
      setHint('请先选择植物或铲子');
      return;
    }

    if (idxPlant) {
      setHint('该格已有植物');
      return;
    }

    const cfg = plantCfg[selected];
    if (!cfg) {
      setHint('无效植物');
      return;
    }

    if (sun < cfg.cost) {
      setHint('阳光不足');
      return;
    }

    sun -= cfg.cost;
    plants.push({
      type: selected,
      c,
      r,
      x: gx(c),
      y: gy(r),
      hp: cfg.hp,
      cd: cfg.cooldown || 0,
      dead: false,
      ...(selected === 'cherry' ? { cd: cfg.fuse } : {}),
    });

    sync();
    setHint(`放置 ${cfg.name}`);
  });

  ui.plants.forEach((btn) => {
    btn.addEventListener('click', () => selectPlant(btn.dataset.plant));
  });

  ui.shovel.addEventListener('click', () => {
    selected = null;
    shovelMode = true;
    setActive();
    setHint('铲子模式：点击草坪可移除植物');
  });

  ui.start.addEventListener('click', () => {
    if (gameOver) gameReset();
    running = true;
    setHint('游戏开始');
  });

  ui.pause.addEventListener('click', () => {
    if (gameOver) return;
    running = false;
    setHint('已暂停');
  });

  ui.reset.addEventListener('click', gameReset);

  function tick(ts) {
    const dt = Math.min((ts - lastTs) / 1000, 0.05);
    lastTs = ts;
    update(dt);
    draw();
    requestAnimationFrame(tick);
  }

  gameReset();
  lastTs = performance.now();
  requestAnimationFrame(tick);
})();