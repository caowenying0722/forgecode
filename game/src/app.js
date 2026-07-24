(() => {
  const ROWS = 4;
  const COLS = 7;
  const SHOP_SIZE = 5;

  const REROLL_COST = {
    1: 1,
    2: 2,
    3: 2,
    4: 2,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 3
  };

  const COST_WEIGHTS = {
    1: [100, 0, 0, 0, 0, 0],
    2: [65, 35, 0, 0, 0, 0],
    3: [55, 35, 10, 0, 0, 0],
    4: [45, 38, 15, 2, 0, 0],
    5: [34, 34, 24, 6, 2, 0],
    6: [27, 37, 26, 8, 2, 0],
    7: [22, 36, 30, 10, 1.5, 0.5],
    8: [18, 30, 33, 13, 4, 2],
    9: [14, 24, 32, 20, 8, 2]
  };

  const TRAIT_BONUS = {
    2: 1.15,
    3: 1.25,
    4: 1.35,
    5: 1.45,
    6: 1.60
  };

  const MIN_XP_TO_ROUND = {
    1: 2,
    2: 4,
    3: 10,
    4: 18,
    5: 28,
    6: 40,
    7: 55,
    8: 70,
    9: 90
  };

  const heroDefs = [
    { key: 'yase', name: '亚瑟', emoji: '⚔️', role: '前排', traits: ['护卫', '先锋'], cost: 1, hp: 950, atk: 85, def: 35, speed: 6 },
    { key: 'nunu', name: '努努', emoji: '🐧', role: '辅助', traits: ['先知', '野兽'], cost: 2, hp: 1000, atk: 62, def: 28, speed: 4 },
    { key: 'yasuo', name: '亚索', emoji: '🌪️', role: '刺客', traits: ['暗影', '浪客'], cost: 3, hp: 920, atk: 110, def: 24, speed: 9 },
    { key: 'xinz', name: '新手小兵', emoji: '🛡️', role: '坦克', traits: ['重装', '护卫'], cost: 4, hp: 1380, atk: 78, def: 48, speed: 5 },
    { key: 'annie', name: '安妮', emoji: '🔥', role: '法师', traits: ['魔导师', '野兽'], cost: 2, hp: 760, atk: 98, def: 22, speed: 5 },
    { key: 'garen', name: '盖伦', emoji: '🛡️', role: '前排', traits: ['护卫', '浪客'], cost: 2, hp: 1080, atk: 88, def: 42, speed: 5 },
    { key: 'lux', name: '拉克丝', emoji: '✨', role: '法师', traits: ['魔导师', '先知'], cost: 5, hp: 860, atk: 132, def: 20, speed: 7 },
    { key: 'jinx', name: '金克斯', emoji: '🚀', role: '射手', traits: ['暗影', '浪客'], cost: 4, hp: 860, atk: 125, def: 18, speed: 6 },
    { key: 'ori', name: '奥莉安娜', emoji: '🔮', role: '法师', traits: ['圣堂', '先知'], cost: 6, hp: 770, atk: 142, def: 19, speed: 8 },
    { key: 'thres', name: '锤石', emoji: '⛓️', role: '控场', traits: ['重装', '圣堂'], cost: 3, hp: 980, atk: 96, def: 36, speed: 5 },
    { key: 'leesin', name: '李青', emoji: '👊', role: '刺客', traits: ['暗影', '刺客'], cost: 4, hp: 940, atk: 116, def: 30, speed: 8 },
    { key: 'ez', name: '伊泽瑞尔', emoji: '🏹', role: '射手', traits: ['浪客', '魔导师'], cost: 3, hp: 860, atk: 110, def: 23, speed: 7 },
    { key: 'alistar', name: '牛头', emoji: '🐂', role: '坦克', traits: ['重装', '护卫'], cost: 5, hp: 1250, atk: 90, def: 50, speed: 4 },
    { key: 'ahri', name: '阿狸', emoji: '🦊', role: '法师', traits: ['魔导师', '暗影'], cost: 3, hp: 820, atk: 104, def: 27, speed: 9 },
    { key: 'teemo', name: '提莫', emoji: '🍄', role: '射手', traits: ['野兽', '浪客'], cost: 1, hp: 760, atk: 74, def: 20, speed: 9 }
  ];

  const state = {
    round: 1,
    phase: '备战',
    gold: 50,
    level: 6,
    xp: 30,
    xpNeed: 60,
    allyHp: 100,
    allyMaxHp: 100,
    enemyHp: 100,
    enemyMaxHp: 100,
    board: new Array(28).fill(null),
    bench: new Array(9).fill(null),
    shop: new Array(5).fill(null),
    selected: null,
    shopLocked: false,
    log: [],
    fighting: false
  };

  const el = {
    roundLabel: document.getElementById('round-label'),
    goldLabel: document.getElementById('gold-label'),
    levelLabel: document.getElementById('level-label'),
    xpLabel: document.getElementById('xp-label'),
    phaseLabel: document.getElementById('phase-label'),
    allyHpFill: document.getElementById('ally-hp-fill'),
    allyHpText: document.getElementById('ally-hp-text'),
    enemyHpFill: document.getElementById('enemy-hp-fill'),
    enemyHpText: document.getElementById('enemy-hp-text'),
    board: document.getElementById('board'),
    bench: document.getElementById('bench'),
    shop: document.getElementById('shop'),
    log: document.getElementById('battle-log'),
    detailName: document.getElementById('detail-name'),
    detailExtra: document.getElementById('detail-extra'),
    btnReroll: document.getElementById('btn-reroll'),
    btnLock: document.getElementById('btn-lock'),
    btnNext: document.getElementById('btn-next'),
    btnReset: document.getElementById('btn-reset'),
    autoBattle: document.getElementById('auto-battle')
  };

  let autoTimer = null;

  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const clone = (v) => JSON.parse(JSON.stringify(v));
  const heroId = () => `h_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;

  function pickByWeights(weightMap) {
    const total = Object.values(weightMap).reduce((a, b) => a + b, 0);
    let rnd = Math.random() * total;
    const keys = Object.keys(weightMap);
    for (const k of keys) {
      rnd -= weightMap[k];
      if (rnd <= 0) return Number(k);
    }
    return Number(keys[keys.length - 1]);
  }

  function costToDef() {
    const level = Math.min(9, Math.max(1, state.level));
    const maxCost = Math.min(6, Math.ceil(level / 1.5) + 1);
    const weights = COST_WEIGHTS[level] || COST_WEIGHTS[9];
    const sum = weights.slice(0, maxCost).reduce((a, b) => a + b, 0);
    const map = {};
    for (let i = 0; i < maxCost; i += 1) {
      map[i + 1] = weights[i] || 0;
    }
    return pickByWeights(map);
  }

  function pickStar() {
    const starRateByLevel = {
      1: [0.95, 0.04, 0.01],
      2: [0.93, 0.05, 0.02],
      3: [0.90, 0.07, 0.03],
      4: [0.85, 0.12, 0.03],
      5: [0.80, 0.15, 0.05],
      6: [0.75, 0.20, 0.05],
      7: [0.70, 0.22, 0.08],
      8: [0.68, 0.24, 0.08],
      9: [0.65, 0.27, 0.08]
    };
    const rates = starRateByLevel[Math.min(9, Math.max(1, state.level))] || starRateByLevel[9];
    const rnd = Math.random();
    if (rnd < rates[0]) return 1;
    if (rnd < rates[0] + rates[1]) return 2;
    return 3;
  }

  const heroDefMap = Object.fromEntries(heroDefs.map((h) => [h.key, h]));

  function getHeroDefByKey(key) {
    return heroDefMap[key];
  }

  function baseRoundCost(level) {
    return MIN_XP_TO_ROUND[Math.min(9, Math.max(1, level))] || 90;
  }

  function newHeroFrom(def, star = 1) {
    const hp = Math.round(def.hp * (1 + (star - 1) * 0.18));
    const atk = Math.round(def.atk * (1 + (star - 1) * 0.22));
    const defVal = Math.round(def.def * (1 + (star - 1) * 0.12));

    return {
      id: heroId(),
      key: def.key,
      name: def.name,
      avatar: def.emoji,
      role: def.role,
      cost: def.cost,
      star,
      maxHp: hp,
      hp,
      atk,
      def: defVal,
      speed: def.speed,
      level: 1
    };
  }

  function traitsOnUnits(units) {
    const counter = {};
    units.forEach((u) => {
      if (!u) return;
      (u.traits || []).forEach((t) => {
        counter[t] = (counter[t] || 0) + 1;
      });
    });

    return counter;
  }

  function getTraitMultiplier(count) {
    if (count >= 6) return TRAIT_BONUS[6];
    if (count >= 5) return TRAIT_BONUS[5];
    if (count >= 4) return TRAIT_BONUS[4];
    if (count >= 3) return TRAIT_BONUS[3];
    if (count >= 2) return TRAIT_BONUS[2];
    return 1;
  }

  function cloneHeroAsBattleReady(unit) {
    const bonus = Math.max(...traitsOnUnits([unit]).map((traitCounts) => getTraitMultiplier(traitCounts)), 1);
    return {
      ...unit,
      atk: Math.round(unit.atk * bonus),
      hp: Math.round(unit.hp * bonus),
      def: Math.round(unit.def * bonus)
    };
  }

  function applyTraitBonuses(units) {
    const traitCounts = traitsOnUnits(units);
    const traitBuff = {};
    Object.keys(traitCounts).forEach((trait) => {
      traitBuff[trait] = getTraitMultiplier(traitCounts[trait]);
    });

    return units
      .filter(Boolean)
      .map((u) => {
        const buff = (u.traits || []).reduce((acc, t) => acc * (traitBuff[t] || 1), 1);
        return {
          ...u,
          atk: Math.round(u.atk * buff),
          hp: Math.round(u.hp * buff),
          def: Math.round(u.def * buff)
        };
      });
  }

  function allSlots() {
    const r = [];
    state.board.forEach((unit, i) => {
      r.push({ type: 'board', index: i, unit });
    });
    state.bench.forEach((unit, i) => {
      r.push({ type: 'bench', index: i, unit });
    });
    return r;
  }

  function tryAutoMerge() {
    const groups = {};
    for (const slot of allSlots()) {
      if (!slot.unit) continue;
      const key = `${slot.unit.key}_${slot.unit.star}`;
      (groups[key] = groups[key] || []).push(slot);
    }

    let merged = false;
    Object.entries(groups).forEach(([key, arr]) => {
      while (arr.length >= MERGE_LOG_LIMIT) {
        const [first, second, third, ...rest] = arr;
        arr.length = 0;
        rest.forEach((r) => arr.push(r));

        const [keyName, starText] = key.split('_');
        const def = getHeroDefByKey(keyName);
        const sourceName = first.unit.name;
        const nextStar = Number(starText) + 1;

        if (!def || nextStar > 3) {
          continue;
        }

        clearSlot(first.type, first.index);
        clearSlot(second.type, second.index);
        setSlot(third.type, third.index, newHeroFrom(def, nextStar));

        merged = true;
        addLog(`${sourceName} 触发进化：3个 ${sourceName}（星级${starText}）合成为 ${nextStar} 星。`);
      }
    });
    return merged;
  }

  function addLog(msg) {
    state.log.unshift(`[${new Date().toLocaleTimeString()}] ${msg}`);
    state.log = state.log.slice(0, 40);
    renderLog();
  }

  function renderLog() {
    el.log.innerHTML = '';
    state.log.forEach((line) => {
      const li = document.createElement('li');
      li.textContent = line;
      el.log.appendChild(li);
    });
  }

  function updateTop() {
    el.roundLabel.textContent = `${state.round}-1`;
    el.goldLabel.textContent = state.gold;
    el.levelLabel.textContent = state.level;
    el.xpLabel.textContent = `${state.xp} / ${state.xpNeed}`;
    el.phaseLabel.textContent = state.phase;

    const ahp = Math.max(0, Math.min(100, (state.allyHp / state.allyMaxHp) * 100));
    const ehp = Math.max(0, Math.min(100, (state.enemyHp / state.enemyMaxHp) * 100));
    el.allyHpFill.style.width = `${ahp}%`;
    el.enemyHpFill.style.width = `${ehp}%`;
    el.allyHpText.textContent = `${state.allyHp} / ${state.allyMaxHp}`;
    el.enemyHpText.textContent = `${state.enemyHp} / ${state.enemyMaxHp}`;
  }

  function unitCard(unit, ctxType, idx) {
    const root = document.createElement('div');
    root.className = 'hero-card';
    root.style.background = unit
      ? 'linear-gradient(130deg, rgba(24, 34, 52, 0.95), rgba(44, 82, 122, 0.72))'
      : 'linear-gradient(130deg, rgba(10,16,28,0.65), rgba(14,30,40,0.55))';
    root.style.cursor = unit ? 'pointer' : 'default';
    root.dataset.type = ctxType;
    root.dataset.index = idx;

    if (state.selected && state.selected.type === ctxType && state.selected.index === idx) {
      root.style.outline = '2px solid var(--yellow)';
      root.style.boxShadow = '0 0 0 2px rgba(255, 196, 68, 0.4)';
    }

    if (!unit) {
      root.innerHTML = '<div class="hero-avatar">空</div>';
      return root;
    }

    const starText = '★'.repeat(unit.star);
    root.innerHTML = `
      <div class="hero-avatar">${unit.avatar}</div>
      <div class="hero-name">${unit.name}</div>
      <div class="hero-meta">${unit.role} / ${unit.cost}费</div>
      <div class="hero-meta">ATK ${unit.atk}  HP ${unit.hp}</div>
      <div class="star" title="星级">${starText}</div>
    `;
    return root;
  }

  function renderBoard() {
    el.board.innerHTML = '';
    state.board.forEach((unit, i) => {
      const cell = document.createElement('div');
      cell.className = 'cell';
      const c = unitCard(unit, 'board', i);
      cell.appendChild(c);
      cell.onclick = () => handleBoardClick(i);
      cell.oncontextmenu = (e) => {
        e.preventDefault();
        if (unit) sellHero('board', i);
      };
      el.board.appendChild(cell);
    });
  }

  function renderBench() {
    el.bench.innerHTML = '';
    state.bench.forEach((unit, i) => {
      const c = unitCard(unit, 'bench', i);
      c.onclick = () => handleBenchClick(i);
      c.oncontextmenu = (e) => {
        e.preventDefault();
        if (unit) sellHero('bench', i);
      };
      el.bench.appendChild(c);
    });
  }

  function renderShop() {
    el.shop.innerHTML = '';
    state.shop.forEach((unit, i) => {
      const c = unitCard(unit, 'shop', i);
      if (!unit) {
        c.innerHTML += '<div class="hero-meta">空位</div>';
      } else {
        c.onclick = () => selectFromShop(i);
      }
      c.style.opacity = state.shopLocked ? '0.9' : '1';
      el.shop.appendChild(c);
    });

    el.btnLock.textContent = state.shopLocked ? '取消锁定' : '锁定商店';
  }

  function updateDetail() {
    if (!state.selected) {
      el.detailName.textContent = '点击一个单位查看详情';
      el.detailExtra.textContent = '';
      return;
    }

    const { type, index } = state.selected;
    const target = (type === 'board' ? state.board[index]
      : type === 'bench' ? state.bench[index]
      : state.shop[index]);

    if (!target) {
      el.detailName.textContent = '当前未选择单位';
      el.detailExtra.textContent = '';
      return;
    }

    el.detailName.textContent = `${target.name}（${target.role}）`;
    el.detailExtra.textContent = `星级:${target.star} | 攻击:${target.atk} | 防御:${target.def} | 速度:${target.speed} | 生命:${target.hp}/${target.maxHp}`;
  }

  function refreshUi() {
    updateTop();
    renderBoard();
    renderBench();
    renderShop();
    updateDetail();
  }

  function ensureShopFilled() {
    for (let i = 0; i < state.shop.length; i++) {
      if (!state.shop[i]) {
        const d = pick(heroDefs);
        state.shop[i] = newHeroFrom(d, Math.random() < 0.18 ? 2 : 1);
      }
    }
  }

  function refreshShop(force = false) {
    if (state.shopLocked && !force) {
      addLog('商店已锁定，暂不重新刷新。');
      return;
    }
    if (!force && state.gold < 2) {
      addLog('金币不足，无法重抽商店。');
      return;
    }

    if (!force) state.gold -= 2;

    for (let i = 0; i < state.shop.length; i++) {
      const d = pick(heroDefs);
      state.shop[i] = newHeroFrom(d, Math.random() < 0.2 ? 2 : 1);
    }
    addLog('商店已刷新。');
    refreshUi();
  }

  function selectFromShop(index) {
    const unit = state.shop[index];
    if (!unit) return;
    state.selected = { type: 'shop', index };
    el.detailName.textContent = `已选中商店英雄：${unit.name}`;
    el.detailExtra.textContent = `右击可快速卖出（当前未购买）`;
    refreshUi();
  }

  function handleBoardClick(index) {
    handleSlotAction('board', index);
  }

  function handleBenchClick(index) {
    handleSlotAction('bench', index);
  }

  function getSlot(type, index) {
    if (type === 'board') return state.board[index];
    if (type === 'bench') return state.bench[index];
    return state.shop[index];
  }

  function setSlot(type, index, hero) {
    if (type === 'board') state.board[index] = hero;
    else if (type === 'bench') state.bench[index] = hero;
    else state.shop[index] = hero;
  }

  function clearSlot(type, index) {
    setSlot(type, index, null);
  }

  function handleSlotAction(targetType, targetIndex) {
    if (state.selected) {
      const source = state.selected;
      if (source.type === 'shop' && source.index === targetIndex && targetType !== 'shop') {
        // prevent immediately consuming own slot when clicked repeatedly
      }

      if (source.type === 'shop') {
        if (targetType === 'shop') {
          state.selected = { type: targetType, index: targetIndex };
          refreshUi();
          return;
        }

        const hero = getSlot('shop', source.index);
        if (!hero) {
          state.selected = null;
          refreshUi();
          return;
        }

        const destUnit = getSlot(targetType, targetIndex);
        if (destUnit) {
          addLog('目标位置已有单位，无法直接放置。先清理后再放置。');
          return;
        }

        if (state.gold < hero.cost) {
          addLog('金币不足，购买失败。');
          return;
        }

        state.gold -= hero.cost;
        setSlot(targetType, targetIndex, newHeroFrom(hero, hero.star));
        clearSlot('shop', source.index);
        setSlot('shop', source.index, null);
        addLog(`购买了【${hero.name}】并放置至${targetType === 'board' ? '棋盘' : '后备'} ${targetIndex + 1}`);
        state.selected = null;
        refreshUi();
        return;
      }

      const sourceUnit = getSlot(source.type, source.index);
      if (!sourceUnit) {
        state.selected = null;
        refreshUi();
        return;
      }

      if (source.type === targetType && source.index === targetIndex) {
        state.selected = null;
        refreshUi();
        return;
      }

      const targetUnit = getSlot(targetType, targetIndex);
      if (!targetUnit) {
        clearSlot(source.type, source.index);
        setSlot(targetType, targetIndex, sourceUnit);
        addLog(`将 ${sourceUnit.name} 从${source.type === 'board' ? '棋盘' : '后备'}移到${targetType === 'board' ? '棋盘' : '后备'}。`);
      } else {
        clearSlot(source.type, source.index);
        clearSlot(targetType, targetIndex);
        setSlot(targetType, targetIndex, sourceUnit);
        setSlot(source.type, source.index, targetUnit);
        addLog(`交换了 ${sourceUnit.name} 与 ${targetUnit.name} 的位置。`);
      }
      state.selected = null;
      refreshUi();
      return;
    }

    const clicked = getSlot(targetType, targetIndex);
    if (!clicked) {
      return;
    }
    state.selected = { type: targetType, index: targetIndex };
    updateDetail();
    refreshUi();
  }

  function sellHero(type, index) {
    const unit = getSlot(type, index);
    if (!unit) return;
    clearSlot(type, index);
    const refund = Math.max(1, Math.ceil(unit.cost / 2));
    state.gold += refund;
    state.selected = null;
    addLog(`出售 ${unit.name} 获得 ${refund} 金币。`);
    refreshUi();
  }

  function makeEnemyTeam() {
    const count = 6 + Math.floor(Math.random() * 3) + Math.floor(state.round / 3);
    const team = [];

    for (let i = 0; i < count; i += 1) {
      const d = pick(heroDefs);
      const star = Math.random() < 0.12 ? 3 : Math.random() < 0.4 ? 2 : 1;
      const unit = newHeroFrom(d, star);
      team.push(unit);
    }
    return team;
  }

  function calcDamage(att, def) {
    const base = Math.max(8, Math.round((att.atk * (0.7 + Math.random() * 0.4)) - def.def * 0.3));
    return Math.max(1, base);
  }

  function randomTarget(list) {
    return list[Math.floor(Math.random() * list.length)];
  }

  function simulateFight() {
    if (state.fighting) return;
    if (state.gold <= 0) {
      addLog('金币枯竭但仍可继续战斗。');
    }

    const aliveBoard = state.board.filter(Boolean).length;
    if (aliveBoard === 0) {
      addLog('我方棋盘无单位，无法作战。');
      return;
    }

    const ally = state.board
      .filter((u) => u)
      .filter((u) => u.hp > 0)
      .map((u) => ({ ...u }));
    const enemy = makeEnemyTeam();

    if (enemy.length === 0) {
      addLog('敌方未布阵，直接判定胜利。');
      return;
    }

    state.fighting = true;
    state.phase = '作战';
    updateTop();
    addLog(`第${state.round}回合开始，敌方共 ${enemy.length} 人，小局开始！`);

    let turns = 0;
    const maxTurns = 160;

    while (ally.some((u) => u.hp > 0) && enemy.some((u) => u.hp > 0) && turns < maxTurns) {
      turns += 1;
      const tick = [...ally.map((u) => ({ ...u, _side: 'ally' })), ...enemy.map((u) => ({ ...u, _side: 'enemy' }))]
        .filter((u) => u.hp > 0)
        .sort((a, b) => b.speed - a.speed);

      for (const actor of tick) {
        if (actor.hp <= 0) continue;
        const source = actor._side === 'ally' ? ally : enemy;
        const targetGroup = actor._side === 'ally' ? enemy : ally;
        const targets = targetGroup.filter((u) => u.hp > 0);
        if (targets.length === 0) break;

        const target = randomTarget(targets);
        const dmg = calcDamage(actor, target);
        target.hp -= dmg;

        if (target.hp <= 0) {
          if (targetGroup === enemy) {
            addLog(`我方 ${actor.name} 斩杀敌方 ${target.name}，造成 ${dmg} 点伤害。`);
          } else {
            addLog(`敌方 ${actor.name} 击破我方 ${target.name}，造成 ${dmg} 点伤害。`);
          }
        }
      }
    }

    const allyAlive = ally.filter((u) => u.hp > 0).length;
    const enemyAlive = enemy.filter((u) => u.hp > 0).length;
    const allyDmg = enemy.filter((u) => u.hp <= 0).length;

    if (allyAlive > 0 && enemyAlive === 0) {
      const damage = allyDmg * 3 + Math.min(allyAlive, 7);
      state.enemyHp = Math.max(0, state.enemyHp - damage);
      const roundGold = 3 + Math.floor(Math.random() * 4);
      state.gold += roundGold;
      state.xp += 22;
      state.round += 1;
      levelUpCheck();
      addLog(`战斗胜利！对方扣除 ${damage} 体力，获得 ${roundGold} 金币，经验+22。`);
    } else if (allyAlive > 0 && enemyAlive > 0) {
      state.enemyHp = Math.max(0, state.enemyHp - 5);
      state.gold = Math.max(0, state.gold - 2);
      addLog('战斗超时，双方未分胜负（算平局），少量伤害与金币损失。');
    } else {
      const lose = 8 + Math.floor(allyAlive * 3);
      state.allyHp = Math.max(0, state.allyHp - lose);
      state.gold = Math.max(0, state.gold - 4);
      addLog(`战斗失败！我方损失 ${lose} 生命，扣 ${lose} ？`);
    }

    state.fighting = false;
    state.phase = '备战';

    if (state.enemyHp <= 0) {
      addLog('敌方主堡被打穿，本局胜利！场景重置。');
      state.enemyHp = state.enemyMaxHp;
    }

    if (state.allyHp <= 0) {
      addLog('我方主堡已被打穿，比赛结束；按重置可开始新局。');
      state.allyHp = state.allyMaxHp;
      state.allyLevel = 1;
    }

    refreshUi();
  }

  function levelUpCheck() {
    if (state.xp >= state.xpNeed) {
      state.xp -= state.xpNeed;
      if (state.level < 9) state.level += 1;
      state.xpNeed = Math.min(120, state.xpNeed + 10);
      addLog(`恭喜升级！当前等级：${state.level}`);
    }
  }

  function toggleLock() {
    state.shopLocked = !state.shopLocked;
    el.btnLock.classList.toggle('warning', state.shopLocked);
    addLog(state.shopLocked ? '商店已锁定。' : '商店解锁，可重新刷新。');
    refreshUi();
  }

  function seedBoard() {
    for (let i = 0; i < state.board.length; i += 1) {
      state.board[i] = null;
    }
    for (let i = 0; i < state.bench.length; i += 1) {
      state.bench[i] = null;
    }

    const initial = [1, 2, 0, 3, 5, 8, 12, 14];
    initial.forEach((idx) => {
      const d = pick(heroDefs);
      state.board[idx] = newHeroFrom(d, Math.random() < 0.25 ? 2 : 1);
    });
    for (let i = 0; i < 4; i += 1) {
      const d = pick(heroDefs);
      state.bench[i] = newHeroFrom(d, Math.random() < 0.2 ? 2 : 1);
    }
  }

  function resetAll() {
    state.round = 1;
    state.phase = '备战';
    state.gold = 50;
    state.level = 6;
    state.xp = 30;
    state.xpNeed = 60;
    state.allyHp = state.allyMaxHp;
    state.enemyHp = state.enemyMaxHp;
    state.fighting = false;
    state.selected = null;
    state.shopLocked = false;
    state.shop = new Array(5).fill(null);
    seedBoard();
    refreshShop(true);
    addLog('场景已重置。');
    refreshUi();
  }

  function bindEvents() {
    el.btnReroll.onclick = () => refreshShop(false);
    el.btnLock.onclick = toggleLock;
    el.btnNext.onclick = simulateFight;
    el.btnReset.onclick = resetAll;
    el.autoBattle.onchange = (e) => {
      if (e.target.checked) {
        autoTimer = setInterval(() => {
          if (!state.fighting) simulateFight();
        }, 1800);
        addLog('已开启自动战斗。');
      } else {
        if (autoTimer) {
          clearInterval(autoTimer);
          autoTimer = null;
        }
        addLog('已关闭自动战斗。');
      }
    };
  }

  function boot() {
    seedBoard();
    refreshShop(true);
    addLog('欢迎来到金铲铲高还原 Web 模拟界面。左侧为状态与控制区，中心棋盘与后备，右侧商店与日志。');
    bindEvents();
    refreshUi();
  }

  boot();
})();
