const CHAMPIONS = [
  { id: 1, name: "小刀仔", atk: 18, hp: 55, cost: 3 },
  { id: 2, name: "弓箭手", atk: 14, hp: 60, cost: 2 },
  { id: 3, name: "法师", atk: 20, hp: 45, cost: 4 },
  { id: 4, name: "骑士", atk: 22, hp: 70, cost: 5 },
  { id: 5, name: "盗贼", atk: 30, hp: 35, cost: 4 },
  { id: 6, name: "奶妈", atk: 8, hp: 95, cost: 3 },
];

const state = {
  round: 1,
  gold: 20,
  playerHP: 100,
  enemyHP: 100,
  bench: [],
  shop: [],
  enemy: [],
  log: [],
  fighting: false,
};

const MAX_BENCH = 6;

const roundEl = document.getElementById("round");
const goldEl = document.getElementById("gold");
const playerHPEl = document.getElementById("player-hp");
const enemyHPEl = document.getElementById("enemy-hp");
const shopEl = document.getElementById("shop");
const benchEl = document.getElementById("bench");
const enemyEl = document.getElementById("enemy");
const logEl = document.getElementById("battle-log");
const refreshBtn = document.getElementById("refresh-shop");
const fightBtn = document.getElementById("fight");
const resetBtn = document.getElementById("reset");
const clearLogBtn = document.getElementById("clear-log");

function randomPick() {
  const clone = [...CHAMPIONS];
  const result = [];
  for (let i = 0; i < 5; i++) {
    result.push(clone[Math.floor(Math.random() * clone.length)]);
  }
  return result;
}

function pushLog(text) {
  state.log.push(text);
  if (state.log.length > 200) state.log.shift();
  logEl.textContent = state.log.join("\n");
  logEl.scrollTop = logEl.scrollHeight;
}

function renderUnitCard(unit, extra = "") {
  return `
    <div class="card">
      <h3>${unit.name}${extra}</h3>
      <p>攻击: ${unit.atk} &nbsp;生命: ${unit.hp}</p>
      <p>费用: ${unit.cost}</p>
    </div>
  `;
}

function render() {
  roundEl.textContent = state.round;
  goldEl.textContent = state.gold;
  playerHPEl.textContent = state.playerHP;
  enemyHPEl.textContent = state.enemyHP;

  shopEl.innerHTML = state.shop
    .map((u, index) => {
      const canBuy = state.gold >= u.cost && state.bench.length < MAX_BENCH;
      return `
        <div class="card">
          ${renderUnitCard(u)}
          <button ${canBuy ? '' : 'disabled'} data-buy="${index}">${canBuy ? "购买" : "无法购买"}</button>
        </div>
      `;
    })
    .join("");

  benchEl.innerHTML = state.bench.length
    ? state.bench.map((u) => renderUnitCard(u)).join("")
    : '<div class="card">暂无棋子，前往商店购买</div>';

  const enemyCards = state.enemy.length ? state.enemy : [
    { name: "（空）", atk: "-", hp: "-", cost: "-" },
  ];
  enemyEl.innerHTML = enemyCards
    .map((u) => `<div class="card">${renderUnitCard(u)}</div>`)
    .join("");

  refreshBtn.disabled = state.gold < 2;
  fightBtn.disabled = state.fighting || state.bench.length === 0;
}

function bindEvents() {
  shopEl.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-buy]");
    if (!btn) return;
    const idx = Number(btn.dataset.buy);
    const unit = state.shop[idx];
    if (!unit || state.gold < unit.cost || state.bench.length >= MAX_BENCH) {
      return;
    }
    state.gold -= unit.cost;
    state.bench.push({ ...unit });
    pushLog(`购买了 ${unit.name}，花费 ${unit.cost} 金币`);
    renderShop();
    render();
  });

  refreshBtn.addEventListener("click", () => {
    if (state.gold < 2) return;
    state.gold -= 2;
    renderShop(true);
    pushLog("刷新商店，消耗2金币");
    render();
  });

  fightBtn.addEventListener("click", fightRound);

  resetBtn.addEventListener("click", reset);

  clearLogBtn.addEventListener("click", () => {
    state.log = [];
    logEl.textContent = "";
  });
}

function renderShop(forceRefresh = false) {
  if (forceRefresh || state.shop.length === 0) {
    state.shop = randomPick();
  }
}

function rollEnemy() {
  const unitCount = 3 + Math.floor(Math.random() * 3);
  state.enemy = [];
  for (let i = 0; i < unitCount; i++) {
    state.enemy.push({ ...CHAMPIONS[Math.floor(Math.random() * CHAMPIONS.length)] });
  }
}

function unitAttackPhase(attacksFrom, hpKey, hpValue, who) {
  const totalAtk = attacksFrom.reduce((sum, u) => sum + u.atk, 0);
  state[hpKey] -= totalAtk;
  pushLog(`${who} 阵容共发动 ${attacksFrom.length} 次攻击，造成 ${totalAtk} 伤害`);
}

function battleStep() {
  const players = [...state.bench].sort((a, b) => b.atk - a.atk);
  const enemys = [...state.enemy].sort((a, b) => b.atk - a.atk);
  const rounds = Math.min(players.length, enemys.length, 4);
  for (let i = 0; i < rounds; i++) {
    const p = players[i];
    const e = enemys[i];

    e.hp -= p.atk;
    pushLog(`${p.name} 攻击 ${e.name}，造成 ${p.atk} 伤害（${e.name}剩余${Math.max(0, e.hp)}）`);
    if (e.hp <= 0) {
      e.hp = 0;
      pushLog(`${e.name} 被消灭`);
    }

    p.hp -= e.atk;
    pushLog(`${e.name} 反击 ${p.name}，造成 ${e.atk} 伤害（${p.name}剩余${Math.max(0, p.hp)}）`);
    if (p.hp <= 0) {
      p.hp = 0;
      pushLog(`${p.name} 被消灭`);
    }

    if (e.hp <= 0 && p.hp <= 0) {
      pushLog("双方同归于尽，本回合互有损失");
    }
  }

  const playerAlive = players.filter((x) => x.hp > 0);
  const enemyAlive = enemys.filter((x) => x.hp > 0);
  unitAttackPhase(playerAlive, "enemyHP", state.enemyHP, "我方");
  unitAttackPhase(enemyAlive, "playerHP", state.playerHP, "敌方");

  state.bench = playerAlive;
  state.enemy = enemyAlive;
}

async function fightRound() {
  if (state.fighting) return;
  state.fighting = true;
  state.log = [];
  logEl.textContent = "";
  render();

  rollEnemy();
  render();

  pushLog(`第 ${state.round} 回合对战开始`);
  battleStep();

  if (state.enemyHP <= 0 || state.playerHP <= 0) {
    const winner = state.enemyHP <= 0 ? "我方" : "敌方";
    pushLog(`对战结束：${winner} 获胜！`);
    state.fighting = false;
    fightBtn.disabled = true;
    return;
  }

  state.round += 1;
  state.gold += 3;
  renderShop(true);
  pushLog(`结算阶段：恢复商店并获得3金币，进入第 ${state.round} 回合`);
  state.fighting = false;
  render();
}

function reset() {
  state.round = 1;
  state.gold = 20;
  state.playerHP = 100;
  state.enemyHP = 100;
  state.bench = [];
  state.enemy = [];
  state.log = [];
  state.fighting = false;
  renderShop(true);
  logEl.textContent = "";
  pushLog("已重置游戏");
  render();
}

function init() {
  renderShop(true);
  bindEvents();
  render();
}

init();
