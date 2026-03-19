# 🔌 P2P-RL-Model: How the AI Learns to Trade Energy
### A Complete Beginner's Guide — No Math Background Required

---

## 📖 What Is This Project? (The Big Picture)

Imagine a small neighbourhood where 4 houses each have solar panels, wind turbines, or electric vehicles (EVs). Instead of everyone just buying expensive electricity from the power company, they can **trade energy directly with each other** — like a neighbourhood electricity marketplace.

The problem is: *who decides how much to trade, at what price, and when to charge/discharge batteries?*

This is where **Reinforcement Learning (RL)** comes in. We train an AI agent — just like training a dog with treats — to make these decisions automatically, so the neighbourhood saves money, reduces carbon emissions, and shares resources fairly.

---

## 🧠 What Is Reinforcement Learning? (Plain English)

Think of RL like learning to play a video game:

| RL Concept | Video Game Analogy | Our Energy Project |
|---|---|---|
| **Agent** | The player | The AI controlling energy decisions |
| **Environment** | The game world | The neighbourhood energy grid |
| **State / Observation** | What you see on screen | Current battery level, energy prices, solar output |
| **Action** | Button press | Charge battery / Sell energy / Set price |
| **Reward** | Points scored | Money earned minus penalties |
| **Episode** | One game run | One week of energy trading |

The AI **tries random actions**, sees what reward (score) it gets, and slowly learns which actions lead to more reward. This cycle repeats **hundreds of thousands of times** until the AI becomes very good.

---

## 🏘️ Who Is Playing? The 4 Agents

There are **4 "prosumers"** (producer + consumer) in this neighbourhood. Each has a unique role:

| Agent | Type | Battery | Special Feature |
|---|---|---|---|
| **Agent 0** | Solar House 🌞 | 5 kWh battery | Has solar panels (PV) |
| **Agent 1** | Wind House 💨 | 5 kWh battery | Gets wind energy |
| **Agent 2** | EV Owner 🚗 | 62 kWh (night) / 2 kWh (day) | Car battery shrinks when car is away |
| **Agent 3** | Standard House 🏠 | 10 kWh battery | Regular household |

> 💡 **kWh = kilowatt-hour.** Think of it like "litres" for electricity. A normal lightbulb uses about 0.06 kWh per hour.

---

## 👀 What Does the AI "See"? (The Observation / State)

Before making a decision each hour, the AI looks at the following information (called the **observation vector**):

### Per-Agent Features (each agent sees these for itself):

| Feature | What it means | Example Value |
|---|---|---|
| **SoC_norm** | How full is my battery right now? | 0.75 = 75% full |
| **retail_norm** | How expensive is grid electricity? | 0.40 = moderately expensive |
| **feed_in_norm** | How much would the grid pay me to sell? | 0.20 = low buyback |
| **sin_time** | What time of day is it? (smooth version 1) | 0.87 = around 10am |
| **cos_time** | What time of day is it? (smooth version 2) | 0.50 = around 10am |
| **total_export** | How much energy is being sold right now? | 0.60 |
| **total_import** | How much is being bought from grid? | 0.10 |
| **temperature** | Current temperature (affects demand) | 0.65 |
| **wind_speed** | Wind speed (affects wind generation) | 0.40 |
| **agent_type** | What type of agent am I? (one-hot encoded) | [1, 0, 0, 0] = Solar |
| **forecast ×4** | Predicted demand + generation for next 4 hours | [0.3, 0.7, 0.4, 0.8, ...] |

### 🔢 Why Use sin/cos for Time?

This is a clever math trick. If we just said "hour = 23", the AI would think hour 23 and hour 0 (midnight) are far apart. But using sine and cosine wraps time around a circle so midnight flows smoothly into hour 0:

```
sin_time = sin(2π × hour / 24)
cos_time = cos(2π × hour / 24)
```

For hour 6 (6am): sin = 1.0, cos = 0.0 → "morning"
For hour 18 (6pm): sin = -1.0, cos = 0.0 → "evening"

---

## 🎮 What Actions Can the AI Take?

Each hour, each agent chooses **3 numbers** (this is the "action"):

| Action | Range | Meaning |
|---|---|---|
| **Battery Power (kW)** | -25 to +25 | Positive = charge battery, Negative = discharge |
| **Grid Trade (kW)** | -200 to +200 | Positive = sell to neighbours, Negative = buy from neighbours |
| **Price Bid** | 0.0 to 1.0 | What price you're willing to trade at |

> 💡 **kW = kilowatt.** Think of it like the "speed" of electricity flow. Charging at 10 kW for 1 hour stores 10 kWh.

With 4 agents each making 3 decisions, the AI outputs **12 numbers** per timestep.

---

## 🔋 How Does the Battery Work? (The Physics Math)

The battery obeys real physics. Its charge level (called **State of Charge, SoC**) changes each hour like this:

### Charging (plugging in more energy):

$$SoC_{t+1} = SoC_t + (P_{in} \times \Delta t \times \sqrt{\eta})$$

### Discharging (using stored energy):

$$SoC_{t+1} = SoC_t - \left(\frac{P_{out} \times \Delta t}{\sqrt{\eta}}\right)$$

**Breaking this down like a recipe:**
- `SoC_t` = How full the battery is right NOW (in kWh)
- `P_in` or `P_out` = Power charging in or draining out (in kW)
- `Δt` = Time duration = **1.0 hour** (each simulation step is 1 hour)
- `η (eta)` = Efficiency = **0.95** (95%). This means 5% of energy is lost as heat — just like how your phone charger gets warm!
- `√η = √0.95 ≈ 0.9747` — we split the loss equally between charging and discharging

### 🧮 Example Calculation:
Battery at 30 kWh. Charging at 10 kW for 1 hour with 95% efficiency:
```
SoC_new = 30 + (10 × 1.0 × √0.95)
SoC_new = 30 + (10 × 0.9747)
SoC_new = 30 + 9.747
SoC_new = 39.75 kWh  ✅ (Not 40, because 5% is lost to heat)
```

### Physical Safety Limits:

The battery can't charge faster than it physically allows:

$$P_{charge} = \min(P_{requested},\; P_{max\_rate},\; \frac{Capacity - SoC_t}{\Delta t \times \sqrt{\eta}})$$

| Battery Parameter | Value | Why it matters |
|---|---|---|
| **Max capacity** | 50 kWh (standard) / 62 kWh (EV) | Can't store more than this |
| **Max charge rate** | 25 kW | Like a speed limit on charging |
| **Round-trip efficiency** | 95% (η = 0.95) | Energy lost as heat |
| **Timestep** | 1.0 hour | Each decision covers 1 hour |

---

## 🏪 How Does the Energy Market Work?

The neighbourhood runs a **Double Auction** market — like eBay, but for electricity: sellers post "ask prices" and buyers post "bid prices," and they're matched.

### Step 1: Sort Orders
- **Sellers** (who want to sell energy) are sorted from **lowest price to highest**
- **Buyers** (who want to buy energy) are sorted from **highest price to lowest**

### Step 2: Find the Clearing Price (P*)

When a buyer and seller agree on a deal, the price is split in the middle:

$$P^* = \frac{P_{buyer} + P_{seller}}{2}$$

**Example:** Agent 0 wants to sell at $0.14/kWh. Agent 3 wants to buy at $0.16/kWh. They match!
```
Clearing Price = (0.14 + 0.16) / 2 = $0.15/kWh ✅
```

If nobody trades internally, the default price is:

$$P^*_{default} = \frac{P_{grid\_buy} + P_{grid\_sell}}{2} = \frac{0.20 + 0.10}{2} = \$0.15/kWh$$

### Step 3: Grid as Backstop

If the neighbourhood can't internally balance, they interact with the main grid:

| Interaction | When it happens | Price |
|---|---|---|
| **Import from Grid** | Not enough local supply | **$0.20/kWh** (retail, expensive) |
| **Export to Grid** | Too much local supply | **$0.10/kWh** (feed-in tariff, cheap) |

> 💡 This is why trading within the neighbourhood is better — you avoid paying the expensive retail price or receiving the cheap feed-in price!

### Price Competition Rules:
- If a **seller** asks for **more than $0.20/kWh** → Trade fails (buyers prefer the grid)
- If a **buyer** bids **less than $0.10/kWh** → Trade fails (sellers prefer the grid)

---

## 🏆 The Reward: How the AI Knows If It Did Well

The reward is a single number the AI receives after each hour. Higher is better! It's calculated as:

$$R_{total} = \sum_{i=1}^{4} (Profit_i - Costs_i) - Penalties$$

Let's break down every piece:

---

### 💰 1. Financial Profit

$$Profit_i = (Q_{trade} \times P_{market} \times \Delta t) - (Q_{curtailed} \times P_{market} \times \Delta t \times 1.5)$$

- `Q_trade` = How many kWh were actually traded
- `P_market` = Price they traded at
- `Q_curtailed` = Energy that was wasted (couldn't be traded or stored)
- The **×1.5 penalty** means wasting energy costs 1.5× what you could have earned — to strongly discourage waste!

---

### 🌿 2. CO₂ (Carbon) Penalty

$$C_{CO_2} = Q_{import} \times I_{CO_2} \times \lambda_{CO_2}$$

| Variable | Meaning | Value |
|---|---|---|
| `Q_import` | kWh bought from the grid | Changes each hour |
| `I_CO₂` | How "dirty" the grid is right now | Mean: **0.4 kg CO₂/kWh** |
| `λ_CO₂` | How much we care about carbon | **1.0** (full weight) |

> 💡 When the grid runs on coal, it's "dirty" (high CO₂). When it's solar/wind, it's "clean." The AI learns to buy from the grid only when it's clean.

---

### ⚡ 3. Grid Overload Penalty

If too many agents try to push energy through the wires at once, the grid gets overloaded (like too many cars on a highway):

$$C_{grid} = \frac{|Q_{agent}|}{Q_{total}} \times (Q_{total} - Q_{limit}) \times \lambda_{overload}$$

| Variable | Meaning | Value |
|---|---|---|
| `Q_limit` | Max safe flow on the grid lines | **200 kW** |
| `λ_overload` | How much we punish overloads | **50.0** (very harsh!) |

> ⚠️ This penalty is **50× stronger** than normal. This forces the AI to treat grid safety as a top priority. It's essentially a "you must never break the grid" rule.

---

### 🔧 4. Battery Wear Cost

Every time energy goes in or out of a battery, it physically degrades (like phone battery wearing out):

$$C_{batt} = E_{throughput} \times \lambda_{batt}$$

| Variable | Meaning | Value |
|---|---|---|
| `E_throughput` | Total kWh cycled through battery | Changes each step |
| `λ_batt` | Cost per kWh cycled | **$0.02/kWh** |

> 💡 This small cost teaches the AI not to charge and discharge the battery unnecessarily.

---

### 🎯 5. Battery State-of-Charge Smoothing

We want batteries to stay around 50% full — not too empty (can't meet emergencies) and not too full (can't absorb solar):

$$C_{soc} = 0.1 \times \left(\frac{SoC - 0.5 \times Capacity}{Capacity}\right)^2$$

The `²` (squared) makes this penalty grow very fast when the battery is far from 50%.

**Examples:**
```
Battery at 50% full → Penalty = 0.1 × (0)²  = 0.0  ✅ Perfect!
Battery at 80% full → Penalty = 0.1 × (0.3)² = 0.009  (mild)
Battery at 10% full → Penalty = 0.1 × (0.4)² = 0.016  (moderate)
Battery at  0% full → Penalty = 0.1 × (0.5)² = 0.025  (strong)
```

---

### ⚖️ 6. Fairness Penalty (Gini Coefficient)

This is the most socially interesting part! We don't want one agent getting super rich while others earn nothing. We use the **Gini Coefficient (G)** — the same metric used by economists to measure wealth inequality in countries:

$$Penalty_{fairness} = \lambda_{fairness} \times G(Profits) \times (1 + 0.05 \times |Q_{total\_export}|)$$

| Variable | Meaning | Value |
|---|---|---|
| `G` | Gini coefficient of profits (0 = perfectly fair, 1 = total inequality) | Calculated each step |
| `λ_fairness` | How much we care about fairness | **0.5** |

**Understanding Gini:**
```
All 4 agents earn exactly the same → G = 0.0 → No penalty ✅
One agent earns everything, others earn $0 → G = 1.0 → Big penalty ❌
```

The `(1 + 0.05 × exports)` part makes the penalty stronger during periods of high trading, when fairness matters most.

---

### 📋 Complete Reward Summary Table

| Reward Component | Weight | Direction | Purpose |
|---|---|---|---|
| **Trading Profit** | Trade × Price | ➕ Add | Earn money |
| **Curtailment Penalty** | 1.5× lost revenue | ➖ Subtract | Don't waste energy |
| **CO₂ Penalty** | λ = 1.0 | ➖ Subtract | Buy green energy |
| **Grid Overload Penalty** | λ = 50.0 ⚠️ | ➖ Subtract | Don't break the wires |
| **Battery Wear** | λ = 0.02 | ➖ Subtract | Don't destroy batteries |
| **SoC Smoothing** | 0.1 | ➖ Subtract | Keep batteries ~50% |
| **Fairness Penalty** | λ = 0.5 | ➖ Subtract | Share profits equally |
| **Peak Import Penalty** | Varies 17-21h | ➖ Subtract | Avoid peak hours |
| **Action Smoothing Penalty** | Varies | ➖ Subtract | Don't jerk actions wildly |
| **V2G Bonus (EV)** | Varies | ➕ Add | Reward smart EV use |

---

## 🧠 How Does the AI Actually Learn? (The PPO Algorithm)

The AI uses an algorithm called **PPO (Proximal Policy Optimization)**. Don't be scared by the name — here's the intuition:

### The Two Neural Networks

The AI has two "brains" — both are multilayer perceptron neural networks (fancy calculators):

```
┌─────────────────────────────────────────────┐
│  INPUT: Observation Vector (state of world) │
│  Size: 4 agents × 17 features = ~68 numbers │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌────────────────┐
│  ACTOR Brain  │    │  CRITIC Brain  │
│  (Policy Net) │    │  (Value Net)   │
│               │    │                │
│  Layer 1: 400 │    │  Layer 1: 400  │
│  Layer 2: 300 │    │  Layer 2: 300  │
│               │    │                │
│  OUTPUT:      │    │  OUTPUT:       │
│  12 numbers   │    │  1 number      │
│  (actions)    │    │  (how good is  │
└───────────────┘    │  this state?)  │
                     └────────────────┘
```

- **Actor (Policy Network):** Decides WHAT TO DO. Given the current state, it outputs the 12 action numbers.
- **Critic (Value Network):** Estimates HOW GOOD IS THIS SITUATION. It helps the Actor learn faster.

---

### The Training Loop (Step by Step)

#### Phase 1: Collect Experience

The AI runs in the simulation for **4,096 steps** (about 170 days of hourly data), making decisions and collecting:
- What state was it in?
- What action did it take?
- What reward did it get?
- What state came next?

#### Phase 2: Calculate How Good Each Action Was (GAE)

We need to know: "Was that action actually good, or did it just accidentally look good?"

We use **Generalized Advantage Estimation (GAE)**:

$$A_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \ldots$$

Where the **TD error** at each step is:

$$\delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t)$$

**Plain English:**
- `r_t` = reward we actually got right now
- `V(s_t)` = Critic's guess of how good this situation is
- `V(s_{t+1})` = Critic's guess of how good the NEXT situation is
- If `r_t + V(next)` > `V(current)` → The action was BETTER than expected → Positive advantage → Do it more!
- If `r_t + V(next)` < `V(current)` → The action was WORSE than expected → Negative advantage → Do it less!

| Parameter | Value | Meaning |
|---|---|---|
| **γ (gamma)** | 0.99 | **Discount factor.** A reward 10 steps in the future is worth 0.99^10 = 90% of its value now. The AI cares a lot about future rewards! |
| **λ (lambda)** | 0.95 | **GAE smoothing.** Balances between looking only 1 step ahead (low bias) vs. many steps ahead (low variance). 0.95 means it considers many future steps. |

---

#### Phase 3: Update the Neural Networks (PPO Clipping)

The key innovation of PPO is the **clipped loss function**. The problem: if we update the neural network too much at once, it might flip from a good strategy to a terrible one.

PPO solves this with a "trust region":

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \cdot A_t,\;\; \text{clip}(r_t(\theta),\; 1-\epsilon,\; 1+\epsilon) \cdot A_t\right)\right]$$

Where:
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

**Plain English:** The ratio `r_t` asks: "How much more (or less) likely is the NEW policy to take this action compared to the OLD policy?"

With `clip_range = 0.2`:
- If the new policy would do this action **more than 1.2×** as often → Clip it (don't overdo good actions)
- If the new policy would do this action **less than 0.8×** as often → Clip it (don't overdo avoiding bad actions)

This prevents the AI from making huge, destabilizing changes in one go.

---

#### Phase 4: Repeat for 10 Epochs

The same batch of 4,096 collected steps is used to update the neural networks **10 times** (n_epochs = 10). Each time, we randomly shuffle the data into **mini-batches of 256 samples**.

This means each collection phase produces:
```
4096 steps ÷ 256 batch size = 16 gradient updates × 10 epochs = 160 gradient updates!
```

---

### Exploration: The Entropy Bonus

How do we stop the AI from getting "stuck" always doing the same thing?

We add an **entropy bonus** to encourage exploration:

$$L_{total} = L^{CLIP} - \underbrace{c_1 \cdot L^{VF}}_{\text{critic loss}} + \underbrace{c_2 \cdot H(\pi)}_{\text{entropy bonus}}$$

Where `H(π)` is the entropy (randomness/uncertainty) of the AI's decisions.

| Parameter | Value | Meaning |
|---|---|---|
| **ent_coef** | 0.01 | How much we reward uncertainty/exploration |

A higher entropy means the AI is trying diverse actions. A lower entropy means it's confident and exploiting learned strategies. `ent_coef = 0.01` is small — just enough to prevent total rigidity without making the AI too random.

---

### The Learning Rate

The learning rate controls how big each update step is:

| Parameter | Value | Schedule |
|---|---|---|
| **Learning Rate** | **3e-4 = 0.0003** | Linearly decays to 0 over training |

Think of it like adjusting the tuning dial on a radio:
- **Large learning rate** = Big jumps → Fast but unstable
- **Small learning rate** = Tiny steps → Slow but stable
- **Decaying** = Start with bigger steps to explore, end with tiny steps to fine-tune

---

## 📊 Observation & Reward Normalization

Raw numbers in the simulation can be wildly different scales ($0.10 prices vs. 200 kW power flows). This confuses neural networks. So we normalize everything using **z-score normalization**:

$$z = \frac{x - \mu}{\sigma}$$

- `x` = the raw value
- `μ` = the running average (mean) of all past values of x
- `σ` = the standard deviation (how spread out the values are)

**Example:**
```
Raw electricity prices: [0.10, 0.15, 0.20, 0.50, 0.12, ...]
Mean (μ) = 0.214
Std Dev (σ) = 0.14
Normalized $0.20 = (0.20 - 0.214) / 0.14 = -0.10  (slightly below average)
Normalized $0.50 = (0.50 - 0.214) / 0.14 = +2.04  (well above average)
```

After normalization, values are confined to roughly the range [-3, +3] and then **clipped** to:
- **Observations:** [-10, +10] (prevents extreme inputs)
- **Rewards:** [-10, +10] (prevents extreme gradient updates)

---

## ⚡ Distribution Line Losses (Real Physics!)

The cables connecting houses aren't perfect — electricity traveling through them generates heat (just like a wire heats up when current flows). We model this with:

$$Loss_{kW} = \frac{I^2 \times R}{1000}$$

Where the current is estimated from power flow and voltage:

$$I = \frac{P_{total}}{V_{grid}}$$

| Parameter | Value |
|---|---|
| **Line Resistance** | 0.05 Ω (ohms) |
| **Grid Voltage** | 0.4 kV (400 Volts — typical UK/EU low-voltage network) |

**Example:**
```
Total power flowing = 80 kW = 80,000 W
Current = 80,000 / 400 = 200 Amps
Loss = (200)² × 0.05 / 1000 = 40,000 × 0.05 / 1000 = 2.0 kW lost as heat
```

---

## 🕰️ The EV Agent's Special Behaviour

Agent 2 (the EV owner) has a dynamic battery — the car drives away during the day!

| Time | Battery Available | Reason |
|---|---|---|
| **08:00 – 17:00** | Only **2 kWh** | Car is away. Small buffer remains at home. |
| **17:00 – 08:00** | Full **62 kWh** | Car is home and plugged in (V2G capable!) |

**V2G = Vehicle-to-Grid:** The EV can discharge its large battery to help the neighbourhood during peak hours (5pm–9pm). The AI gets a **V2G bonus reward** for doing this intelligently!

---

## 🏋️ Training Configuration Summary

| Hyperparameter | Value | Plain English |
|---|---|---|
| **Total Training Steps** | 100,000 | How many hourly decisions the AI makes during training |
| **Steps per Update (n_steps)** | 4,096 | Collect this many steps before updating networks |
| **Mini-batch Size** | 256 | Update networks on 256 samples at a time |
| **Epochs per Update (n_epochs)** | 10 | Reuse collected data 10 times per update |
| **Discount Factor (γ)** | 0.99 | Future rewards are 99% as valuable as now |
| **GAE Lambda (λ)** | 0.95 | How far into the future to consider when evaluating actions |
| **PPO Clip Range (ε)** | 0.2 | Max 20% change in policy per update |
| **Learning Rate** | 3×10⁻⁴ | Step size for network weight updates |
| **Entropy Coefficient** | 0.01 | How much randomness to encourage |
| **Policy Network** | [400, 300] neurons | Sizes of Actor's hidden layers |
| **Value Network** | [400, 300] neurons | Sizes of Critic's hidden layers |

---

## 🔄 The Complete Learning Loop — Visualized

```
┌─ ENVIRONMENT ─────────────────────────────────────────────┐
│  4 Houses, Batteries, Solar/Wind, EV, Energy Market       │
│                                                           │
│  Hour 1 → Hour 2 → ... → Hour 4096                       │
└──────────────┬──────────────────────────────┬─────────────┘
               │ State (what the AI sees)     │ Reward (score)
               ▼                             ▲
┌─ AI AGENT ──────────────────────────────────────────────┐
│                                                         │
│  ACTOR Network         → Picks actions (what to do)    │
│  [400 → 300 → 12]                                       │
│                                                         │
│  CRITIC Network        → Estimates how good state is   │
│  [400 → 300 → 1]                                        │
│                                                         │
└──────────────────────────────────┬──────────────────────┘
                                   │ Actions (agent decisions)
                                   ▼
               ┌────────────────────────────────┐
               │  PHYSICS ENGINE                │
               │  • Battery charges/discharges  │
               │  • Line losses computed        │
               │  • EV constraints applied      │
               └────────────────┬───────────────┘
                                │
               ┌────────────────▼───────────────┐
               │  DOUBLE AUCTION MARKET         │
               │  • Orders sorted by price      │
               │  • Buyers ↔ Sellers matched    │
               │  • Clearing price P* computed  │
               └────────────────┬───────────────┘
                                │
               ┌────────────────▼───────────────┐
               │  REWARD CALCULATOR             │
               │  + Trade profit                │
               │  - CO₂ penalty                 │
               │  - Overload penalty (×50!)     │
               │  - Battery wear                │
               │  - SoC deviation               │
               │  - Fairness (Gini)             │
               └────────────────────────────────┘

After 4,096 steps → PPO updates networks → Repeat until 100,000 steps done!
```

---

## 📈 How Do We Know the AI Is Learning?

We track several metrics during training using **TensorBoard** (a visual dashboard):

| Metric | What it shows |
|---|---|
| **ep_rew_mean** | Average total reward per episode — should go UP 📈 |
| **policy_gradient_loss** | How much the Actor is changing — should stabilize 📉 |
| **value_loss** | How wrong the Critic's predictions are — should go DOWN 📉 |
| **entropy_loss** | How diverse/exploratory actions are — should gradually decrease |
| **explained_variance** | How well Critic predicts rewards — should approach 1.0 |

---

## 🎓 Key Takeaways

1. **The AI learns by trial and error** — it tries thousands of decisions and learns from the reward signal (like a report card)

2. **The reward function is the most important design choice** — it encodes all our goals: profit, carbon reduction, fairness, and grid safety

3. **PPO keeps learning stable** — the clipping mechanism prevents catastrophic forgetting and wild swings in strategy

4. **Two neural networks work together** — the Critic tells the Actor if it's doing well, making learning faster and more efficient

5. **Real physics are simulated** — battery efficiency, line resistance, EV charging schedules — the AI learns about the REAL world, not just an abstract game

6. **Fairness is explicitly rewarded** — the Gini coefficient ensures no single agent monopolizes the profits while others suffer

---

*Document generated from codebase: `energy_env_robust.py`, `multi_p2p_env.py`, `mathematical_formulation.md`*
*Last updated: March 2026*
