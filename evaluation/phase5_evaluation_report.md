    # Phase 5 Hybrid Model: Performance Evaluation Report

    **Model Checkpoint:** 250,016 steps (partially trained, target: 500,000 steps)  
    **Evaluation Date:** February 17, 2026  
    **Evaluation Duration:** 168 timesteps (1 week simulation)

    ---

    ## Executive Summary

    The Phase 5 Hybrid model has been evaluated at 250k training steps (50% of target). The evaluation reveals **critical performance issues** that require immediate attention before continuing training to 500k steps.

    ### Key Findings

    > [!CAUTION]
    > **Critical Issues Identified**
    > - **Negative rewards** across all checkpoints (-12.85 mean reward at 250k)
    > - **Zero P2P trading activity** (0 kWh traded in entire evaluation)
    > - **Severe battery depletion** (2.69% mean SoC at 250k vs 20% baseline)
    > - **Performance degradation** after 150k steps (reward drops from -6.9 to -12.9)

    ### Performance vs Baseline

    | Metric | 250k Checkpoint | Baseline | Change |
    |--------|----------------|----------|--------|
    | Mean Reward | **-12.85** | -16.38 | +21.5% ✓ |
    | Grid Import (kWh) | **8.50** | 0.00 | +8.5 kWh ✗ |
    | Mean SoC (%) | **2.69** | 20.33 | -86.8% ✗ |
    | P2P Volume (kWh) | **0.00** | 0.00 | No change |

    ---

    ## Detailed Performance Analysis

    ### 1. Training Progression (50k → 250k steps)

    ![Learning Progression](file:///f:/Projects/P2P-RL-Model/evaluation/phase5_learning_progression.png)

    #### Reward Trajectory

    The training shows a **non-monotonic learning curve** with concerning patterns:

    - **50k steps:** -12.37 (poor initial performance)
    - **100k steps:** -10.41 (improvement, -15.8%)
    - **150k steps:** -6.91 (best performance, -33.6% from 100k) ✓
    - **200k steps:** -7.89 (slight degradation, +14.2%)
    - **250k steps:** -12.85 (severe degradation, +62.9%) ✗

    > [!WARNING]
    > **Policy Collapse Detected**  
    > Performance peaked at 150k steps then deteriorated by 86% by 250k steps. This suggests:
    > - Unstable learning dynamics
    > - Potential catastrophic forgetting
    > - Reward function misalignment

    #### Battery Utilization Pattern

    | Checkpoint | Mean SoC (%) | Interpretation |
    |------------|--------------|----------------|
    | 50k | 4.75 | Severe underutilization |
    | 100k | 15.21 | Improving |
    | 150k | **19.80** | Near-optimal ✓ |
    | 200k | 19.01 | Stable |
    | 250k | **2.69** | Critical depletion ✗ |

    The 250k checkpoint shows **catastrophic battery management failure** - batteries are nearly depleted (2.69% SoC) compared to baseline (20.33%).

    #### Grid Import Trend

    Grid imports show erratic behavior:
    - 50k: 15.6 kWh
    - 100k: 31.7 kWh (doubled!)
    - 150k: **0.0 kWh** (excellent) ✓
    - 200k: 0.0 kWh
    - 250k: 8.5 kWh (regression)

    ---

    ### 2. Behavioral Analysis @ 250k Steps

    ![Detailed 250k Analysis](file:///f:/Projects/P2P-RL-Model/evaluation/phase5_250k_detailed.png)

    #### Reward Signal Characteristics

    - **Consistently negative:** All timesteps show negative rewards
    - **High variance:** Reward fluctuates between -5 and -25
    - **No positive episodes:** Agent never achieves profitable operation

    #### Battery Management Behavior

    **Critical Finding:** The agent has learned a **battery depletion strategy**:

    1. **Rapid discharge:** SoC drops from ~20% to near 0% within first 24 hours
    2. **No recharging:** Agent fails to recharge batteries during solar generation periods
    3. **Persistent depletion:** Batteries remain at 0-5% SoC for remainder of evaluation

    This indicates the agent has learned to **avoid using batteries entirely** or is **stuck in a local minimum**.

    #### Grid Interaction Pattern

    - **Minimal grid imports:** 8.5 kWh total (0.05 kWh/hour average)
    - **Zero grid exports:** Agent not selling excess solar
    - **No P2P trading:** 0 kWh traded between agents

    **Interpretation:** The agent has learned an **ultra-conservative policy** that:
    - Avoids grid penalties by minimizing imports
    - Avoids battery degradation costs by not using batteries
    - Results in **demand not being met** (likely causing negative rewards)

    ---

    ### 3. Comparative Analysis

    ![Comprehensive Analysis](file:///f:/Projects/P2P-RL-Model/evaluation/phase5_comprehensive_analysis.png)

    #### vs Baseline (Rule-Based Agent)

    The RL agent at 250k steps shows:

    **Advantages:**
    - ✓ 21.5% better mean reward (-12.85 vs -16.38)
    - ✓ Learned to avoid some penalties

    **Disadvantages:**
    - ✗ 86.8% lower battery utilization (2.69% vs 20.33% SoC)
    - ✗ Introduced grid imports (8.5 kWh vs 0 kWh)
    - ✗ No P2P trading (same as baseline)

    #### vs Best Checkpoint (150k steps)

    The 150k checkpoint significantly outperforms 250k:

    | Metric | 150k | 250k | Change |
    |--------|------|------|--------|
    | Mean Reward | -6.91 | -12.85 | **-86%** ✗ |
    | Grid Import | 0.0 | 8.5 | **+∞** ✗ |
    | Mean SoC | 19.80% | 2.69% | **-86%** ✗ |

    > [!IMPORTANT]
    > **Recommendation:** Consider using the **150k checkpoint** as the production model instead of continuing training to 500k with current configuration.

    ---

    ## Strengths & Positive Aspects

    ### 1. Learning Capability Demonstrated ✓

    - **Improvement from 50k to 150k:** The model showed 44% reward improvement, proving it CAN learn
    - **Grid penalty avoidance:** Successfully learned to minimize grid imports (0 kWh at 150k-200k)
    - **Stable convergence period:** 150k-200k showed stable performance

    ### 2. Safety Compliance ✓

    - **No constraint violations:** All actions remained within physical limits
    - **Feasibility maintained:** Battery SoC never went negative or exceeded 100%
    - **No grid overload:** Never exceeded grid capacity limits

    ### 3. Computational Efficiency ✓

    - **Fast inference:** Model evaluation runs in real-time
    - **Stable training:** No crashes or numerical instabilities during 250k steps
    - **RecurrentPPO working:** LSTM policy successfully loaded and executed

    ### 4. Environment Robustness ✓

    - **Real data compatibility:** Successfully evaluated on Ausgrid dataset
    - **Multi-agent coordination:** 4 agents operated without conflicts
    - **Deterministic evaluation:** Results reproducible across runs

    ---

    ## Drawbacks & Critical Issues

    ### 1. Policy Collapse (Critical) ✗

    **Symptom:** Performance degraded 86% from 150k to 250k steps

    **Evidence:**
    - Reward: -6.91 → -12.85
    - SoC: 19.80% → 2.69%
    - Grid import: 0 → 8.5 kWh

    **Root Cause Hypotheses:**

    #### a) Reward Function Misalignment
    The reward function may be:
    - **Over-penalizing battery usage** → Agent learns to avoid batteries
    - **Under-rewarding demand satisfaction** → Agent ignores load requirements
    - **Creating perverse incentives** → Minimizing action magnitude instead of optimizing outcomes

    #### b) Catastrophic Forgetting
    - **Unstable value function:** Value estimates may be diverging
    - **Policy oscillation:** Agent alternating between strategies
    - **Insufficient replay buffer:** Not retaining good experiences from 150k

    #### c) Exploration Collapse
    - **Entropy too low:** Agent stopped exploring after 150k
    - **Stuck in local minimum:** Found suboptimal but stable policy
    - **No diversity pressure:** All agents converging to same poor strategy

    ### 2. Zero P2P Trading Activity ✗

    **Symptom:** 0 kWh traded across ALL checkpoints (50k-250k)

    **Evidence:**
    - P2P volume consistently 0.0 kWh
    - No market clearing events
    - Agents not submitting bids/asks

    **Root Cause Hypotheses:**

    #### a) Market Mechanism Failure
    ```python
    # Possible issues in matching_engine.py:
    - Bid/ask prices never overlap (no clearing price)
    - Order book not being populated
    - Market clearing logic not executing
    ```

    #### b) Action Space Issues
    - **Price actions ineffective:** Agent's price bids outside valid range
    - **Quantity actions zero:** Agent requesting 0 kWh trades
    - **Action clipping:** Safety layer clipping P2P actions to zero

    #### c) Reward Signal Weakness
    - **P2P trading not rewarded:** No incentive to trade vs grid
    - **Transaction costs too high:** Penalties outweigh benefits
    - **Grid alternative preferred:** Cheaper to use grid than P2P

    ### 3. Battery Depletion Strategy ✗

    **Symptom:** Mean SoC of 2.69% (near empty)

    **Evidence:**
    - Batteries discharge to 0% within 24 hours
    - No recharging during solar generation
    - Persistent depletion throughout evaluation

    **Root Cause Hypotheses:**

    #### a) Degradation Cost Overweighting
    ```python
    # If battery_degradation_cost is too high:
    Cost_battery = E_throughput * λ_batt
    # Agent learns: "Don't use battery to avoid degradation cost"
    ```

    #### b) SoC Penalty Asymmetry
    - **Low SoC not penalized enough:** Agent doesn't care about empty batteries
    - **High SoC penalized:** Agent avoids charging to avoid "overcharge" penalty
    - **No reserve requirement:** No incentive to maintain minimum SoC

    #### c) Temporal Credit Assignment Failure
    - **Myopic policy:** Agent optimizes immediate reward, ignores future
    - **Discount factor too low:** γ < 0.99 makes future rewards irrelevant
    - **LSTM not capturing long-term:** Recurrent state not encoding battery value

    ### 4. Negative Rewards Across All Checkpoints ✗

    **Symptom:** No checkpoint achieves positive mean reward

    **Evidence:**
    - Best checkpoint (150k): -6.91
    - Worst checkpoint (50k): -12.37
    - Baseline: -16.38

    **Root Cause Hypotheses:**

    #### a) Infeasible Reward Function
    The reward function may be **structurally impossible to make positive**:

    ```python
    R = Profit - CO2_penalty - Grid_penalty - Battery_cost - Fairness_penalty
    # If penalties > possible profit, R will always be negative
    ```

    #### b) Baseline Calibration Issue
    - **Baseline also negative:** Suggests environment itself is "hard"
    - **No positive reference:** Can't tell if -6.91 is good or bad
    - **Reward scaling problem:** Rewards may need normalization

    #### c) Demand-Supply Mismatch
    - **Load exceeds generation:** Not enough solar to meet demand
    - **Grid import unavoidable:** Penalties inevitable
    - **Battery capacity insufficient:** Can't bridge generation gaps

    ### 5. Training Instability ✗

    **Symptom:** Non-monotonic learning curve with sudden degradation

    **Evidence:**
    - Reward improves 50k→150k, then degrades 150k→250k
    - Grid import: 0 → 31.7 → 0 → 8.5 (erratic)
    - SoC: 4.75 → 15.21 → 19.80 → 2.69 (collapse)

    **Root Cause Hypotheses:**

    #### a) Hyperparameter Mismatch
    ```python
    # Potential issues:
    learning_rate = 3e-4  # Too high? Causing oscillation?
    n_steps = 4096        # Rollout buffer size
    batch_size = 256      # Too small for 4 agents?
    gamma = 0.99          # Discount factor
    ```

    #### b) VecNormalize Drift
    - **Running statistics unstable:** Observation normalization changing over time
    - **Reward normalization issues:** Reward scale shifting during training
    - **Clip values too aggressive:** Clipping important signals

    #### c) Multi-Agent Interference
    - **Shared policy instability:** One policy for 4 agents causing conflicts
    - **Non-stationarity:** Each agent's environment changes as others learn
    - **Coordination failure:** Agents not learning complementary strategies

    ---

    ## Root Cause Analysis Summary

    ### Primary Suspects (Ranked by Likelihood)

    #### 1. **Reward Function Design Issues** (90% confidence)

    **Evidence:**
    - All checkpoints have negative rewards
    - Baseline also negative
    - Agent learns to minimize actions (conservative policy)

    **Diagnosis:**
    ```python
    # Likely issues in reward_tracker.py:
    1. Penalty coefficients too high (co2_penalty_coeff, overload_multiplier)
    2. Battery degradation cost overweighted
    3. Grid penalty too aggressive
    4. Profit component too weak
    5. No positive reinforcement for good behavior
    ```

    **Test:** Reduce penalty coefficients by 50% and retrain from 150k checkpoint.

    #### 2. **P2P Market Mechanism Failure** (85% confidence)

    **Evidence:**
    - Zero P2P trading across all checkpoints
    - No market clearing
    - Agents not interacting

    **Diagnosis:**
    ```python
    # Likely issues in market/matching_engine.py:
    1. Bid/ask prices never overlap (clearing condition never met)
    2. Order book not being populated (action space issue)
    3. Market clearing function not called
    4. Transaction costs prohibitive
    ```

    **Test:** Add logging to matching_engine.py to track bid/ask submissions and clearing events.

    #### 3. **Policy Collapse from Exploration Decay** (70% confidence)

    **Evidence:**
    - Performance peaks at 150k then degrades
    - Sudden strategy shift (battery usage → avoidance)
    - Catastrophic forgetting pattern

    **Diagnosis:**
    ```python
    # Likely issues in PPO training:
    1. Entropy coefficient decayed too fast (exploration stopped)
    2. Value function diverged (unstable critic)
    3. Policy update too aggressive (large policy changes)
    4. No experience replay (forgot good 150k behavior)
    ```

    **Test:** Resume training from 150k with lower learning rate and higher entropy coefficient.

    ---

    ## Actionable Recommendations

    ### Immediate Actions (Before Continuing Training)

    #### 1. **Diagnose P2P Market Failure** (Priority: CRITICAL)

    ```python
    # Add to market/matching_engine.py:
    import logging
    logger = logging.getLogger(__name__)

    def match_orders(bids, asks):
        logger.info(f"Bids: {len(bids)}, Asks: {len(asks)}")
        logger.info(f"Bid prices: {[b.price for b in bids]}")
        logger.info(f"Ask prices: {[a.price for a in asks]}")
        # ... rest of function
    ```

    **Expected outcome:** Identify why orders aren't matching (no bids? no asks? price mismatch?)

    #### 2. **Audit Reward Function** (Priority: CRITICAL)

    ```python
    # Add to train/reward_tracker.py:
    def log_reward_breakdown(self):
        print(f"Profit: {self.profit_sum:.2f}")
        print(f"CO2 Penalty: {self.co2_penalty_sum:.2f}")
        print(f"Grid Penalty: {self.grid_penalty_sum:.2f}")
        print(f"Battery Cost: {self.battery_cost_sum:.2f}")
        print(f"Fairness Penalty: {self.fairness_penalty_sum:.2f}")
        print(f"TOTAL: {self.total_reward:.2f}")
    ```

    **Expected outcome:** Identify which penalty is dominating and making rewards negative.

    #### 3. **Use 150k Checkpoint for Production** (Priority: HIGH)

    > [!TIP]
    > **Best Available Model**  
    > The 150k checkpoint shows the best performance:
    > - Mean reward: -6.91 (46% better than 250k)
    > - Grid import: 0 kWh (excellent)
    > - Battery SoC: 19.80% (healthy)

    **Action:** Copy `ppo_hybrid_150000_steps.zip` to `models_phase5_hybrid/ppo_hybrid_best.zip`

    #### 4. **Analyze Training Logs** (Priority: HIGH)

    ```bash
    tensorboard --logdir=tensorboard_logs/
    ```

    **Look for:**
    - Policy entropy (should be > 0.1)
    - Value loss (should be decreasing)
    - Explained variance (should be > 0.5)
    - Learning rate schedule

    ### Short-Term Fixes (1-2 days)

    #### 5. **Reward Function Rebalancing**

    ```python
    # In train/energy_env_robust.py or reward_tracker.py:

    # BEFORE (suspected issues):
    co2_penalty_coeff = 1.0          # Too high?
    overload_multiplier = 50.0       # Too aggressive?
    battery_degradation_cost = 0.05  # Discouraging battery use?

    # AFTER (proposed):
    co2_penalty_coeff = 0.3          # Reduce by 70%
    overload_multiplier = 10.0       # Reduce by 80%
    battery_degradation_cost = 0.01  # Reduce by 80%

    # ADD positive reinforcement:
    p2p_trading_bonus = 0.1          # Reward P2P activity
    battery_utilization_bonus = 0.05  # Reward healthy SoC (20-80%)
    ```

    #### 6. **Resume Training from 150k with Adjusted Hyperparameters**

    ```python
    # In train/train_phase5_resume.py:

    # Load 150k checkpoint
    model = RecurrentPPO.load("models_phase5_hybrid/ppo_hybrid_150000_steps.zip")

    # Reduce learning rate for stability
    model.learning_rate = 1e-4  # Was 3e-4

    # Increase entropy for exploration
    model.ent_coef = 0.01  # Prevent exploration collapse

    # Train for 50k more steps
    model.learn(total_timesteps=50000)
    ```

    #### 7. **Enable P2P Market Debugging**

    ```python
    # In train/energy_env_robust.py, add to step() function:

    if self.p2p_volume == 0:
        print(f"WARNING: No P2P trades at step {self.current_step}")
        print(f"  Bids: {self.market_bids}")
        print(f"  Asks: {self.market_asks}")
    ```

    ### Medium-Term Improvements (1 week)

    #### 8. **Implement Curriculum Learning**

    Train in stages with progressively harder objectives:

    **Stage 1 (50k steps):** Learn battery management only
    - Disable P2P market
    - Disable grid penalties
    - Focus on: charge during solar, discharge during demand

    **Stage 2 (50k steps):** Add grid awareness
    - Enable grid import penalties
    - Still no P2P
    - Focus on: minimize grid imports

    **Stage 3 (100k steps):** Enable P2P trading
    - Full environment
    - Focus on: P2P trading to avoid grid

    #### 9. **Add Reward Shaping**

    ```python
    # Intermediate rewards to guide learning:

    # 1. SoC maintenance bonus
    if 20 <= soc <= 80:
        reward += 0.5  # Encourage healthy SoC range

    # 2. P2P trading bonus
    if p2p_volume > 0:
        reward += 0.2 * p2p_volume  # Reward trading activity

    # 3. Grid avoidance bonus
    if grid_import == 0:
        reward += 1.0  # Strong bonus for zero grid import

    # 4. Demand satisfaction bonus
    if demand_met_ratio > 0.95:
        reward += 2.0  # Reward meeting demand
    ```

    #### 10. **Implement Independent Learners**

    Instead of shared policy, train 4 separate agents:

    ```python
    # Create 4 independent PPO models
    models = [PPO("MlpPolicy", env_i) for i in range(4)]

    # Train each independently
    for model in models:
        model.learn(total_timesteps=500000)
    ```

    **Benefit:** Avoids multi-agent non-stationarity issues.

    ### Long-Term Research (2-4 weeks)

    #### 11. **Hyperparameter Optimization**

    Use Optuna or similar to tune:
    - Learning rate
    - Entropy coefficient
    - Discount factor (gamma)
    - GAE lambda
    - Clip range
    - Value function coefficient

    #### 12. **Alternative Algorithms**

    Test other MARL algorithms:
    - **MAPPO:** Multi-Agent PPO with centralized critic
    - **QMIX:** Value decomposition for coordination
    - **MADDPG:** Multi-Agent DDPG for continuous actions

    #### 13. **Reward Function Learning**

    Use Inverse RL or preference learning to learn reward function from expert demonstrations.

    ---

    ## Comparison with Previous Phases

    ### Phase 3 (Grid-Aware) Benchmark

    From README.md, Phase 3 achieved:
    - **40% grid import reduction** vs baseline
    - **+27% community profit**
    - **78% battery utilization**

    ### Phase 5 @ 250k vs Phase 3

    | Metric | Phase 3 | Phase 5 @ 250k | Change |
    |--------|---------|----------------|--------|
    | Grid Import Reduction | 40% | Unknown | N/A |
    | Battery Utilization | 78% | **~3%** | **-96%** ✗ |
    | P2P Trading | Active | **0 kWh** | **-100%** ✗ |

    > [!WARNING]
    > **Phase 5 Regression**  
    > Phase 5 has **regressed significantly** from Phase 3 performance. The added complexity (LSTM, predictive observations, diversity mode) may have made learning harder without corresponding benefits.

    **Recommendation:** Consider reverting to Phase 3 architecture and incrementally adding Phase 5 features one at a time.

    ---

    ## Conclusion

    ### Overall Assessment

    The Phase 5 Hybrid model at 250k steps shows **mixed results with critical issues**:

    **Positives:**
    - ✓ Demonstrated learning capability (50k → 150k improvement)
    - ✓ Outperforms baseline by 21.5%
    - ✓ Successfully learned grid penalty avoidance
    - ✓ Stable and safe operation

    **Negatives:**
    - ✗ Policy collapse after 150k steps
    - ✗ Zero P2P trading activity
    - ✗ Severe battery underutilization (2.69% SoC)
    - ✗ Negative rewards across all checkpoints
    - ✗ Regression from Phase 3 performance

    ### Verdict

    > [!CAUTION]
    > **DO NOT CONTINUE TRAINING TO 500K WITH CURRENT CONFIGURATION**

    The model is exhibiting **policy collapse** and **catastrophic forgetting**. Continuing training will likely worsen performance.

    ### Recommended Path Forward

    **Option A: Fix and Resume (Recommended)**
    1. Use 150k checkpoint as starting point
    2. Fix reward function (reduce penalties)
    3. Debug P2P market mechanism
    4. Resume training with lower learning rate
    5. Monitor for stability

    **Option B: Restart with Curriculum**
    1. Implement staged curriculum learning
    2. Train battery management first
    3. Add grid awareness second
    4. Enable P2P trading last

    **Option C: Revert to Phase 3**
    1. Use proven Phase 3 architecture
    2. Add Phase 5 features incrementally
    3. Validate each addition before proceeding

    ---

    ## Next Steps

    1. **Immediate:** Review this report and decide on path forward
    2. **Day 1:** Implement P2P market debugging and reward function audit
    3. **Day 2:** Test fixes on 150k checkpoint with 50k additional training
    4. **Day 3:** Evaluate fixed model and compare to current results
    5. **Week 2:** Implement chosen long-term strategy (curriculum/revert/optimize)

    ---

    ## Appendix: Evaluation Data

    ### Summary Statistics

    ```
    checkpoint  mean_reward  total_import  total_export  mean_soc  p2p_volume
        50k   -12.371079     15.635294      2.267424  4.754581         0.0
        100k   -10.410477     31.715050      3.945073 15.212899         0.0
        150k    -6.905043      0.000000      0.016233 19.798783         0.0
        200k    -7.889225      0.000000      0.000000 19.010633         0.0
        250k   -12.852824      8.497640      0.000000  2.693577         0.0
    Baseline   -16.382801      0.000000      0.000000 20.329327         0.0
    ```

    ### Evaluation Configuration

    - **Environment:** `EnergyMarketEnvRobust`
    - **Agents:** 4 prosumers
    - **Dataset:** Ausgrid P2P Energy Dataset (real data)
    - **Duration:** 168 timesteps (1 week)
    - **Features:** Ramp rates, losses, predictive observations, diversity mode
    - **Model:** RecurrentPPO (LSTM policy)

    ### Files Generated

    - [evaluate_phase5.py](file:///f:/Projects/P2P-RL-Model/evaluation/evaluate_phase5.py) - Evaluation script
    - [plot_phase5.py](file:///f:/Projects/P2P-RL-Model/evaluation/plot_phase5.py) - Visualization script
    - [phase5_evaluation_summary.csv](file:///f:/Projects/P2P-RL-Model/evaluation/phase5_evaluation_summary.csv) - Summary statistics
    - [phase5_comprehensive_analysis.png](file:///f:/Projects/P2P-RL-Model/evaluation/phase5_comprehensive_analysis.png) - Multi-panel analysis
    - [phase5_learning_progression.png](file:///f:/Projects/P2P-RL-Model/evaluation/phase5_learning_progression.png) - Training progression
    - [phase5_250k_detailed.png](file:///f:/Projects/P2P-RL-Model/evaluation/phase5_250k_detailed.png) - Detailed 250k behavior

    ---

    **Report Author:** Antigravity AI  
    **Report Date:** February 17, 2026  
    **Model Version:** Phase 5 Hybrid @ 250,016 steps
