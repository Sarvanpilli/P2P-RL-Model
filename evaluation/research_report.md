# P2P Energy Trading: Research Report

**Date**: 2025-12-18 08:28
**Data Source**: `evaluation/evaluation_results.csv`

## 1. Executive Summary

| Metric | Value | Unit | Description |
| :--- | :--- | :--- | :--- |
| **Grid Independence** | 100.00% | % | Portion of demand met by local/P2P resources. |
| **Average Gini** | 0.245 | [0,1] | Inequality in profit distribution (Lower is better). |
| **Total Carbon Emissions** | 0.00 | kg | Total CO2 generated from grid imports. |
| **Grid Overload Freq** | 0.00% | % | Steps where line capacity was exceeded. |
| **Net Social Welfare** | 10.39 | $ | Total cumulative reward (Proxy for economic efficiency). |

## 2. Detailed Verification

### A. Energy Conservation
*   **Total Demand**: 481.14 kWh
*   **Total Imported**: 0.00 kWh
*   **Total Exported**: 10.49 kWh
*   **Net Grid Interaction**: 10.49 kWh

### B. Agent Performance
*   **Agent 0**: Profit = $4.44, Avg SoC = 34.4 kWh
*   **Agent 1**: Profit = $0.78, Avg SoC = 18.0 kWh
*   **Agent 2**: Profit = $5.42, Avg SoC = 31.3 kWh
*   **Agent 3**: Profit = $2.75, Avg SoC = 31.2 kWh
