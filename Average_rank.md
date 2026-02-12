# LLM Temperature Optimization & Ranking Analysis

##Average_rank calculation 

By aggregating performance metrics across a sweep of temperatures (from Greedy Decoding $T=0$ to higher stochastic settings), this script identifies the best configuration for a given model.

## Objective
Comparing LLM performance across temperatures is difficult because metrics vary in scale and direction. 
* **Precision/NDCG** measure accuracy (Higher is better).
* **Entropy** measures diversity (Higher is usually desired).
* **Gini Index** measures inequality (Lower is better).

This tool solves that problem using a **Multi-Metric Rank Aggregation** approach.

---

## Methodology: The Average Rank Algorithm

To determine the "Best Temperature," we do not rely on a single metric. Instead, we compute a normalized score based on the **Average Rank** across four key dimensions.

### 1. The Metrics
| Metric | Direction | Description |
| :--- | :--- | :--- |
| **Precision@10** | ⬆️ Maximize | Measures top-tier accuracy. |
| **NDCG@10** | ⬆️ Maximize | Measures ranking quality of the output. |
| **Entropy** | ⬆️ Maximize | Measures the diversity of the generated items. |
| **Gini** | ⬇️ Minimize | Measures the "fairness" or distribution equality. |

### 2. The Ranking Logic
For every **(Model, Style)** pair, we compare all available temperatures:

1.  **Rank Assignment:** * For **Precision, NDCG, and Entropy**, the temperature with the *highest* value gets **Rank 1**.
    * For **Gini**, the temperature with the *lowest* value gets **Rank 1**.
2.  **Aggregation:**
    We calculate the mean of these four ranks. The formula is:

    ```
    Avg Rank = (Rank_Prec + Rank_NDCG + Rank_Ent + Rank_Gini) / 4
    ```
3.  **Winner Selection:**
    The configuration with the **lowest** `avg_rank` is marked as the optimal setting.

   **Note on Metric Selection:**
    * **For Output Format Analysis:** We calculated the average rank using **NDCG@10** (Accuracy), **Gini** (Fairness), and **Entropy** (Diversity). This analysis was conducted at a fixed deterministic temperature ($T=0$).
    * **For Temperature Analysis:** We expanded the evaluation set to include **Precision@10** (i.e., NDCG, Precision, Gini, and Entropy). Since increasing temperature introduces stochasticity and potential hallucinations, incorporating a second strict accuracy metric was critical. This ensured that the selected "optimal" temperature maintains generative stability and relevance, rather than merely achieving high diversity scores at the expense of validity.

### Example Calculation (Temperature Analysis)
*If T=0.7 ranks 1st in Precision, 1st in Gini, but 4th in Entropy and 4th in NDCG:*
$$\text{Avg Rank} = \frac{1 + 1 + 4 + 4}{4} = 2.5$$

