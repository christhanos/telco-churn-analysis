


## 1. Data Cleaning & Exploratory Insights
Before training the predictive models, the initial data analysis revealed the following:
* **Data Quality:** We identified 11 brand-new customers (`tenure = 0`) with missing `TotalCharges`. These null values were filled with 0, as these customers practically haven't generated their first bill yet.
* **Class Imbalance:** The dataset was heavily imbalanced. Out of 7,043 customers, 5,174 (73%) stayed, while only 1,869 (27%) churned.
* **Contract Insight:** Customers on a "Month-to-month" contract exhibit significantly higher churn rates compared to other contract types *(See: plots/churn.png)*.

---

## 2. Model Evolution: Solving the "Accuracy Paradox"

### Phase 1: The Baseline Model
The initial Logistic Regression model (with default parameters) achieved a seemingly high **Accuracy of 82%**. However, this metric masked the algorithm's inability to handle the imbalanced data:
* **The Problem of Missed Customers (Recall = 60%):** Out of all the customers who *actually* canceled their service, the model only successfully identified 60%. The remaining 40% slipped through the cracks. In a real-world scenario, this translates directly to lost revenue, as the company would miss the opportunity to intervene with a retention offer.

### Phase 2: The Class Weighting Strategy (Final Model)
To stop this customer "leakage," we applied mathematical weights (`class_weight='balanced'`), forcing the algorithm to pay more attention to the minority class (the churners). 

**The Business Trade-off:**
* **Recall skyrocketed to 84%.** The model now successfully catches almost all customers who are at risk of leaving.
* Accuracy dropped to 75% and Precision to 52%. This means the model will generate more "false alarms" (False Positives).
* **Conclusion:** In the telecommunications industry, the cost of offering a proactive discount to someone who wasn't actually planning to leave (a False Positive) is negligible compared to the heavy cost of losing a customer to a competitor forever. Therefore, optimizing for Recall makes this the ideal model for production.

---

## 3. Final Business Insights (The "Why")
After statistically safeguarding the model (removing multicollinearity noise via VIF diagnostics), we extracted the final model weights to uncover the true drivers behind customer decisions:

** Top 3 Churn Drivers (Risk Factors):**
1. **Fiber Optic Service:** The strongest indicator of churn. This heavily suggests underlying technical issues, frequent outages, or unsatisfactory speeds with the premium fiber service that are driving customers away.
2. **Multiple Lines (No phone service):** Customers with this specific, non-standard line configuration show a higher propensity to leave.
3. **Streaming Movies:** Subscribing to movie streaming actually increases churn risk. This indicates that the company's content offering might be severely lacking compared to external competitors (e.g., Netflix), making customers feel they are overpaying.

** Top 3 Retention Drivers (Loyalty Factors):**
1. **Long-Term Contracts (1 & 2 Years):** These serve as the absolute strongest defense against competitor poaching.
2. **Tenure:** The more months a customer spends on the network, the less likely they are to churn. Loyalty compounds over time.
3. **No Internet Service:** A specific demographic segment (likely older customers relying solely on landlines) remains extremely loyal and is largely unaffected by aggressive market offers.