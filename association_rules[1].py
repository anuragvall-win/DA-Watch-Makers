import pandas as pd
import numpy as np
from itertools import combinations

def run_association_rules(df, min_support=0.10, min_confidence=0.60, min_lift=1.5):
    """
    Pure-pandas Apriori implementation.
    Returns a DataFrame of association rules with support, confidence, lift.
    """
    cat_cols = [
        'preferred_batch_size', 'customisation_frequency',
        'blockchain_familiarity', 'primary_pain_point',
        'lead_stage', 'contacted_supplier_last_6mo',
        'certification_level',
    ]
    ohe = pd.get_dummies(df[cat_cols], prefix_sep='=').astype(bool)
    n = len(ohe)
    cols = ohe.columns.tolist()

    # ── Frequent single items ─────────────────────────────────────────────
    item_support = {col: ohe[col].sum() / n for col in cols}
    freq_items   = {(col,): sup for col, sup in item_support.items()
                    if sup >= min_support}

    # ── Frequent pairs ────────────────────────────────────────────────────
    freq_pairs = {}
    for (a,), (b,) in combinations(freq_items.keys(), 2):
        sup = (ohe[a] & ohe[b]).sum() / n
        if sup >= min_support:
            freq_pairs[(a, b)] = sup

    # ── Generate rules from pairs ─────────────────────────────────────────
    rules = []
    for (a, b), sup_ab in freq_pairs.items():
        for antecedent, consequent in [(a, b), (b, a)]:
            sup_ant  = item_support[antecedent]
            sup_con  = item_support[consequent]
            conf     = sup_ab / sup_ant
            lift     = conf / sup_con
            leverage = sup_ab - sup_ant * sup_con
            if conf >= min_confidence and lift >= min_lift:
                rules.append({
                    'antecedent':  antecedent,
                    'consequent':  consequent,
                    'support':     round(sup_ab, 4),
                    'confidence':  round(conf,   4),
                    'lift':        round(lift,   4),
                    'leverage':    round(leverage,4),
                })

    # ── Frequent triples ──────────────────────────────────────────────────
    freq_single_cols = [k[0] for k in freq_items]
    for a, b, c in combinations(freq_single_cols, 3):
        sup_abc = (ohe[a] & ohe[b] & ohe[c]).sum() / n
        if sup_abc < min_support:
            continue
        for antecedents, consequent in [
            ((a, b), c), ((a, c), b), ((b, c), a)
        ]:
            sup_ant = (ohe[antecedents[0]] & ohe[antecedents[1]]).sum() / n
            if sup_ant == 0:
                continue
            sup_con = item_support[consequent]
            conf    = sup_abc / sup_ant
            lift    = conf / sup_con if sup_con > 0 else 0
            leverage = sup_abc - sup_ant * sup_con
            ant_str = " + ".join(antecedents)
            if conf >= min_confidence and lift >= min_lift:
                rules.append({
                    'antecedent':  ant_str,
                    'consequent':  consequent,
                    'support':     round(sup_abc, 4),
                    'confidence':  round(conf,    4),
                    'lift':        round(lift,    4),
                    'leverage':    round(leverage, 4),
                })

    rules_df = pd.DataFrame(rules).drop_duplicates(
        subset=['antecedent','consequent']
    ).sort_values('lift', ascending=False).reset_index(drop=True)

    # ── Business-friendly labels ──────────────────────────────────────────
    def pretty(s):
        return s.replace('preferred_batch_size=','Batch: ')\
                .replace('customisation_frequency=','Customisation: ')\
                .replace('blockchain_familiarity=','Blockchain: ')\
                .replace('primary_pain_point=','Pain: ')\
                .replace('lead_stage=','Stage: ')\
                .replace('contacted_supplier_last_6mo=','Contacted supplier: ')\
                .replace('certification_level=','Cert: ')\
                .replace('_',' ')

    rules_df['antecedent'] = rules_df['antecedent'].apply(pretty)
    rules_df['consequent'] = rules_df['consequent'].apply(pretty)
    rules_df['rule']       = rules_df.apply(
        lambda r: f"IF  {r['antecedent']}  →  THEN  {r['consequent']}", axis=1
    )
    rules_df['confidence_pct'] = (rules_df['confidence'] * 100).round(1)
    rules_df['support_pct']    = (rules_df['support']    * 100).round(1)

    return rules_df

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/silicon_watch_app')
    from generate_data import generate_dataset
    df = generate_dataset()
    rules = run_association_rules(df)
    print(f"Total rules found: {len(rules)}")
    print("\nTop 10 by lift:")
    print(rules[['antecedent','consequent','support_pct','confidence_pct','lift']].head(10).to_string(index=False))
