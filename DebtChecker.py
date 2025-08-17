import re
import pandas as pd
Food = ['Hollywood', 'TST', 'GUESTAURANT']
Bills = [ 'From', 'To', 'Dave', 'Instacart', 'TMOBILE', 'UNITED']

Bills += Food

df = pd.read_csv('Current.csv')

# 1) Normalize + guarantee string dtype (avoids NaN/bytes issues)
desc = df['Description'].astype(str).str.normalize('NFKC')

# 2) Build a safe regex that matches ANY snippet (case-insensitive)
#    - escape to avoid regex metacharacters blowing up the pattern
#    - add word boundaries so "To" doesn't match the 'to' in "Total" (optional; remove \b if you want loose contains)
escaped_bills = [re.escape(s) for s in Bills]
escaped_food = [re.escape(s) for s in Food]
bounded_food = [fr'\b{s}\b' for s in escaped_food]
bounded = [fr'\b{s}\b' for s in escaped_bills]  # drop the \b wrappers if you want looser matches
pattern = '(' + '|'.join(bounded) + ')'
food_pattern = '(' + '|'.join(bounded_food) + ')'

# 3) Vectorized filter
mask = desc.str.contains(pattern, case=False, na=False, regex=True)
food_mask = desc.str.contains(food_pattern, case=False, na=False, regex=True)
food_hits = df.loc[food_mask].copy()
hits = df.loc[~mask].copy()
sum = hits['Amount'].sum()
food_sum = food_hits['Amount'].sum()
# 4) Extract WHICH bill matched into a new column

print(f"Matches: {mask.sum()}")
print(hits)
print(f'Total overage this week {(abs(sum) - 50)}')
print(food_hits)
print(f'Total Food sum this week: {abs(food_sum)}')
