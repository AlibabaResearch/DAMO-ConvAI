# Evaluation

This directory includes some example gold, predicted, and evaluation result files:


- `gold_example.txt`: sample gold file where each line is `gold SQL \t db_id`
- `pred_example.txt`: sample predicted file where each line is a predicted SQL
- `eval_result_example.txt`: sample evaluation result file


The evaluation script prints out all wrong predictions. The final evaluation scores can be found at the bottom of the result file. The current evaluation script doesn't consider `DISTINCT` key words. 


## Different Evaluation Metrics 

Our evaluation script contains:

- SQL Hardness Criteria
- Exact Matching without Values
- Partial Matching without Values
- Execution Accuracy with Value Selection

### SQL Hardness Criteria

To better understand the model performance on different queries, we divide SQL queries into 4 levels: easy, medium, hard, extra hard. We define the difficulty as the following:

First, we define:
- SQL components 1: WHERE, GROUP BY, ORDER BY, LIMIT, JOIN, OR, LIKE，HAVING
- SQL components 2: EXCEPT, UNION, INTERSECT, NESTED
- Others: number of agg > 1, number of select columns > 1, number of where conditions > 1, number of group by clauses > 1, number of group by clauses > 1 (no consider col1-col2 math equations etc.)

Then different hardness levels are determined as follows.

1. Easy: if SQL key words have ZERO or exact ONE from [SQL components 1] and SQL do not satisfy any conditions in [Others] above. AND no word from [SQL components 2].
2. Medium: SQL satisfies no more than two rules in [Others], and does not have more than one word from [SQL components 1], and no word from [SQL components 2]. OR, SQL has exact 2 words from [SQL components 1] and less than 2 rules in [Others], and no word from [SQL components 2]
3. Hard: SQL satisfies more than two rules in [Others], with no more than 2 key words in [SQL components 1] and no word in [SQL components 2]. OR, SQL has 2 < number key words in [SQL components 1] <= 3 and satisfies no more than two rules in [Others] but no word in [SQL components 2]. OR, SQL has no more than 1 key word in [SQL components 1] and no rule in [Others], but exact one key word in [SQL components 2].
4. Extra Hard: All others left.
5. All: all SQL-question pairs to be scored.


### Partial Matching without Values

In order to know model's performance on different SQL components, we provide the detailed scores on each part. Since models in our paper do not predict value string, our Partial and Exact Matching evaluation metrics do not take value strings into account.

For each SQL, we compute accuracy and F1 scores for all following components:

1. `SELECT COLUMN`: e.g. gold: ([select, col1, none], [select, col2, max]) and predicted: ([select, col1, none], [select, col3, min]) compute accuracy, recall, precision and F1 scores.
2. `SELECT COLUMN WITHOUT AGG`: e.g. gold: ([select, col1], [select, col2]) and predicted: ([select, col1], [select, col3]) compute accuracy, recall, precision and F1 scores.
3. `WHERE COLUMN`: ([where, col4, NOT IN, NESTED SQL], [where, col1, >=, novalue], [where, col2, =, novalue])
4. `WHERE COLUMN WITHOUT OP`:  ([where, col1], [where, col4])
5. `GROUP BY`: ([groupby, col2], [groupby, col5])
6. `GROUP BY HAVING`: ([groupby, col2, having col1, count, >=])
7. `ORDER BY`: ([orderby, col1, no agg, desc, no limit], [orderby, *, count, asc, 3])
8. `AND/OR`: ([where, col1, col2, and], [where, col3, col2, or])
9. `EXCEPT, UNION, INTERSECT, NESTED SQL`: get the `except/union/intersect/nested` part in all SQLs containing `except/union/intersect/nested`, check if predicted `except/union/intersect/nested` part equals to the gold `except/union/intersect/nested part`.
10. `SQL KEY WORDS`: for gold and predicted sql, create a set of SQL key words if they are in `[where, group by, having, desc, asc, order by, limit, except, union, intersect, not in, in, or, like]`.


### Exact Matching without Values

To avoid the order problems in each SQL component, we should NOT just compare the whole predicted SQL with the whole gold SQL.
Otherwise, for example, `SELECT col1, col2` would not be evaluated as the same as `SELECT col2, col1`. 

Thus, we first compare set matching for each component and then see if predicted SQL get all SQL parts right. If the predicted result gets all SQL parts right, then the score of Exact Matching without Values for this predicted example is 1, otherwise 0.


### Execution Accuracy

All our databases have executable SQLite files, so we can measure execution accuracy as well. However, it is also important to note that Execution Accuracy can create false positive evaluation as a predicted SQL could return the same result (for example, NULL) as the gold SQL when they are semantically different.

Due to time constraint, our current models do not predict any value in SQL conditions so that we do not provide execution accuracies. However, we encourage you to provide it in the future submissions. For value prediction, you can assume that a list of gold values for each question is given (but your model should not use values to predict columns. Otherwise, you need to notice us). Your model has to fill them into the right slots in the SQL.
