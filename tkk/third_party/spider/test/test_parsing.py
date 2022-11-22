from process_sql import get_schema, Schema, get_sql


def test_schema():
    return Schema(get_schema('test/db.sqlite'))


def test_parse_col():
    ground_truth = (False, [(3, (0, (0, '__papers.id__', True), None))])
    assert get_sql(test_schema(),
                   'SELECT COUNT(DISTINCT(papers.id)) FROM papers')['select'] == ground_truth
    assert get_sql(test_schema(),
                   'SELECT COUNT(DISTINCT papers.id) FROM papers')['select'] == ground_truth

    ground_truth = (True, [(0, (0, (0, '__papers.id__', False), None))])
    assert get_sql(test_schema(),
                   'SELECT DISTINCT(papers.id) FROM papers')['select'] == ground_truth
    assert get_sql(test_schema(),
                   'SELECT DISTINCT papers.id FROM papers')['select'] == ground_truth

    ground_truth = (False, [(3, (0, (0, '__all__', False), None))])
    assert get_sql(test_schema(),
                   'SELECT COUNT(*) FROM (SELECT papers.id FROM papers)')['select'] == ground_truth


def test_joins():
    ground_truth = {'conds': [],
                    'table_units': [('table_unit', '__papers__'), ('table_unit', '__coauthored__')]}
    assert get_sql(test_schema(),
                   'SELECT * FROM papers JOIN coauthored')['from'] == ground_truth
    assert get_sql(test_schema(),
                   'SELECT * FROM papers INNER JOIN coauthored')['from'] == ground_truth
    assert get_sql(test_schema(),
                   'SELECT * FROM papers, coauthored')['from'] == ground_truth


def test_different_not_equal_operators():
    ground_truth = [(False, 7, (0, (0, '__papers.title__', False), None), '"bar"', None)]
    assert get_sql(test_schema(),
                   'SELECT * FROM papers WHERE papers.title <> "bar"')['where'] == ground_truth
    assert get_sql(test_schema(),
                   'SELECT * FROM papers WHERE papers.title != "bar"')['where'] == ground_truth
