import json


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """

    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table):
        column_names_original = table["column_names_original"]
        table_names_original = table["table_names_original"]
        # print 'column_names_original: ', column_names_original
        # print 'table_names_original: ', table_names_original
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {"*": i}
            else:
                key = table_names_original[tab_id].lower()
                val = col.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[key] = i

        return idMap


def _get_schemas_from_json(data: dict):
    db_names = [db["db_id"] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db["db_id"]
        schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db["column_names_original"]
        table_names_original = db["table_names_original"]
        tables[db_id] = {
            "column_names_original": column_names_original,
            "table_names_original": table_names_original,
        }
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables


def get_schemas_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return _get_schemas_from_json(data)
