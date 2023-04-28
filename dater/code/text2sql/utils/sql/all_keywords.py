CLAUSE_KEYWORDS = (
        "select",
        "from",
        "where",
        "group",
        "order",
        "limit",
        "intersect",
        "union",
        "except",
    )
JOIN_KEYWORDS = ("join", "on", "as")

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")

ALL_KEY_WORDS = CLAUSE_KEYWORDS + JOIN_KEYWORDS + WHERE_OPS + UNIT_OPS + AGG_OPS
