# NL2SQL-BERT

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

Content Enhanced BERT-based Text-to-SQL Generation https://arxiv.org/abs/1910.07179

### REASONABLE: Incorporating database design rule into text-to-sql generation: 

1. We use the matching information of the table cells and question string to construct a vector where
its length is the same to the question length. This question vector mainly improves the performance
of WHERE-VALUE inference result. Because it injects the knowledge that the answer cell and its
corresponding table header are bound together. **If we locate the answer cell then we locate the answer
column which contains the answer cell.**

2. We use the matching information of all the table headers and the question string to construct a
vector where its length is the same to the table headers’ length. This header vector mainly improves the
performance of WHERE-COLUMN inference result.

# Requirements

python 3.6

records 0.5.3   

torch 1.1.0   

# Run

1, Data prepare:
Download all origin data( https://drive.google.com/file/d/1iJvsf38f16el58H4NPINQ7uzal5-V4v4 or https://download.csdn.net/download/guotong1988/13008037) and put them at `data_and_model` directory.

Then run
`data_and_model/output_entity.py`

2, Train and eval:

`train.py`

# Results on BERT-Base-Uncased without Execution-Guided-Decoding
| **Model**   | Dev <br />logical form <br />accuracy | Dev<br />execution<br/> accuracy | Test<br /> logical form<br /> accuracy | Test<br /> execution<br /> accuracy |
| ----------- | ------------------------------------- | -------------------------------- | -------------------------------------- | ----------------------------------- |
| SQLova    | 80.6                      | 86.5                  | 80.0                        | 85.5                   |
| Our Method | 84.3                      | 90.3                | 83.7                      | 89.2 |

# Data
One data look:
```
{
	"table_id": "1-1000181-1",
	"phase": 1,
	"question": "Tell me what the notes are for South Australia ",
	"question_tok": ["Tell", "me", "what", "the", "notes", "are", "for", "South", "Australia"],
	"sql": {
		"sel": 5,
		"conds": [
			[3, 0, "SOUTH AUSTRALIA"]
		],
		"agg": 0
	},
	"query": {
		"sel": 5,
		"conds": [
			[3, 0, "SOUTH AUSTRALIA"]
		],
		"agg": 0
	},
	"wvi_corenlp": [
		[7, 8]
	],
	"bertindex_knowledge": [0, 0, 0, 0, 4, 0, 0, 1, 3],
	"header_knowledge": [2, 0, 0, 2, 0, 1]
}
```
The Corresponding Table:
```
{
	"id": "1-1000181-1",
	"header": ["State/territory", "Text/background colour", "Format", "Current slogan", "Current series", "Notes"],
	"rows": [
		["Australian Capital Territory", "blue/white", "Yaa·nna", "ACT · CELEBRATION OF A CENTURY 2013", "YIL·00A", "Slogan screenprinted on plate"],
		["New South Wales", "black/yellow", "aa·nn·aa", "NEW SOUTH WALES", "BX·99·HI", "No slogan on current series"],
		["New South Wales", "black/white", "aaa·nna", "NSW", "CPX·12A", "Optional white slimline series"],
		["Northern Territory", "ochre/white", "Ca·nn·aa", "NT · OUTBACK AUSTRALIA", "CB·06·ZZ", "New series began in June 2011"],
		["Queensland", "maroon/white", "nnn·aaa", "QUEENSLAND · SUNSHINE STATE", "999·TLG", "Slogan embossed on plate"],
		["South Australia", "black/white", "Snnn·aaa", "SOUTH AUSTRALIA", "S000·AZD", "No slogan on current series"],
		["Victoria", "blue/white", "aaa·nnn", "VICTORIA - THE PLACE TO BE", "ZZZ·562", "Current series will be exhausted this year"]
	]
}
```

# Trained model

https://drive.google.com/open?id=18MBm9qzobTBgWPZlpA2EErCQtsMhlTN2

# Reference 

https://github.com/naver/sqlova
