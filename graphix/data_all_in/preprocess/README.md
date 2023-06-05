## Database & Schema Linking Process

#### Database Processing

Pre-process database items and align them with question entities. The main functions appear in this folder. we refer to the approach of LGESQL for pre-processing. For more details, please check out https://github.com/rhythmcao/text2sql-lgesql

#### Syntax Processing

As introduced in the paper, we introduce high-level relations among question nodes since we capture that the nouns and their modifiers are crucial to question-answering tasks. In order to reduce the influence from long-tail distributions (e.g. negative: not, none, ...), we only present two relations: MODIFIER & ARGMENT.
