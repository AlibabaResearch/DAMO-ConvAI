difficuty_score_prompt_latest = """Evaluate the difficulty and complexity of each rewritten sample compared to its original. Determine if the rewritten sample has improved in quality, and provide a 'yes' or 'no' answer.:
Additionally, quantitatively score the difficulty and complexity of each rewritten sample on a scale of 1 to 10, where higher scores represent higher difficulty and complexity.
You must just give yes or no, a score for each rewritten sample, and reasons. Return with json format as follows. Note that the number of scores must correspond exactly to the number of rewritten samples.

```
[
    {{"improved": "yes/no", "score": 1-10, "reason": the reason for the improvement and score}},
    {{"improved": "yes/no", "score": 1-10, "reason": the reason for the improvement and score}},
]
```

### Evaluation Criteria:
Multidimensional Scoring:
    - Length: Longer Q&A pairs generally have more detail and thus are considered more complex.
    - Semantic Complexity: Use of more sophisticated language or concepts.
    - Visual Information: Q&As that incorporate more elements like objects, scenes, and spatial relationships.
    - Format Variations: Q&As with varied formats such as multiple choice, matching, or creative formats are considered more complex.
    - Visual Independence: Q&As that can be answered without visual information are directly considered to have no improvement and receive a score of 0.

### Example QA samples with increasing difficulty (1-10)
    1. Example: 
    Original: "What color is the sky?"
    Rewritten: "Can you tell me the color of the sky?"
    {{"score": 1, "reason": "Minimal change in length and complexity, slight rephrasing."}}

    2. Example: 
    Original: "What is the boy holding?"
    Rewritten: "What object is the boy holding in his hand?"
    {{"score": 2, "reason": "Slight increase in length, no significant semantic complexity."}}

    3. Example: 
    Original: "What is on the table?"
    Rewritten: "Can you identify the objects placed on the table in the picture?"
    {{"score": 3, "reason": "Moderate increase in length and slight increase in semantic complexity."}}

    4. Example: 
    Original: "What is the man doing?"
    Rewritten: "What activity is the man engaged in, and where is he doing it?"
    {{"score": 4, "reason": "Added detail, incorporating spatial relationship."}}

    5. Example: 
    Original: "Where is the cat?"
    Rewritten: "In the image, can you describe the location of the cat relative to the sofa and the coffee table?"
    {{"score": 5, "reason": "Notable increase in length and visual complexity, inclusion of multiple objects."}}

    6. Example: 
    Original: "What are the people doing?"
    Rewritten: "Describe the various activities that the people are engaged in around the park, mentioning any interactions between them."
    {{"score": 6, "reason": "Significant increase in length and semantic complexity, multiple visual relationships."}}

    7. Example: 
    Original: "What is in the room?"
    Rewritten: "Can you create a detailed inventory of all the items you see in the room, and describe their positions relative to each other?"
    {{"score": 7, "reason": "High increase in visual and semantic complexity, multiple objects, expanded context."}}

    8. Example: 
    Original: "Who is sitting at the table?"
    Rewritten: "Identify each person sitting at the table and describe their activities, the objects they are using, and their interactions with each other."
    {{"score": 8, "reason": "Very high increase in complexity, requiring multiple inference and comprehensive scene understanding."}}

    9. Example: 
    Original: "Where is the ball?"
    Rewritten: "From the image, infer the exact location of the ball by describing its position relative to at least three other objects and explain how it might have ended up there."
    {{"score": 9, "reason": "Exceptionally high detail, extensive use of spatial and contextual relationships, complex semantics."}}

    10. Example: 
    Original: "What is happening?"
    Rewritten: "Provide a comprehensive narrative of the entire scene depicted in the image, including the roles and actions of all visible participants, the sequence of events, and any potential underlying context or backstories."
    {{"score": 10, "reason": "Extremely high complexity exceeding basic visual analysis, requiring deep visual and contextual interpretation."}}

### Given QA samples:
{given_qa}

### Rewritten QA samples:
{rewritten_qa}

## Evaluation and Score:"""