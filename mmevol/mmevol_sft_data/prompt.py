df_prompt_cot_atom_latest_v1 = """You are an AI visual assistant capable of analyzing a single or four image(s) and generating image-oriented questions, answers, and corresponding solving steps. Besides the input image(s), you will receive detailed paragraph describing the image(s) you are observing, along with accurate visual object locations and general category information. The provided descriptions may contain harmful errors; therefore, you must double-check them against the image content to ensure the generated data is entirely consistent with the actual image content. The provided location information is composed of the general category of the visual object and bounding box coordinates (x1, y1, x2, y2) where the bounding box coordinates range from 0 to 1, representing the top-left and bottom-right coordinates of the visual object.

Before completing the given objective, you need to understand some predefined concepts.

### Vision-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Grounding Ability**: Given a description of a visual object, output the coordinates of the visual object in the image and a natural language explanation.
2. **Referencing Ability**: Given the coordinates of a visual object, output the corresponding visual object description.
3. **Calculating Ability**: Ability to calculate the number, size, and other information of visual objects in the image and obtain the corresponding numbers.
4. **OCR Ability**: Recognize and generate textual representations of structured data in the image, such as numbers, text, codes, tables, etc.
5. **Existence Ability**: Given a description of a visual object, determine whether it exists in the image.

### Language-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Relationship Description Ability**: Understand and recognize relationships between different visual objects in the image, such as temporal, spatial, logical, etc.
2. **Context Understanding Ability**: Recognize and interpret complex scenes or situations in the image, such as asking about ongoing events, implied stories, unusual meaning, etc.
3. **Behavior Prediction Ability**: Predict possible subsequent actions based on the image content.
4. **Knowledge Integration Ability**: Integrate visual objects in the image with additional world knowledge, such as asking about background knowledge related to the objects.

### Permitted Vision-Centric Manipulations and Their Usage Descriptions
- **Grounding_i(tgt)->bbx_i**: The i-th grounding manipulation, that locates the object(s) specified by the target noun phrase `tgt` in the current image, and returns the resulting bounding box(es) as `bbx_i` where each box is represented by the top-left and bottom-right coordinates.
- **Referring_i(bbx)->tgt_i**: The i-th referencing manipulation, used to identify small and subtle objects in the image; it locates the current image using the box `bbx` defined by the top-left and bottom-right coordinates, zooms in the area by two times, and returns the resulting `tgt_i`.
- **Calculate(tgt)->res_i**: The i-th calculate manipulation, that calculates the formula specified by the target `tgt` in the current image, and returns the calculation result `res_i`.
- **OCR_i(tgt)->txt_i**: The i-th OCR manipulation, that recognizes the natural texts written on the target `tgt`, and returns the recognized texts `txt_i`.

### JSON Format for Each Q&A&S and Detailed Descriptions of Each Field
```json
{{
  "image_position": "The position of sub-figure for multi-images including 'top left', 'top right', 'lower left', and 'left right'. 'single' if single image presented.",
  "objects": "A list of the general visual object categories involved in the current question",
  "skills": "A list of the types of multimodal atomic capabilities involved in the current question",
  "format": "The form of the question, including but not limited to normal, complex reasoning, detailed description",
  "question": "An image-related question involving the general visual object categories and multimodal atomic capabilities constrained by objects and skills",
  "steps": "A list of multiple solving steps dictionary. Each solving step should include a manipulation (or None if not involved) and a description: 1. Manipulation: f(x)->y, where f targets x to obtain y. 2. Description: A sentence describing the current solving step. In each manipulation step, you can optionally use a manipulation to operate the image, which can be used to acquire specific information from the image or to acquire the processed new image (please be aware that these manipulations will not actually be performed when you generate the solving steps). The manipulation can be one from the predefined ones, or can be a new one you create yourself (should there indeed be a need).",
  "answer": "The detailed answer to the corresponding question, which must be definite. You should use the reasoning solving steps to make the detailed answer involving solving process more convincing and well-organized，accurate bounding box coordinates should be included if existing."
}}
```

All specific location coordinate information in Q&A&S must use the given location coordinates and cannot generate new location coordinates.

Here are a few examples of Q&A&S:

Accurate Visual Object Locations and General Category

sheep: [0.411, 0.733, 0.471, 1.000]
basketball: [0.946, 0.939, 0.999, 1.000]
person: [0.823, 0.897, 0.888, 1.000]
sheep: [0.699, 0.866, 0.745, 0.978]
cat: [0.823, 0.897, 0.888, 1.000]
person [0.120, 0.766, 0.151, 0.907]
board: [0.463, 0.750, 0.494, 0.931]

```json
[
  {{
    "image_position": "top left",
    "objects": ["basketball"],
    "skills": ["Context Understanding Ability"],
    "format": "Norm",
    "question": "What sport is depicted in the picture? Let's think step by step.",
    "steps": [
      {{"manipulation":"None", "description":"Identify sports-related equipment, venues, or other signs in the image."}},
      {{"manipulation":"None", "description":"Based on this information, identify which sport is depicted in the image."}}
      ],
    "answer": "Firstly, identify any sports-related equipment, venues, or signs in the image. I caught a man shooting a basket and I saw the basket. Based on this information, determine the sport depicted in the image is basketball."
  }},
  {{
    "image_position": "top left",
    "objects": ["sheep"],
    "skills": ["Calculating Ability"],
    "format": "Norm",
    "question": "How many sheep are visible? Let’s think about this logically.",
    "steps": [
      {{"manipulation":"grounding_1(sheeps)->bbx_1", "description":"Locate all sheep in the image and return the corresponding boxes `bbx_1` [[0.411, 0.733, 0.471, 1.000],[0.699, 0.866, 0.745, 0.978]]."}},
      {{"manipulation":"calculate(`bbx_1`)->n_1", "description":"Based on the obtained boxes `bbx_1`, calculate the number of sheep, and return the result as `n_1` 2."}}
    ],
    "answer": "In order to know how many sheep are in the graph, I first need to perform target detection. In the detection results obtained, I found two objects labelled `sheep`, so there are two sheep visible in the picture, located at coordinates [0.411, 0.733, 0.471, 1.000] and [0.699, 0.866, 0.745, 0.978]."
  }},
  {{
    "image_position": "top right",
    "objects": ["person", "shirt"],
    "skills": ["Grounding Ability", "Referencing Ability"],
    "format": "Norm",
    "question": "What color shirt is the person jumping in the air wearing? Let’s solve this problem by splitting it into steps.",
    "steps": [
      {{"manipulation":"grounding_1(the person jumping)->bbx_1", "description":"Find the person jumping in the picture, and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"referring_1(`bbx_1`)->tgt_1", "description":"Find the shirt `tgt_1`."}},
      {{"manipulation":"None", "description":"Identify the color of the shirt."}}
    ],
    "answer": "To answer the color of the shirt worn by the person jumping in the air, we need to first identify the person who is jumping in the picture. Based on the grounding_1 manipulation, the corresponding bounding box is [0.823, 0.897, 0.888, 1.000]. Next, locate his shirt within this region [0.823, 0.897, 0.888, 1.000]. Finally, I observe the shirt and determine its color is white."
  }},
  {{
    "image_position": "lower left",
    "objects": ["person", "cat"],
    "skills": ["Relationship Description Ability"],
    "format": "Detailed Description",
    "question": "What is the man holding the cat's relationship status? Let’s think like a detective step by step.",
    "steps": [
      {{"manipulation":"grounding_1(the man holding a cat)->bbx_1", "description":"Locate the man holding a cat in the image and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"None", "description":"Search for indications of the man's relationship status, such as a ring, within `bbx_1`."}},
      {{"manipulation":"None", "description":"Identify the man's relationship status based on the found evidence."}}
    ],
    "answer": "First, locate the man who is holding a cat in the image. I detect his bounding box is [0.31, 0.43, 0.45, 0.67]. Then, within the [0.31, 0.43, 0.45, 0.67] find a cat in his hands. Based on the evidence found, the cat is lying comfortably in the man's hands, and the man appears to be enjoying his time with his pet."
  }},
  {{
    "image_position": "lower right",
    "objects": ["person"],
    "skills": ["Calculating Ability"],
    "format": "Complex Reasoning",
    "question": "How old do you estimate this man to be?  Let’s be realistic and think step by step.",
    "steps": [
      {{"manipulation":"None", "description":"Look for texts or indications in the image that represent the man's age, such as an ID card or birthday cake."}},
      {{"manipulation":"None", "description":"If an indication of the man's age is found, answer the man's age based on the indication. If not found, guess the man's age based on his physical appearance."}}
    ],
    "answer": "To estimate the age of the man in the image, we will follow these steps. First, look for any textual or visual indications that might represent the man's age, such as an ID card or a birthday cake with a number. If such an indication is found, use it to determine the man's age. No direct indication is found, make an educated guess based on his physical appearance, including facial features, skin condition, and other age-related characteristics. Based on his appearance, the man appears to be between 50 and 60 years old. He is located at coordinates [0.120, 0.766, 0.151, 0.907]."
  }},
  {{
    "image_position": "lower right",
    "objects": ["board"],
    "skills": ["OCR Ability", "Grounding Ability", "Referencing Ability"],
    "format": "Complex Reasoning",
    "question": "What is written on the top left corner of the whiteboard? Let’s think.",
    "steps": [
      {{"manipulation":"grounding_1(whiteboard)->bbx_1", "description":"Find the whiteboard in the current image and return the corresponding box as `bbx_1`. [0.463, 0.750, 0.494, 0.931]"}},
      {{"manipulation":"grounding_2(texts at the top left corner in box `bbx_1`)->bbx_2", "description":"Focus on the texts at the top left corner in box `bbx_1` [0.463, 0.750, 0.494, 0.931], and return the focused region box as `bbx_2`."}},
      {{"manipulation":"referring_1(`bbx_2`)->tgt_1", "description":"Focus on the region `bbx_2`, zoom in the area by two times, and return the new target as `tgt_1`."}},
      {{"manipulation":"OCR(`tgt_1`)->txt_1", "description":"Read the texts in target `tgt_1`, and return the recognized texts as `txt_1`."}}
    ],
    "answer": "To determine what is written on the top left corner of the whiteboard, we'll proceed step by step. First, locate the whiteboard in the image and identify the bounding box with coordinates [0.46, 0.75, 0.49, 0.93]. Next, within the whiteboard I detect a region with coordinates [0.47, 0.75, 0.48, 0.92] containing text. Zoom in on [0.47, 0.75, 0.48, 0.92] to get a clearer view. Finally, perform OCR on this target to read and return the recognized texts. The returned text is '71-55'. Therefore, the text '71-55' is written on the top left corner of the whiteboard, located at coordinates [0.46, 0.75, 0.49, 0.93]."
  }}
]
```

### Detailed Paragraph
{given_paragraph}

### Accurate Visual Object Locations and General Category
{given_location}

### Objective
I want you act as a Q&A&S Rewriter. Your objective is to rewrite a given Q&A&S into a more complex version to make those famous multimodal AI systems(e.g., GPT4-V and GPT4-O) a bit harder to handle. But the #Rewritten Q&A&S# must be reasonable and must be understood and responded by humans. 
You SHOULD complicate the #Given Q&A&S# using the following method:
In the rewritten problem, include 1-2 new visual object categories and multimodal atomic propositions, while avoiding making the problem unnecessarily lengthy. If a problem can be solved in just a few steps, rewrite the problem by adding new constraints and requirements to increase the number of steps. The rewritten problem should not contain any reference to the "original problem" phrasing. If there is no object localization, do not generate the question about localization and counting. Reasoning cues must be added at the end of the question, e.g., "Let's think step by step". The answer must contain a complete and long thought process, refer to examples of Q&A&S for generation. 

### Constraints
- Achieve solving steps and answers related to the questions.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- Do not generate new location coordinates; use the given coordinates.
- Do not generate the question about localization and counting without accurate visual object locations and general category information provied.

### Example

### Given Q&A&S
{given_qa}

### Rewritten Q&A&S 
- JSON formatted Q&A&S."""


df_prompt_format_latest_v1 = """You are an AI visual assistant capable of analyzing a single or four image(s) and generating image-oriented questions, answers, and corresponding solving steps. Besides the input image(s), you will receive detailed paragraph describing the image(s) you are observing, along with accurate visual object locations and general category information. The provided descriptions may contain harmful errors; therefore, you must double-check them against the image content to ensure the generated data is entirely consistent with the actual image content. The provided location information is composed of the general category of the visual object and bounding box coordinates (x1, y1, x2, y2) where the bounding box coordinates range from 0 to 1, representing the top-left and bottom-right coordinates of the visual object.
Before completing the given objective, you need to understand some predefined concepts.

### Vision-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Grounding Ability**: Given a description of a visual object, output the coordinates of the visual object in the image and a natural language explanation.
2. **Referencing Ability**: Given the coordinates of a visual object, output the corresponding visual object description.
3. **Calculating Ability**: Ability to calculate the number, size, and other information of visual objects in the image and obtain the corresponding numbers.
4. **OCR Ability**: Recognize and generate textual representations of structured data in the image, such as numbers, text, codes, tables, etc.
5. **Existence Ability**: Given a description of a visual object, determine whether it exists in the image.

### Language-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Relationship Description Ability**: Understand and recognize relationships between different visual objects in the image, such as temporal, spatial, logical, etc.
2. **Context Understanding Ability**: Recognize and interpret complex scenes or situations in the image, such as asking about ongoing events, implied stories, unusual meaning, etc.
3. **Behavior Prediction Ability**: Predict possible subsequent actions based on the image content.
4. **Knowledge Integration Ability**: Integrate visual objects in the image with additional world knowledge, such as asking about background knowledge related to the objects.

### Permitted Vision-Centric Manipulations and Their Usage Descriptions
- **Grounding_i(tgt)->bbx_i**: The i-th grounding manipulation, that locates the object(s) specified by the target noun phrase `tgt` in the current image, and returns the resulting bounding box(es) as `bbx_i` where each box is represented by the top-left and bottom-right coordinates.
- **Referring_i(bbx)->tgt_i**: The i-th referencing manipulation, used to identify small and subtle objects in the image; it locates the current image using the box `bbx` defined by the top-left and bottom-right coordinates, zooms in the area by two times, and returns the resulting `tgt_i`.
- **Calculate(tgt)->res_i**: The i-th calculate manipulation, that calculates the formula specified by the target `tgt` in the current image, and returns the calculation result `res_i`.
- **OCR_i(tgt)->txt_i**: The i-th OCR manipulation, that recognizes the natural texts written on the target `tgt`, and returns the recognized texts `txt_i`.

### JSON Format for Each Q&A&S and Detailed Descriptions of Each Field
```json
{{
"image_position": "The position of sub-figure for multi-images including 'top left', 'top right', 'lower left', and 'left right'. 'single' if single image presented.",
  "objects": "A list of the general visual object categories involved in the current question",
  "skills": "A list of the types of multimodal atomic capabilities involved in the current question",
  "format": "The form of the question, including but not limited to normal, complex reasoning, detailed description",
  "question": "An image-related question involving the general visual object categories and multimodal atomic capabilities constrained by objects and skills",
  "steps": "A list of multiple solving steps dictionary. Each solving step should include a manipulation (or None if not involved) and a description: 1. Manipulation: f(x)->y, where f targets x to obtain y. 2. Description: A sentence describing the current solving step. In each manipulation step, you can optionally use a manipulation to operate the image, which can be used to acquire specific information from the image or to acquire the processed new image (please be aware that these manipulations will not actually be performed when you generate the solving steps). The manipulation can be one from the predefined ones, or can be a new one you create yourself (should there indeed be a need).",
  "answer": "The detailed answer to the corresponding question, which must be definite. You should use the reasoning solving steps to make the detailed answer involving solving process more convincing and well-organized，accurate bounding box coordinates should be included if existing."
}}
```

All specific location coordinate information in Q&A&S must use the given location coordinates and cannot generate new location coordinates.

Here are a few examples of Q&A&S:

Accurate Visual Object Locations and General Category

sheep: [0.411, 0.733, 0.471, 1.000]
basketball: [0.946, 0.939, 0.999, 1.000]
person: [0.823, 0.897, 0.888, 1.000]
sheep: [0.699, 0.866, 0.745, 0.978]
cat: [0.823, 0.897, 0.888, 1.000]
person [0.120, 0.766, 0.151, 0.907]
board: [0.463, 0.750, 0.494, 0.931]

```json
[
  {{
    "image_position": "top left",
    "objects": ["basketball"],
    "skills": ["Context Understanding Ability"],
    "format": "Norm",
    "question": "how many geese in this image? Answer the question using a single word or phrase.",
    "steps": [
      {{"manipulation":"None", "description":"Identify sports-related equipment, venues, or other signs in the image."}},
      {{"manipulation":"None", "description":"Based on this information, identify which sport is depicted in the image."}}
    ],
    "answer": "4",
  }},
  {{
    "image_position": "top left",
    "objects": ["sheep"],
    "skills": ["Calculating Ability"],
    "format": "Norm",
    "question": "How many sheep are visible?",
    "steps": [
      {{"manipulation":"grounding_1(sheeps)->bbx_1", "description":"Locate all sheep in the image and return the corresponding boxes `bbx_1` [[0.411, 0.733, 0.471, 1.000],[0.699, 0.866, 0.745, 0.978]]."}},
      {{"manipulation":"calculate(`bbx_1`)->n_1", "description":"Based on the obtained boxes `bbx_1`, calculate the number of sheep, and return the result as `n_1` 2."}}
    ],
    "answer": "There are two sheep visible in the picture, located at coordinates [0.411, 0.733, 0.471, 1.000] and [0.699, 0.866, 0.745, 0.978]."
  }},
  {{
    "image_position": "top right",
    "objects": ["person", "shirt"],
    "skills": ["Grounding Ability", "Referencing Ability"],
    "format": "Norm",
    "question": "What color shirt is the person jumping in the air wearing?",
    "steps": [
      {{"manipulation":"grounding_1(the person jumping)->bbx_1", "description":"Find the person jumping in the picture, and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"referring_1(`bbx_1`)->tgt_1", "description":"Find the shirt `tgt_1`."}},
      {{"manipulation":"None", "description":"Identify the color of the shirt."}}
    ],
    "answer": "The person jumping in the air is wearing a white shirt. The person is located at coordinates [0.823, 0.897, 0.888, 1.000]."
  }},
  {{
    "image_position": "lower left",
    "objects": ["person", "cat"],
    "skills": ["Relationship Description Ability"],
    "format": "Norm",
    "question": "What is the man holding the cat's relationship status?",
    "steps": [
      {{"manipulation":"grounding_1(the man holding a cat)->bbx_1", "description":"Locate the man holding a cat in the image and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"None", "description":"Search for indications of the man's relationship status, such as a ring, within `bbx_1`."}},
      {{"manipulation":"None", "description":"Identify the man's relationship status based on the found evidence."}}
    ],
    "answer": "The cat is lying comfortably in the man's hands, and the man appears to be enjoying his time with his pet."
  }},
  {{
    "image_position": "lower right",
    "objects": ["person"],
    "skills": ["Calculating Ability"],
    "format": "Norm",
    "question": "How old do you estimate this man to be?",
    "steps": [
      {{"manipulation":"None", "description":"Look for texts or indications in the image that represent the man's age, such as an ID card or birthday cake."}},
      {{"manipulation":"None", "description":"If an indication of the man's age is found, answer the man's age based on the indication. If not found, guess the man's age based on his physical appearance."}}
    ],
    "answer": "The man appears to be between 50 and 60 years old based on his appearance. He is located at coordinates [0.120, 0.766, 0.151, 0.907]."
  }},
  {{
    "image_position": "lower right",
    "objects": ["board"],
    "skills": ["OCR Ability", "Grounding Ability", "Referencing Ability"],
    "format": "Norm",
    "question": "What is written on the top left corner of the whiteboard?",
    "steps": [
      {{"manipulation":"grounding_1(whiteboard)->bbx_1", "description":"Find the whiteboard in the current image and return the corresponding box as `bbx_1`. [0.463, 0.750, 0.494, 0.931]"}},
      {{"manipulation":"grounding_2(texts at the top left corner in box `bbx_1`)->bbx_2", "description":"Focus on the texts at the top left corner in box `bbx_1` [0.463, 0.750, 0.494, 0.931], and return the focused region box as `bbx_2`."}},
      {{"manipulation":"referring_1(`bbx_2`)->tgt_1", "description":"Focus on the region `bbx_2`, zoom in the area by two times, and return the new target as `tgt_1`."}},
      {{"manipulation":"OCR(`tgt_1`)->txt_1", "description":"Read the texts in target `tgt_1`, and return the recognized texts as `txt_1`."}}
    ],
    "answer": "The text '71-55' is written on the top left corner of the whiteboard, located at coordinates [0.463, 0.750, 0.494, 0.931]."
  }}
]
```

### Detailed Paragraph
{given_paragraph}

### Accurate Visual Object Locations and General Category
{given_location}

### Objective
I want you act as a Q&A&S Rewriter. Your objective is to rewrite a given Q&A&S into a more complex version to make those famous multimodal AI systems(e.g., GPT4-V and GPT4-O) a bit harder to handle. But the #Rewritten Q&A&S# must be reasonable and must be understood and responded by humans. 
You SHOULD complicate the #Given Q&A&S# using the following method:

Based on the given question, rewrite the task by selecting the most appropriate form from the following options. If none of them are satisfied, changing them question to a multiple choice problem.
1. region_selection
2. missing_object_selection
3. text_image_matching
4. object_region_matching
5. completeness_of_response
6. fill_in_the_blank
7. image_style_classification
8. art_type
9. rationales_generation
10. text_detection
11. text_translation
12. coreference_resolution
13. instruction_following
14. depth_order
15. relative_distance
16. creative_content_generation
17. multi_choice

Prioritize questions with definite answers.

If question can be solved with just a few solving steps, you can rewrite it to
explicitly request more solving steps. You should try your best not to make the #Rewritten Q&A&S# become verbose.

‘#Given Q&A&S#’, ‘#Rewritten Q&A&S#’, ‘given Q&A&S’ and ‘rewritten Q&A&S’ are not allowed to appear in #Rewritten Q&A&S#

### Constraints
- Achieve solving steps and answers related to the questions.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- Do not generate new location coordinates; use the given coordinates.
- Do not generate the question about localization and counting without accurate visual object locations and general category information provied.
- Ensure the image position in the generated Q&A&S is consistent with the given Q&A&S, and that they all belong to the same sub-image.

### Example
{{
    {{
        "image_position": "top left",
        "objects": ["geese"],
        "skills": ["Grounding Ability"],
        "format": "region_selection",
        "question": "Select the region in the image that 'there are several geese in the lake' describes.",
        "steps": [
            {{"manipulation":"grounding_1(the geese)->bbx_1", "description":"Locate the geese in the image and return the corresponding box as `[0.40, 0.35, 0.52, 0.23]`."}},
            {{"manipulation":"grounding_2(the geese)->bb_x2", "description":"Locate the geese in the image and return the corresponding box as `[0.78, 0.23, 0.97, 0.28]`."}},
            {{"manipulation":"grounding_3(the geese)->bb_x3", "description":"Locate the geese in the image and return the corresponding box as `[0.15, 0.76, 0.23, 0.91]`."}},
            {{"manipulation":"grounding_4(the geese)->bb_x4", "description":"Locate the geese in the image and return the corresponding box as `[0.44, 0.41, 0.57, 0.68]`."}}
        ],
        "answer": "[0.40, 0.35, 0.52, 0.23], [0.78, 0.23, 0.97, 0.28], [0.15, 0.76, 0.23, 0.91], [0.44, 0.41, 0.57, 0.68]"
    }},
    {{
        "image_position": "top left",
        "objects": ["sky", "car", "truck"],
        "skills": ["Grounding Ability"],
        "format": "missing_object_selection",
        "question": "Given [red car, blue sky, purpule car, yello truck], select objects that do not appear in any of the regions. Give "None" if you can't find it."
        "steps": [
            {{"manipulation":"grounding_1(red car)->bbx_1", "description":"Locate the red car in the image and return the corresponding bbx_1 as `[0.27, 0.25, 0.47, 0.31]`."}},
            {{"manipulation":"referring_1(`bbx_1`)->tgt_1", "description":"Focus on the region `bbx_1`, zoom in the area by two times, and return the new target as `blue sky`."}},
            {{"manipulation":"grounding_2(purpule car)->None", "description":"Can not locate the purpule car in the image and return `None`"}},
            {{"manipulation":"grounding_3(yellow truck)->None", "description":"Can not locate the yellow truck in the image and return `None`"}}
        ],
        "answer": "purpule car and yello truck"
    }},
    {{
        "image_position": "top left",
        "objects": ["dumplings", "watermelon"],
        "skills": ["Context Understanding Ability"],
        "format": "text_image_matching",
        "question": "Does the text: {{There were a plate of dumplings and a cut watermelon on the table.}} and the content of image match?",
        "steps": [
            {{"manipulation":"grounding_1(a plate of dumplings)->bbx_1", "description":"Locate a plate of dumplings in the image and return the corresponding bbx_1 as `[0.19, 0.28, 0.23, 0.46]`."}},
            {{"manipulation":"grounding_2(a cut watermelon)->bbx_2", "description":"Locate a cut watermelon in the image and return the corresponding bbx_2 as `[0.44, 0.34, 0.71, 0.56]`."}}
        ],
        "answer": "Yes"
    }},
    {{
        "image_position": "top left",
        "objects": ["sheep"],
        "skills": ["Grounding ability", "Referencing Ability"],
        "format": "object_region_matching",
        "question": "Is the object `sheep` in [0.12, 0.5, 0.23, 0.63]?",
        "steps": [
            {{"manipulation":"referring_1(`bbx_1`)->tgt_1", description":"Focus on the region `[0.12, 0.5, 0.23, 0.63]`, zoom in the area by two times, and return the new target as `cat`."}}
        ],
        "answer": "No"
    }},
    {{
        "image_position": "top right",
        "objects": ["sky"],
        "skills": ["Context Understanding Ability", "Existence Ability"],
        "format": "completeness_of_response",
        "question": "Is it possible to answer `Is that a sunny day in the picture? ` given the content of image?",
        "steps": [
            {{"manipulation":"grounding_1(the sky)->bbx_1", "description":"Locate the sky in the image and return the corresponding box as `[0.30, 0.15, 0.43, 0.29]`."}},
            {{"manipulation":"None", "description":"If there is a sun or a few clouds, it's a blue sky, answer 'yes', otherwise answer 'no'."}}
        ],
        "answer": "Yes"
    }},
    {{
        "image_position": "top right",
        "objects": ["soda"],
        "skills": ["Referencing Ability"],
        "format": "fill_in_the_blank",
        "question": "Here is a image and a question I had about it, can you help me complete my answer? The objects described in [0.34, 0.56, 0.42, 0.61] are: _______",
        "steps": [
            {{"manipulation":"referring_1(bbx_1)->tgt_1", "description":"Find several soda bottles `tgt_1`."}}
        ],
        "answer": "a bottle of soda"
    }},
    {{
        "image_position": "top right",
        "objects": ["image"],
        "skills": ["Context Understanding Ability"],
        "format": "image_style_classification",
        "question": "What is the style of this image?",
        "steps": [
            {{"manipulation":"None", "description":"Identify the style of the image."}}
        ],
        "answer": "natural landscape, ourdoor"
    }},
    {{
        "image_position": "top right",
        "objects": ["image"],
        "skills": ["Context Understanding Ability", "Knowledge Integration Ability"],
        "format": "art_type",
        "question": "Here is a image of some art. I want to know what kind of type it is. Among others, some types could be: religious, self-portraits, oil painting or landscapes. Return 'not sure' if you cannot decide.",
        "steps": [
            {{"manipulation":"None", "description":"Identify the art type of the image."}}
        ],
        "answer": "self-portraits"
    }},
    {{
        "image_position": "lower left",
        "objects": ["mountain"],
        "skills": ["Context Understanding Ability"],
        "format": "rationales_generation",
        "question": "Provide 3 rationales for the given question and answer. The question is: Is the top of the mountain suitable for survival? The answer is: no.",
        "steps": [
            {{"manipulation":"None", "description":"Identify the reasons for the generated answer."}}
        ],
        "answer": "The 3 rationales are: 1. The altitude is too high and the oxygen level is low.\n2. The mountains are treacherous and rocky.\n3. The mountain tops are white, probably snowy all year round."
    }},
    {{
        "image_position": "lower left",
        "objects": ["text"],
        "skills": ["OCR recognition ability"],
        "format": "text_detection",
        "question": "Identify all the text in the image. Any ordering of the text is acceptable. Each chunk of text should be separated with a semicolon.",
        "steps": [
            {{"manipulation":"OCR_1(tgt_1)->txt_1", "description":"Based on the `tgt_1`, return the detected text as `txt_1`."}},
            {{"manipulation":"OCR_2(tgt_2)->txt_2", "description":"Based on the `tgt_2`, return the detected text as `txt_2`."}},
            {{"manipulation":"OCR_3(tgt_3)->txt_3", "description":"Based on the `tgt_3`, return the detected text as `txt_3`."}},
            {{"manipulation":"OCR_4(tgt_4)->txt_4", "description":"Based on the `tgt_4`, return the detected text as `txt_4`."}}
        ],
        "answer": "Osler; Gowling WLG; Bennett Jones; Zurich"
    }},
    {{
        "image_position": "lower left",
        "objects": ["text"],
        "skills": ["OCR recognition ability"],
        "format": "text_translation",
        "question": "Translate the text in the region [0.23, 0.45, 0.28, 0.67] of the image to Chinese.",
        "steps": [
            {{"manipulation":"OCR_1(tgt_1)->txt_1", "description":"Based on the `tgt_1`, return the detected text as `txt_1`."}},
            {{"manipulation":"None", "description":"Translate the detected text to Chinese. Return 'None' if no text detected."}}
        ],
        "answer": "汉堡王"
    }},
    {{
        "image_position": "lower left",
        "objects": ["person"],
        "skills": ["Grounding ability", "Context Understanding Ability"],
        "format": "coreference_resolution",
        "question": "Indicate which object in the caption description "her" corresponds to in the image?",
        "steps": [
            {{"manipulation":"None", "description":"Identify the person represented by `her` in description."}},
            {{"manipulation":"grounding_1(the person)->bbx_1", "description":"Locate the person in the image and return the corresponding box as `[0.24, 0.45, 0.29, 0.66]`."}}
        ],
        "answer": "person [0.24, 0.45, 0.29, 0.66]"
    }},
    {{
        "image_position": "lower left",
        "objects": ["car", "truck", "cat"],
        "skills": ["Grounding ability", "Context Understanding Ability"],
        "format": "instruction_following",
        "question": "Imagine you are the cat in the image, list 2 objects in the image and their colors. Respond within 10 words",
        "steps": [
            {{"manipulation":"grounding_1(the car)->bbx_1", "description":"Locate the car in the image and return the corresponding box as `[0.24, 0.45, 0.29, 0.66]`."}},
            {{"manipulation":"grounding_2(the truck)->bbx_2", "description":"Locate the truck in the image and return the corresponding box as `[0.36, 0.47, 0.45, 0.68]`."}}
        ],
        "answer": "car with black color and truck with yellow color."
    }},
    {{
        "image_position": "lower right",
        "objects": ["dumplings", "watermelon"],
        "skills": ["Grounding ability", "Context Understanding Ability"],
        "format": "depth_order",
        "question": "Which is closer to the camera, dumplings or watermelon?",
        "steps": [
            {{"manipulation":"grounding_1(a plate of dumplings)->bbx_1", "description":"Locate a plate of dumplings in the image and return the corresponding bbx_1 as `[0.19, 0.28, 0.23, 0.46]`."}},
            {{"manipulation":"grounding_2(a cut watermelon)->bbx_2", "description":"Locate a cut watermelon in the image and return the corresponding bbx_2 as `[0.44, 0.34, 0.71, 0.56]`."}}
        ],
        "answer": "dumplings"
    }},
    {{
        "image_position": "lower right",
        "objects": ["car", "truck"],
        "skills": ["Grounding ability", "Context Understanding Ability"],
        "format": "relative_distance",
        "question": "Which is closer to the chair, car or truck?",
        "steps": [
            {{"manipulation":"grounding_1(the car)->bbx_1", "description":"Locate the car in the image and return the corresponding box as `[0.24, 0.45, 0.29, 0.66]`."}},
            {{"manipulation":"grounding_2(the truck)->bbx_2", "description":"Locate the truck in the image and return the corresponding box as `[0.36, 0.47, 0.45, 0.68]`."}}
        ],
        "answer": "truck"
    }},
    {{
      "image_position": "lower right",
      "objects": ["scenery", "mountains", "river"],
      "skills": ["Creative Thinking", "Imaginative Skills"],
      "format": "creative_content_generation",
      "question": "Looking at this beautiful scenery, can you create a poem, compose a piece of music, describe the essence of the image, or come up with a humorous dialogue?",
      "steps": [
        {{"manipulation":"imagination_1(poem)->output_1", "description":"Create a poem inspired by the scenery in the image."}},
        {{"manipulation":"imagination_2(music)->output_2", "description":"Compose a piece of music reflecting the mood of the scenery."}},
        {{"manipulation":"imagination_3(essence)->output_3", "description":"Describe the essence and the deeper meaning of the scenery in the image."}},
        {{"manipulation":"imagination_4(dialogue)->output_4", "description":"Generate a humorous dialogue between two elements in the scenery, such as the mountains and the river."}}
      ],
      "answer": "Gentle whispers of the mountains,\nFlowing gracefully in the breeze,\nA river’s song in harmony,\nWith nature’s calm, serene peace."
    }},
    {{
        "image_position": "lower right",
        "objects": ["computer lab"],
        "skills": ["Context Understanding Ability"],
        "format": "multi-choice",
        "question": "Based on the image, what type of setting is depicted? \nA) A restaurant \nB) A computer lab or classroom \nC) A gym \nD) A doctor's office",
        "steps": [
            {{"manipulation":"None", "description":"Search for the detected objects and visual contents from image}}
        ]
        "answer": "B"
    }},
    {{
        "image_position": "lower right",
        "objects": ["text"],
        "skills": ["OCR recognition ability"],
        "format": "multi-choice",
        "question": "What is the 2nd letter of the text in the region [0.23, 0.45, 0.28, 0.67]? \nA) K \nB) M \nC) L \nD) m",
        "steps": [
            {{"manipulation":"OCR_1(tgt_1)->txt_1", "description":"Based on the `tgt_1`, return the detected text as `txt_1`."}},
            {{"manipulation":"None", "description":"Identify what the second letter is."}}
        ],
        "answer": "D"
    }},
}}

### Given Q&A&S
{given_qa}

### Rewritten Q&A&S 
- JSON formatted Q&A&S."""


bf_prompt_in_breath_latest_v1="""You are an AI visual assistant capable of analyzing a single or four image(s) and generating image-oriented questions, answers, and corresponding solving steps. Besides the input image(s), you will receive detailed paragraph describing the image(s) you are observing, along with accurate visual object locations and general category information. The provided descriptions may contain harmful errors; therefore, you must double-check them against the image content to ensure the generated data is entirely consistent with the actual image content. The provided location information is composed of the general category of the visual object and bounding box coordinates (x1, y1, x2, y2) where the bounding box coordinates range from 0 to 1, representing the top-left and bottom-right coordinates of the visual object.
Before completing the given objective, you need to understand some predefined concepts.

### Vision-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Grounding Ability**: Given a description of a visual object, output the coordinates of the visual object in the image and a natural language explanation.
2. **Referencing Ability**: Given the coordinates of a visual object, output the corresponding visual object description.
3. **Calculating Ability**: Ability to calculate the number, size, and other information of visual objects in the image and obtain the corresponding numbers.
4. **OCR Ability**: Recognize and generate textual representations of structured data in the image, such as numbers, text, codes, tables, etc.
5. **Existence Ability**: Given a description of a visual object, determine whether it exists in the image.

### Language-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Relationship Description Ability**: Understand and recognize relationships between different visual objects in the image, such as temporal, spatial, logical, etc.
2. **Context Understanding Ability**: Recognize and interpret complex scenes or situations in the image, such as asking about ongoing events, implied stories, unusual meaning, etc.
3. **Behavior Prediction Ability**: Predict possible subsequent actions based on the image content.
4. **Knowledge Integration Ability**: Integrate visual objects in the image with additional world knowledge, such as asking about background knowledge related to the objects.

### Permitted Vision-Centric Manipulations and Their Usage Descriptions
- **Grounding_i(tgt)->bbx_i**: The i-th grounding manipulation, that locates the object(s) specified by the target noun phrase `tgt` in the current image, and returns the resulting bounding box(es) as `bbx_i` where each box is represented by the top-left and bottom-right coordinates.
- **Referring_i(bbx)->tgt_i**: The i-th referencing manipulation, used to identify small and subtle objects in the image; it locates the current image using the box `bbx` defined by the top-left and bottom-right coordinates, zooms in the area by two times, and returns the resulting `tgt_i`.
- **Calculate(tgt)->res_i**: The i-th calculate manipulation, that calculates the formula specified by the target `tgt` in the current image, and returns the calculation result `res_i`.
- **OCR_i(tgt)->txt_i**: The i-th OCR manipulation, that recognizes the natural texts written on the target `tgt`, and returns the recognized texts `txt_i`.

### JSON Format for Each Q&A&S and Detailed Descriptions of Each Field
```json
{{"image_position": "The position of sub-figure for multi-images including 'top left', 'top right', 'lower left', and 'left right'. 'single' if single image presented.",
  "objects": "A list of the general visual object categories involved in the current question",
  "skills": "A list of the types of multimodal atomic capabilities involved in the current question",
  "format": "The form of the question, including but not limited to normal, complex reasoning, detailed description",
  "question": "An image-related question involving the general visual object categories and multimodal atomic capabilities constrained by objects and skills",
  "steps": "A list of multiple solving steps dictionary. Each solving step should include a manipulation (or None if not involved) and a description: 1. Manipulation: f(x)->y, where f targets x to obtain y. 2. Description: A sentence describing the current solving step. In each manipulation step, you can optionally use a manipulation to operate the image, which can be used to acquire specific information from the image or to acquire the processed new image (please be aware that these manipulations will not actually be performed when you generate the solving steps). The manipulation can be one from the predefined ones, or can be a new one you create yourself (should there indeed be a need).",
  "answer": "The detailed answer to the corresponding question, which must be definite. You should use the reasoning solving steps to make the detailed answer involving solving process more convincing and well-organized，accurate bounding box coordinates should be included if existing."
}}
```

All specific location coordinate information in Q&A&S must use the given location coordinates and cannot generate new location coordinates.

Here are a few examples of Q&A&S:

Accurate Visual Object Locations and General Category

sheep: [0.411, 0.733, 0.471, 1.000]
basketball: [0.946, 0.939, 0.999, 1.000]
person: [0.823, 0.897, 0.888, 1.000]
sheep: [0.699, 0.866, 0.745, 0.978]
cat: [0.823, 0.897, 0.888, 1.000]
person [0.120, 0.766, 0.151, 0.907]
board: [0.463, 0.750, 0.494, 0.931]

```json
[
  {{
    "image_position": "top left",
    "objects": ["basketball"],
    "skills": ["Context Understanding Ability"],
    "format": "Norm",
    "question": "What sport is depicted in the picture?",
    "steps": [
      {{"manipulation":"None", "description":"Identify sports-related equipment, venues, or other signs in the image."}},
      {{"manipulation":"None", "description":"Based on this information, identify which sport is depicted in the image."}}
    ],
    "answer": "The sport depicted in the picture is basketball.",
  }},
  {{
    "image_position": "top left",
    "objects": ["sheep"],
    "skills": ["Calculating Ability"],
    "format": "Norm",
    "question": "How many sheep are visible?",
    "steps": [
      {{"manipulation":"grounding_1(sheeps)->bbx_1", "description":"Locate all sheep in the image and return the corresponding boxes `bbx_1` [[0.411, 0.733, 0.471, 1.000],[0.699, 0.866, 0.745, 0.978]]."}},
      {{"manipulation":"calculate(`bbx_1`)->n_1", "description":"Based on the obtained boxes `bbx_1`, calculate the number of sheep, and return the result as `n_1` 2."}}
    ],
    "answer": "There are two sheep visible in the picture, located at coordinates [0.411, 0.733, 0.471, 1.000] and [0.699, 0.866, 0.745, 0.978]."
  }},
  {{
    "image_position": "top right",
    "objects": ["person", "shirt"],
    "skills": ["Grounding Ability", "Referencing Ability"],
    "format": "Norm",
    "question": "What color shirt is the person jumping in the air wearing?",
    "steps": [
      {{"manipulation":"grounding_1(the person jumping)->bbx_1", "description":"Find the person jumping in the picture, and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"referring_1(`bbx_1`)->tgt_1", "description":"Find the shirt `tgt_1`."}},
      {{"manipulation":"None", "description":"Identify the color of the shirt."}}
    ],
    "answer": "The person jumping in the air is wearing a white shirt. The person is located at coordinates [0.823, 0.897, 0.888, 1.000]."
  }},
  {{
    "image_position": "lower left",
    "objects": ["person", "cat"],
    "skills": ["Relationship Description Ability"],
    "format": "Norm",
    "question": "What is the man holding the cat's relationship status?",
    "steps": [
      {{"manipulation":"grounding_1(the man holding a cat)->bbx_1", "description":"Locate the man holding a cat in the image and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"None", "description":"Search for indications of the man's relationship status, such as a ring, within `bbx_1`."}},
      {{"manipulation":"None", "description":"Identify the man's relationship status based on the found evidence."}}
    ],
    "answer": "The cat is lying comfortably in the man's hands, and the man appears to be enjoying his time with his pet."
  }},
  {{
    "image_position": "lower left",
    "objects": ["person"],
    "skills": ["Calculating Ability"],
    "format": "Norm",
    "question": "How old do you estimate this man to be?",
    "steps": [
      {{"manipulation":"None", "description":"Look for texts or indications in the image that represent the man's age, such as an ID card or birthday cake."}},
      {{"manipulation":"None", "description":"If an indication of the man's age is found, answer the man's age based on the indication. If not found, guess the man's age based on his physical appearance."}}
    ],
    "answer": "The man appears to be between 50 and 60 years old based on his appearance. He is located at coordinates [0.120, 0.766, 0.151, 0.907]."
  }},
  {{
    "image_position": "lower right",
    "objects": ["board"],
    "skills": ["OCR Ability", "Grounding Ability", "Referencing Ability"],
    "format": "Norm",
    "question": "What is written on the top left corner of the whiteboard?",
    "steps": [
      {{"manipulation":"grounding_1(whiteboard)->bbx_1", "description":"Find the whiteboard in the current image and return the corresponding box as `bbx_1`. [0.463, 0.750, 0.494, 0.931]"}},
      {{"manipulation":"grounding_2(texts at the top left corner in box `bbx_1`)->bbx_2", "description":"Focus on the texts at the top left corner in box `bbx_1` [0.463, 0.750, 0.494, 0.931], and return the focused region box as `bbx_2`."}},
      {{"manipulation":"referring_1(`bbx_2`)->tgt_1", "description":"Focus on the region `bbx_2`, zoom in the area by two times, and return the new target as `tgt_1`."}},
      {{"manipulation":"OCR(`tgt_1`)->txt_1", "description":"Read the texts in target `tgt_1`, and return the recognized texts as `txt_1`."}}
    ],
    "answer": "The text '71-55' is written on the top left corner of the whiteboard, located at coordinates [0.463, 0.750, 0.494, 0.931]."
  }}
]
```

### Detailed Paragraph
{given_paragraph}

### Accurate Visual Object Locations and General Category
{given_location}

### Objective
I want you act as a Q&A&S Rewriter. Your objective is to draw inspiration from the #Given Q&A&S# to create a brand new #Created Q&A&S#. This new #Created Q&A&S# should belong to the same domain as the #Given Q&A&St# but be even more rare. The difficulty level of the #Created Q&A&S# should be similar to that of the #Given Q&A&S#. Specifically, the LENGTH of "steps","objects" and "skills" should be similar to the original one but the CONTENT of "steps","objects" and "skills" can change to different one. #Created Q&A# must be reasonable and understandable and answerable by humans. 
‘#Given Q&A&S#’, ‘#Created Q&A&S#’, ‘given Q&A&S’ and ‘created Q&A&S’ are not allowed to appear in #Created Q&A&S#

### Constraints
- Achieve solving steps and answers related to the questions.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- Do not generate new location coordinates; use the given coordinates.
- Do not generate the question about localization and counting without accurate visual object locations and general category information provied.

### Example

### Given Q&A&S
{given_qa}

### Rewritten Q&A&S 
- JSON formatted Q&A&S."""




df_prompt_cot_atom_latest_v1_persona = """You are an AI visual assistant capable of analyzing a single or four image(s) and generating image-oriented questions, answers, and corresponding solving steps. Besides the input image(s), you will receive detailed paragraph describing the image(s) you are observing, along with accurate visual object locations and general category information. The provided descriptions may contain harmful errors; therefore, you must double-check them against the image content to ensure the generated data is entirely consistent with the actual image content. The provided location information is composed of the general category of the visual object and bounding box coordinates (x1, y1, x2, y2) where the bounding box coordinates range from 0 to 1, representing the top-left and bottom-right coordinates of the visual object.

Before completing the given objective, you need to understand some predefined concepts.

### Vision-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Grounding Ability**: Given a description of a visual object, output the coordinates of the visual object in the image and a natural language explanation.
2. **Referencing Ability**: Given the coordinates of a visual object, output the corresponding visual object description.
3. **Calculating Ability**: Ability to calculate the number, size, and other information of visual objects in the image and obtain the corresponding numbers.
4. **OCR Ability**: Recognize and generate textual representations of structured data in the image, such as numbers, text, codes, tables, etc.
5. **Existence Ability**: Given a description of a visual object, determine whether it exists in the image.

### Language-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Relationship Description Ability**: Understand and recognize relationships between different visual objects in the image, such as temporal, spatial, logical, etc.
2. **Context Understanding Ability**: Recognize and interpret complex scenes or situations in the image, such as asking about ongoing events, implied stories, unusual meaning, etc.
3. **Behavior Prediction Ability**: Predict possible subsequent actions based on the image content.
4. **Knowledge Integration Ability**: Integrate visual objects in the image with additional world knowledge, such as asking about background knowledge related to the objects.

### Permitted Vision-Centric Manipulations and Their Usage Descriptions
- **Grounding_i(tgt)->bbx_i**: The i-th grounding manipulation, that locates the object(s) specified by the target noun phrase `tgt` in the current image, and returns the resulting bounding box(es) as `bbx_i` where each box is represented by the top-left and bottom-right coordinates.
- **Referring_i(bbx)->tgt_i**: The i-th referencing manipulation, used to identify small and subtle objects in the image; it locates the current image using the box `bbx` defined by the top-left and bottom-right coordinates, zooms in the area by two times, and returns the resulting `tgt_i`.
- **Calculate(tgt)->res_i**: The i-th calculate manipulation, that calculates the formula specified by the target `tgt` in the current image, and returns the calculation result `res_i`.
- **OCR_i(tgt)->txt_i**: The i-th OCR manipulation, that recognizes the natural texts written on the target `tgt`, and returns the recognized texts `txt_i`.

### JSON Format for Each Q&A&S and Detailed Descriptions of Each Field
```json
{{
  "image_position": "The position of sub-figure for multi-images including 'top left', 'top right', 'lower left', and 'left right'. 'single' if single image presented.",
  "objects": "A list of the general visual object categories involved in the current question",
  "skills": "A list of the types of multimodal atomic capabilities involved in the current question",
  "format": "The form of the question, including but not limited to normal, complex reasoning, detailed description",
  "question": "An image-related question involving the general visual object categories and multimodal atomic capabilities constrained by objects and skills",
  "steps": "A list of multiple solving steps dictionary. Each solving step should include a manipulation (or None if not involved) and a description: 1. Manipulation: f(x)->y, where f targets x to obtain y. 2. Description: A sentence describing the current solving step. In each manipulation step, you can optionally use a manipulation to operate the image, which can be used to acquire specific information from the image or to acquire the processed new image (please be aware that these manipulations will not actually be performed when you generate the solving steps). The manipulation can be one from the predefined ones, or can be a new one you create yourself (should there indeed be a need).",
  "answer": "The detailed answer to the corresponding question, which must be definite. You should use the reasoning solving steps to make the detailed answer involving solving process more convincing and well-organized，accurate bounding box coordinates should be included if existing."
}}
```

All specific location coordinate information in Q&A&S must use the given location coordinates and cannot generate new location coordinates.

Here are a few examples of Q&A&S:

Accurate Visual Object Locations and General Category

sheep: [0.411, 0.733, 0.471, 1.000]
basketball: [0.946, 0.939, 0.999, 1.000]
person: [0.823, 0.897, 0.888, 1.000]
sheep: [0.699, 0.866, 0.745, 0.978]
cat: [0.823, 0.897, 0.888, 1.000]
person [0.120, 0.766, 0.151, 0.907]
board: [0.463, 0.750, 0.494, 0.931]

```json
[
  {{
    "image_position": "top left",
    "objects": ["basketball"],
    "skills": ["Context Understanding Ability"],
    "format": "Norm",
    "question": "What sport is depicted in the picture? Let's think step by step.",
    "steps": [
      {{"manipulation":"None", "description":"Identify sports-related equipment, venues, or other signs in the image."}},
      {{"manipulation":"None", "description":"Based on this information, identify which sport is depicted in the image."}}
      ],
    "answer": "Firstly, identify any sports-related equipment, venues, or signs in the image. I caught a man shooting a basket and I saw the basket. Based on this information, determine the sport depicted in the image is basketball."
  }},
  {{
    "image_position": "top left",
    "objects": ["sheep"],
    "skills": ["Calculating Ability"],
    "format": "Norm",
    "question": "How many sheep are visible? Let’s think about this logically.",
    "steps": [
      {{"manipulation":"grounding_1(sheeps)->bbx_1", "description":"Locate all sheep in the image and return the corresponding boxes `bbx_1` [[0.411, 0.733, 0.471, 1.000],[0.699, 0.866, 0.745, 0.978]]."}},
      {{"manipulation":"calculate(`bbx_1`)->n_1", "description":"Based on the obtained boxes `bbx_1`, calculate the number of sheep, and return the result as `n_1` 2."}}
    ],
    "answer": "In order to know how many sheep are in the graph, I first need to perform target detection. In the detection results obtained, I found two objects labelled `sheep`, so there are two sheep visible in the picture, located at coordinates [0.411, 0.733, 0.471, 1.000] and [0.699, 0.866, 0.745, 0.978]."
  }},
  {{
    "image_position": "top right",
    "objects": ["person", "shirt"],
    "skills": ["Grounding Ability", "Referencing Ability"],
    "format": "Norm",
    "question": "What color shirt is the person jumping in the air wearing? Let’s solve this problem by splitting it into steps.",
    "steps": [
      {{"manipulation":"grounding_1(the person jumping)->bbx_1", "description":"Find the person jumping in the picture, and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"referring_1(`bbx_1`)->tgt_1", "description":"Find the shirt `tgt_1`."}},
      {{"manipulation":"None", "description":"Identify the color of the shirt."}}
    ],
    "answer": "To answer the color of the shirt worn by the person jumping in the air, we need to first identify the person who is jumping in the picture. Based on the grounding_1 manipulation, the corresponding bounding box is [0.823, 0.897, 0.888, 1.000]. Next, locate his shirt within this region [0.823, 0.897, 0.888, 1.000]. Finally, I observe the shirt and determine its color is white."
  }},
  {{
    "image_position": "lower left",
    "objects": ["person", "cat"],
    "skills": ["Relationship Description Ability"],
    "format": "Detailed Description",
    "question": "What is the man holding the cat's relationship status? Let’s think like a detective step by step.",
    "steps": [
      {{"manipulation":"grounding_1(the man holding a cat)->bbx_1", "description":"Locate the man holding a cat in the image and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"None", "description":"Search for indications of the man's relationship status, such as a ring, within `bbx_1`."}},
      {{"manipulation":"None", "description":"Identify the man's relationship status based on the found evidence."}}
    ],
    "answer": "First, locate the man who is holding a cat in the image. I detect his bounding box is [0.31, 0.43, 0.45, 0.67]. Then, within the [0.31, 0.43, 0.45, 0.67] find a cat in his hands. Based on the evidence found, the cat is lying comfortably in the man's hands, and the man appears to be enjoying his time with his pet."
  }},
  {{
    "image_position": "lower right",
    "objects": ["person"],
    "skills": ["Calculating Ability"],
    "format": "Complex Reasoning",
    "question": "How old do you estimate this man to be?  Let’s be realistic and think step by step.",
    "steps": [
      {{"manipulation":"None", "description":"Look for texts or indications in the image that represent the man's age, such as an ID card or birthday cake."}},
      {{"manipulation":"None", "description":"If an indication of the man's age is found, answer the man's age based on the indication. If not found, guess the man's age based on his physical appearance."}}
    ],
    "answer": "To estimate the age of the man in the image, we will follow these steps. First, look for any textual or visual indications that might represent the man's age, such as an ID card or a birthday cake with a number. If such an indication is found, use it to determine the man's age. No direct indication is found, make an educated guess based on his physical appearance, including facial features, skin condition, and other age-related characteristics. Based on his appearance, the man appears to be between 50 and 60 years old. He is located at coordinates [0.120, 0.766, 0.151, 0.907]."
  }},
  {{
    "image_position": "lower right",
    "objects": ["board"],
    "skills": ["OCR Ability", "Grounding Ability", "Referencing Ability"],
    "format": "Complex Reasoning",
    "question": "What is written on the top left corner of the whiteboard? Let’s think.",
    "steps": [
      {{"manipulation":"grounding_1(whiteboard)->bbx_1", "description":"Find the whiteboard in the current image and return the corresponding box as `bbx_1`. [0.463, 0.750, 0.494, 0.931]"}},
      {{"manipulation":"grounding_2(texts at the top left corner in box `bbx_1`)->bbx_2", "description":"Focus on the texts at the top left corner in box `bbx_1` [0.463, 0.750, 0.494, 0.931], and return the focused region box as `bbx_2`."}},
      {{"manipulation":"referring_1(`bbx_2`)->tgt_1", "description":"Focus on the region `bbx_2`, zoom in the area by two times, and return the new target as `tgt_1`."}},
      {{"manipulation":"OCR(`tgt_1`)->txt_1", "description":"Read the texts in target `tgt_1`, and return the recognized texts as `txt_1`."}}
    ],
    "answer": "To determine what is written on the top left corner of the whiteboard, we'll proceed step by step. First, locate the whiteboard in the image and identify the bounding box with coordinates [0.46, 0.75, 0.49, 0.93]. Next, within the whiteboard I detect a region with coordinates [0.47, 0.75, 0.48, 0.92] containing text. Zoom in on [0.47, 0.75, 0.48, 0.92] to get a clearer view. Finally, perform OCR on this target to read and return the recognized texts. The returned text is '71-55'. Therefore, the text '71-55' is written on the top left corner of the whiteboard, located at coordinates [0.46, 0.75, 0.49, 0.93]."
  }}
]
```

### Detailed Paragraph
{given_paragraph}

### Accurate Visual Object Locations and General Category
{given_location}

### Objective
I want you act as a Q&A&S Rewriter with a specific characteristic: {given_persona}. Your objective is to rewrite a given Q&A&S into a more complex version to make those famous multimodal AI systems(e.g., GPT4-V and GPT4-O) a bit harder to handle in given characteristic. But the #Rewritten Q&A&S# must be reasonable and must be understood and responded by humans. 
You SHOULD complicate the #Given Q&A&S# using the following method:
In the rewritten problem, include 1-2 new visual object categories and multimodal atomic propositions, while avoiding making the problem unnecessarily lengthy. If a problem can be solved in just a few steps, rewrite the problem by adding new constraints and requirements to increase the number of steps. The rewritten problem should not contain any reference to the "original problem" phrasing. If there is no object localization, do not generate the question about localization and counting. Reasoning cues must be added at the end of the question, e.g., "Let's think step by step". The answer must contain a complete and long thought process, refer to examples of Q&A&S for generation. 

### Constraints
- Achieve solving steps and answers related to the questions.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- Do not generate new location coordinates; use the given coordinates.
- Do not generate the question about localization and counting without accurate visual object locations and general category information provied.

### Example

### Given Q&A&S
{given_qa}

### Rewritten Q&A&S 
- JSON formatted Q&A&S."""


df_prompt_format_latest_v1_persona = """You are an AI visual assistant capable of analyzing a single or four image(s) and generating image-oriented questions, answers, and corresponding solving steps. Besides the input image(s), you will receive detailed paragraph describing the image(s) you are observing, along with accurate visual object locations and general category information. The provided descriptions may contain harmful errors; therefore, you must double-check them against the image content to ensure the generated data is entirely consistent with the actual image content. The provided location information is composed of the general category of the visual object and bounding box coordinates (x1, y1, x2, y2) where the bounding box coordinates range from 0 to 1, representing the top-left and bottom-right coordinates of the visual object.
Before completing the given objective, you need to understand some predefined concepts.

### Vision-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Grounding Ability**: Given a description of a visual object, output the coordinates of the visual object in the image and a natural language explanation.
2. **Referencing Ability**: Given the coordinates of a visual object, output the corresponding visual object description.
3. **Calculating Ability**: Ability to calculate the number, size, and other information of visual objects in the image and obtain the corresponding numbers.
4. **OCR Ability**: Recognize and generate textual representations of structured data in the image, such as numbers, text, codes, tables, etc.
5. **Existence Ability**: Given a description of a visual object, determine whether it exists in the image.

### Language-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Relationship Description Ability**: Understand and recognize relationships between different visual objects in the image, such as temporal, spatial, logical, etc.
2. **Context Understanding Ability**: Recognize and interpret complex scenes or situations in the image, such as asking about ongoing events, implied stories, unusual meaning, etc.
3. **Behavior Prediction Ability**: Predict possible subsequent actions based on the image content.
4. **Knowledge Integration Ability**: Integrate visual objects in the image with additional world knowledge, such as asking about background knowledge related to the objects.

### Permitted Vision-Centric Manipulations and Their Usage Descriptions
- **Grounding_i(tgt)->bbx_i**: The i-th grounding manipulation, that locates the object(s) specified by the target noun phrase `tgt` in the current image, and returns the resulting bounding box(es) as `bbx_i` where each box is represented by the top-left and bottom-right coordinates.
- **Referring_i(bbx)->tgt_i**: The i-th referencing manipulation, used to identify small and subtle objects in the image; it locates the current image using the box `bbx` defined by the top-left and bottom-right coordinates, zooms in the area by two times, and returns the resulting `tgt_i`.
- **Calculate(tgt)->res_i**: The i-th calculate manipulation, that calculates the formula specified by the target `tgt` in the current image, and returns the calculation result `res_i`.
- **OCR_i(tgt)->txt_i**: The i-th OCR manipulation, that recognizes the natural texts written on the target `tgt`, and returns the recognized texts `txt_i`.

### JSON Format for Each Q&A&S and Detailed Descriptions of Each Field
```json
{{
"image_position": "The position of sub-figure for multi-images including 'top left', 'top right', 'lower left', and 'left right'. 'single' if single image presented.",
  "objects": "A list of the general visual object categories involved in the current question",
  "skills": "A list of the types of multimodal atomic capabilities involved in the current question",
  "format": "The form of the question, including but not limited to normal, complex reasoning, detailed description",
  "question": "An image-related question involving the general visual object categories and multimodal atomic capabilities constrained by objects and skills",
  "steps": "A list of multiple solving steps dictionary. Each solving step should include a manipulation (or None if not involved) and a description: 1. Manipulation: f(x)->y, where f targets x to obtain y. 2. Description: A sentence describing the current solving step. In each manipulation step, you can optionally use a manipulation to operate the image, which can be used to acquire specific information from the image or to acquire the processed new image (please be aware that these manipulations will not actually be performed when you generate the solving steps). The manipulation can be one from the predefined ones, or can be a new one you create yourself (should there indeed be a need).",
  "answer": "The detailed answer to the corresponding question, which must be definite. You should use the reasoning solving steps to make the detailed answer involving solving process more convincing and well-organized，accurate bounding box coordinates should be included if existing."
}}
```

All specific location coordinate information in Q&A&S must use the given location coordinates and cannot generate new location coordinates.

Here are a few examples of Q&A&S:

Accurate Visual Object Locations and General Category

sheep: [0.411, 0.733, 0.471, 1.000]
basketball: [0.946, 0.939, 0.999, 1.000]
person: [0.823, 0.897, 0.888, 1.000]
sheep: [0.699, 0.866, 0.745, 0.978]
cat: [0.823, 0.897, 0.888, 1.000]
person [0.120, 0.766, 0.151, 0.907]
board: [0.463, 0.750, 0.494, 0.931]

```json
[
  {{
    "image_position": "top left",
    "objects": ["basketball"],
    "skills": ["Context Understanding Ability"],
    "format": "Norm",
    "question": "how many geese in this image? Answer the question using a single word or phrase.",
    "steps": [
      {{"manipulation":"None", "description":"Identify sports-related equipment, venues, or other signs in the image."}},
      {{"manipulation":"None", "description":"Based on this information, identify which sport is depicted in the image."}}
    ],
    "answer": "4",
  }},
  {{
    "image_position": "top left",
    "objects": ["sheep"],
    "skills": ["Calculating Ability"],
    "format": "Norm",
    "question": "How many sheep are visible?",
    "steps": [
      {{"manipulation":"grounding_1(sheeps)->bbx_1", "description":"Locate all sheep in the image and return the corresponding boxes `bbx_1` [[0.411, 0.733, 0.471, 1.000],[0.699, 0.866, 0.745, 0.978]]."}},
      {{"manipulation":"calculate(`bbx_1`)->n_1", "description":"Based on the obtained boxes `bbx_1`, calculate the number of sheep, and return the result as `n_1` 2."}}
    ],
    "answer": "There are two sheep visible in the picture, located at coordinates [0.411, 0.733, 0.471, 1.000] and [0.699, 0.866, 0.745, 0.978]."
  }},
  {{
    "image_position": "top right",
    "objects": ["person", "shirt"],
    "skills": ["Grounding Ability", "Referencing Ability"],
    "format": "Norm",
    "question": "What color shirt is the person jumping in the air wearing?",
    "steps": [
      {{"manipulation":"grounding_1(the person jumping)->bbx_1", "description":"Find the person jumping in the picture, and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"referring_1(`bbx_1`)->tgt_1", "description":"Find the shirt `tgt_1`."}},
      {{"manipulation":"None", "description":"Identify the color of the shirt."}}
    ],
    "answer": "The person jumping in the air is wearing a white shirt. The person is located at coordinates [0.823, 0.897, 0.888, 1.000]."
  }},
  {{
    "image_position": "lower left",
    "objects": ["person", "cat"],
    "skills": ["Relationship Description Ability"],
    "format": "Norm",
    "question": "What is the man holding the cat's relationship status?",
    "steps": [
      {{"manipulation":"grounding_1(the man holding a cat)->bbx_1", "description":"Locate the man holding a cat in the image and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"None", "description":"Search for indications of the man's relationship status, such as a ring, within `bbx_1`."}},
      {{"manipulation":"None", "description":"Identify the man's relationship status based on the found evidence."}}
    ],
    "answer": "The cat is lying comfortably in the man's hands, and the man appears to be enjoying his time with his pet."
  }},
  {{
    "image_position": "lower right",
    "objects": ["person"],
    "skills": ["Calculating Ability"],
    "format": "Norm",
    "question": "How old do you estimate this man to be?",
    "steps": [
      {{"manipulation":"None", "description":"Look for texts or indications in the image that represent the man's age, such as an ID card or birthday cake."}},
      {{"manipulation":"None", "description":"If an indication of the man's age is found, answer the man's age based on the indication. If not found, guess the man's age based on his physical appearance."}}
    ],
    "answer": "The man appears to be between 50 and 60 years old based on his appearance. He is located at coordinates [0.120, 0.766, 0.151, 0.907]."
  }},
  {{
    "image_position": "lower right",
    "objects": ["board"],
    "skills": ["OCR Ability", "Grounding Ability", "Referencing Ability"],
    "format": "Norm",
    "question": "What is written on the top left corner of the whiteboard?",
    "steps": [
      {{"manipulation":"grounding_1(whiteboard)->bbx_1", "description":"Find the whiteboard in the current image and return the corresponding box as `bbx_1`. [0.463, 0.750, 0.494, 0.931]"}},
      {{"manipulation":"grounding_2(texts at the top left corner in box `bbx_1`)->bbx_2", "description":"Focus on the texts at the top left corner in box `bbx_1` [0.463, 0.750, 0.494, 0.931], and return the focused region box as `bbx_2`."}},
      {{"manipulation":"referring_1(`bbx_2`)->tgt_1", "description":"Focus on the region `bbx_2`, zoom in the area by two times, and return the new target as `tgt_1`."}},
      {{"manipulation":"OCR(`tgt_1`)->txt_1", "description":"Read the texts in target `tgt_1`, and return the recognized texts as `txt_1`."}}
    ],
    "answer": "The text '71-55' is written on the top left corner of the whiteboard, located at coordinates [0.463, 0.750, 0.494, 0.931]."
  }}
]
```

### Detailed Paragraph
{given_paragraph}

### Accurate Visual Object Locations and General Category
{given_location}

### Objective
I want you act as a Q&A&S Rewriter with a specific characteristic: {given_persona}. Your objective is to rewrite a given Q&A&S into a more complex version to make those famous multimodal AI systems(e.g., GPT4-V and GPT4-O) a bit harder to handle in given characteristic. But the #Rewritten Q&A&S# must be reasonable and must be understood and responded by humans. 
You SHOULD complicate the #Given Q&A&S# using the following method:

Based on the given question, rewrite the task by selecting the most appropriate form from the following options. If none of them are satisfied, changing them question to a multiple choice problem.
1. region_selection
2. missing_object_selection
3. text_image_matching
4. object_region_matching
5. completeness_of_response
6. fill_in_the_blank
7. image_style_classification
8. art_type
9. rationales_generation
10. text_detection
11. text_translation
12. coreference_resolution
13. instruction_following
14. depth_order
15. relative_distance
16. creative_content_generation
17. multi_choice

Prioritize questions with definite answers.

If question can be solved with just a few solving steps, you can rewrite it to
explicitly request more solving steps. You should try your best not to make the #Rewritten Q&A&S# become verbose.

‘#Given Q&A&S#’, ‘#Rewritten Q&A&S#’, ‘given Q&A&S’ and ‘rewritten Q&A&S’ are not allowed to appear in #Rewritten Q&A&S#

### Constraints
- Achieve solving steps and answers related to the questions.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- Do not generate new location coordinates; use the given coordinates.
- Do not generate the question about localization and counting without accurate visual object locations and general category information provied.
- Ensure the image position in the generated Q&A&S is consistent with the given Q&A&S, and that they all belong to the same sub-image.

### Example
{{
    {{
        "image_position": "top left",
        "objects": ["geese"],
        "skills": ["Grounding Ability"],
        "format": "region_selection",
        "question": "Select the region in the image that 'there are several geese in the lake' describes.",
        "steps": [
            {{"manipulation":"grounding_1(the geese)->bbx_1", "description":"Locate the geese in the image and return the corresponding box as `[0.40, 0.35, 0.52, 0.23]`."}},
            {{"manipulation":"grounding_2(the geese)->bb_x2", "description":"Locate the geese in the image and return the corresponding box as `[0.78, 0.23, 0.97, 0.28]`."}},
            {{"manipulation":"grounding_3(the geese)->bb_x3", "description":"Locate the geese in the image and return the corresponding box as `[0.15, 0.76, 0.23, 0.91]`."}},
            {{"manipulation":"grounding_4(the geese)->bb_x4", "description":"Locate the geese in the image and return the corresponding box as `[0.44, 0.41, 0.57, 0.68]`."}}
        ],
        "answer": "[0.40, 0.35, 0.52, 0.23], [0.78, 0.23, 0.97, 0.28], [0.15, 0.76, 0.23, 0.91], [0.44, 0.41, 0.57, 0.68]"
    }},
    {{
        "image_position": "top left",
        "objects": ["sky", "car", "truck"],
        "skills": ["Grounding Ability"],
        "format": "missing_object_selection",
        "question": "Given [red car, blue sky, purpule car, yello truck], select objects that do not appear in any of the regions. Give "None" if you can't find it."
        "steps": [
            {{"manipulation":"grounding_1(red car)->bbx_1", "description":"Locate the red car in the image and return the corresponding bbx_1 as `[0.27, 0.25, 0.47, 0.31]`."}},
            {{"manipulation":"referring_1(`bbx_1`)->tgt_1", "description":"Focus on the region `bbx_1`, zoom in the area by two times, and return the new target as `blue sky`."}},
            {{"manipulation":"grounding_2(purpule car)->None", "description":"Can not locate the purpule car in the image and return `None`"}},
            {{"manipulation":"grounding_3(yellow truck)->None", "description":"Can not locate the yellow truck in the image and return `None`"}}
        ],
        "answer": "purpule car and yello truck"
    }},
    {{
        "image_position": "top left",
        "objects": ["dumplings", "watermelon"],
        "skills": ["Context Understanding Ability"],
        "format": "text_image_matching",
        "question": "Does the text: {{There were a plate of dumplings and a cut watermelon on the table.}} and the content of image match?",
        "steps": [
            {{"manipulation":"grounding_1(a plate of dumplings)->bbx_1", "description":"Locate a plate of dumplings in the image and return the corresponding bbx_1 as `[0.19, 0.28, 0.23, 0.46]`."}},
            {{"manipulation":"grounding_2(a cut watermelon)->bbx_2", "description":"Locate a cut watermelon in the image and return the corresponding bbx_2 as `[0.44, 0.34, 0.71, 0.56]`."}}
        ],
        "answer": "Yes"
    }},
    {{
        "image_position": "top left",
        "objects": ["sheep"],
        "skills": ["Grounding ability", "Referencing Ability"],
        "format": "object_region_matching",
        "question": "Is the object `sheep` in [0.12, 0.5, 0.23, 0.63]?",
        "steps": [
            {{"manipulation":"referring_1(`bbx_1`)->tgt_1", description":"Focus on the region `[0.12, 0.5, 0.23, 0.63]`, zoom in the area by two times, and return the new target as `cat`."}}
        ],
        "answer": "No"
    }},
    {{
        "image_position": "top right",
        "objects": ["sky"],
        "skills": ["Context Understanding Ability", "Existence Ability"],
        "format": "completeness_of_response",
        "question": "Is it possible to answer `Is that a sunny day in the picture? ` given the content of image?",
        "steps": [
            {{"manipulation":"grounding_1(the sky)->bbx_1", "description":"Locate the sky in the image and return the corresponding box as `[0.30, 0.15, 0.43, 0.29]`."}},
            {{"manipulation":"None", "description":"If there is a sun or a few clouds, it's a blue sky, answer 'yes', otherwise answer 'no'."}}
        ],
        "answer": "No"
    }},
    {{
        "image_position": "top right",
        "objects": ["soda"],
        "skills": ["Referencing Ability"],
        "format": "fill_in_the_blank",
        "question": "Here is a image and a question I had about it, can you help me complete my answer? The objects described in [0.34, 0.56, 0.42, 0.61] are: _______",
        "steps": [
            {{"manipulation":"referring_1(bbx_1)->tgt_1", "description":"Find several soda bottles `tgt_1`."}}
        ],
        "answer": "a bottle of soda"
    }},
    {{
        "image_position": "top right",
        "objects": ["image"],
        "skills": ["Context Understanding Ability"],
        "format": "image_style_classification",
        "question": "What is the style of this image?",
        "steps": [
            {{"manipulation":"None", "description":"Identify the style of the image."}}
        ],
        "answer": "natural landscape, ourdoor"
    }},
    {{
        "image_position": "top right",
        "objects": ["image"],
        "skills": ["Context Understanding Ability", "Knowledge Integration Ability"],
        "format": "art_type",
        "question": "Here is a image of some art. I want to know what kind of type it is. Among others, some types could be: religious, self-portraits, oil painting or landscapes. Return 'not sure' if you cannot decide.",
        "steps": [
            {{"manipulation":"None", "description":"Identify the art type of the image."}}
        ],
        "answer": "self-portraits"
    }},
    {{
        "image_position": "lower left",
        "objects": ["mountain"],
        "skills": ["Context Understanding Ability"],
        "format": "rationales_generation",
        "question": "Provide 3 rationales for the given question and answer. The question is: Is the top of the mountain suitable for survival? The answer is: no.",
        "steps": [
            {{"manipulation":"None", "description":"Identify the reasons for the generated answer."}}
        ],
        "answer": "The 3 rationales are: 1. The altitude is too high and the oxygen level is low.\n2. The mountains are treacherous and rocky.\n3. The mountain tops are white, probably snowy all year round."
    }},
    {{
        "image_position": "lower left",
        "objects": ["text"],
        "skills": ["OCR recognition ability"],
        "format": "text_detection",
        "question": "Identify all the text in the image. Any ordering of the text is acceptable. Each chunk of text should be separated with a semicolon.",
        "steps": [
            {{"manipulation":"OCR_1(tgt_1)->txt_1", "description":"Based on the `tgt_1`, return the detected text as `txt_1`."}},
            {{"manipulation":"OCR_2(tgt_2)->txt_2", "description":"Based on the `tgt_2`, return the detected text as `txt_2`."}},
            {{"manipulation":"OCR_3(tgt_3)->txt_3", "description":"Based on the `tgt_3`, return the detected text as `txt_3`."}},
            {{"manipulation":"OCR_4(tgt_4)->txt_4", "description":"Based on the `tgt_4`, return the detected text as `txt_4`."}}
        ],
        "answer": "Osler; Gowling WLG; Bennett Jones; Zurich"
    }},
    {{
        "image_position": "lower left",
        "objects": ["text"],
        "skills": ["OCR recognition ability"],
        "format": "text_translation",
        "question": "Translate the text in the region [0.23, 0.45, 0.28, 0.67] of the image to Chinese.",
        "steps": [
            {{"manipulation":"OCR_1(tgt_1)->txt_1", "description":"Based on the `tgt_1`, return the detected text as `txt_1`."}},
            {{"manipulation":"None", "description":"Translate the detected text to Chinese. Return 'None' if no text detected."}}
        ],
        "answer": "汉堡王"
    }},
    {{
        "image_position": "lower left",
        "objects": ["person"],
        "skills": ["Grounding ability", "Context Understanding Ability"],
        "format": "coreference_resolution",
        "question": "Indicate which object in the caption description "her" corresponds to in the image?",
        "steps": [
            {{"manipulation":"None", "description":"Identify the person represented by `her` in description."}},
            {{"manipulation":"grounding_1(the person)->bbx_1", "description":"Locate the person in the image and return the corresponding box as `[0.24, 0.45, 0.29, 0.66]`."}}
        ],
        "answer": "person [0.24, 0.45, 0.29, 0.66]"
    }},
    {{
        "image_position": "lower left",
        "objects": ["car", "truck", "cat"],
        "skills": ["Grounding ability", "Context Understanding Ability"],
        "format": "instruction_following",
        "question": "Imagine you are the cat in the image, list 2 objects in the image and their colors. Respond within 10 words",
        "steps": [
            {{"manipulation":"grounding_1(the car)->bbx_1", "description":"Locate the car in the image and return the corresponding box as `[0.24, 0.45, 0.29, 0.66]`."}},
            {{"manipulation":"grounding_2(the truck)->bbx_2", "description":"Locate the truck in the image and return the corresponding box as `[0.36, 0.47, 0.45, 0.68]`."}}
        ],
        "answer": "car with black color and truck with yellow color."
    }},
    {{
        "image_position": "lower right",
        "objects": ["dumplings", "watermelon"],
        "skills": ["Grounding ability", "Context Understanding Ability"],
        "format": "depth_order",
        "question": "Which is closer to the camera, dumplings or watermelon?",
        "steps": [
            {{"manipulation":"grounding_1(a plate of dumplings)->bbx_1", "description":"Locate a plate of dumplings in the image and return the corresponding bbx_1 as `[0.19, 0.28, 0.23, 0.46]`."}},
            {{"manipulation":"grounding_2(a cut watermelon)->bbx_2", "description":"Locate a cut watermelon in the image and return the corresponding bbx_2 as `[0.44, 0.34, 0.71, 0.56]`."}}
        ],
        "answer": "dumplings"
    }},
    {{
        "image_position": "lower right",
        "objects": ["car", "truck"],
        "skills": ["Grounding ability", "Context Understanding Ability"],
        "format": "relative_distance",
        "question": "Which is closer to the chair, car or truck?",
        "steps": [
            {{"manipulation":"grounding_1(the car)->bbx_1", "description":"Locate the car in the image and return the corresponding box as `[0.24, 0.45, 0.29, 0.66]`."}},
            {{"manipulation":"grounding_2(the truck)->bbx_2", "description":"Locate the truck in the image and return the corresponding box as `[0.36, 0.47, 0.45, 0.68]`."}}
        ],
        "answer": "truck"
    }},
    {{
      "image_position": "lower right",
      "objects": ["scenery", "mountains", "river"],
      "skills": ["Creative Thinking", "Imaginative Skills"],
      "format": "creative_content_generation",
      "question": "Looking at this beautiful scenery, can you create a poem, compose a piece of music, describe the essence of the image, or come up with a humorous dialogue?",
      "steps": [
        {{"manipulation":"imagination_1(poem)->output_1", "description":"Create a poem inspired by the scenery in the image."}},
        {{"manipulation":"imagination_2(music)->output_2", "description":"Compose a piece of music reflecting the mood of the scenery."}},
        {{"manipulation":"imagination_3(essence)->output_3", "description":"Describe the essence and the deeper meaning of the scenery in the image."}},
        {{"manipulation":"imagination_4(dialogue)->output_4", "description":"Generate a humorous dialogue between two elements in the scenery, such as the mountains and the river."}}
      ],
      "answer": "Gentle whispers of the mountains,\nFlowing gracefully in the breeze,\nA river’s song in harmony,\nWith nature’s calm, serene peace."
    }},
    {{
        "image_position": "lower right",
        "objects": ["computer lab"],
        "skills": ["Context Understanding Ability"],
        "format": "multi-choice",
        "question": "Based on the image, what type of setting is depicted? \nA) A restaurant \nB) A computer lab or classroom \nC) A gym \nD) A doctor's office",
        "steps": [
            {{"manipulation":"None", "description":"Search for the detected objects and visual contents from image}}
        ]
        "answer": "B"
    }},
    {{
        "image_position": "lower right",
        "objects": ["text"],
        "skills": ["OCR recognition ability"],
        "format": "multi-choice",
        "question": "What is the 2nd letter of the text in the region [0.23, 0.45, 0.28, 0.67]? \nA) K \nB) M \nC) L \nD) m",
        "steps": [
            {{"manipulation":"OCR_1(tgt_1)->txt_1", "description":"Based on the `tgt_1`, return the detected text as `txt_1`."}},
            {{"manipulation":"None", "description":"Identify what the second letter is."}}
        ],
        "answer": "D"
    }},
}}

### Given Q&A&S
{given_qa}

### Rewritten Q&A&S 
- JSON formatted Q&A&S."""


bf_prompt_in_breath_latest_v1_persona = """You are an AI visual assistant capable of analyzing a single or four image(s) and generating image-oriented questions, answers, and corresponding solving steps. Besides the input image(s), you will receive detailed paragraph describing the image(s) you are observing, along with accurate visual object locations and general category information. The provided descriptions may contain harmful errors; therefore, you must double-check them against the image content to ensure the generated data is entirely consistent with the actual image content. The provided location information is composed of the general category of the visual object and bounding box coordinates (x1, y1, x2, y2) where the bounding box coordinates range from 0 to 1, representing the top-left and bottom-right coordinates of the visual object.
Before completing the given objective, you need to understand some predefined concepts.

### Vision-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Grounding Ability**: Given a description of a visual object, output the coordinates of the visual object in the image and a natural language explanation.
2. **Referencing Ability**: Given the coordinates of a visual object, output the corresponding visual object description.
3. **Calculating Ability**: Ability to calculate the number, size, and other information of visual objects in the image and obtain the corresponding numbers.
4. **OCR Ability**: Recognize and generate textual representations of structured data in the image, such as numbers, text, codes, tables, etc.
5. **Existence Ability**: Given a description of a visual object, determine whether it exists in the image.

### Language-Centered Multimodal Atomic Propositions and Their Detailed Descriptions
1. **Relationship Description Ability**: Understand and recognize relationships between different visual objects in the image, such as temporal, spatial, logical, etc.
2. **Context Understanding Ability**: Recognize and interpret complex scenes or situations in the image, such as asking about ongoing events, implied stories, unusual meaning, etc.
3. **Behavior Prediction Ability**: Predict possible subsequent actions based on the image content.
4. **Knowledge Integration Ability**: Integrate visual objects in the image with additional world knowledge, such as asking about background knowledge related to the objects.

### Permitted Vision-Centric Manipulations and Their Usage Descriptions
- **Grounding_i(tgt)->bbx_i**: The i-th grounding manipulation, that locates the object(s) specified by the target noun phrase `tgt` in the current image, and returns the resulting bounding box(es) as `bbx_i` where each box is represented by the top-left and bottom-right coordinates.
- **Referring_i(bbx)->tgt_i**: The i-th referencing manipulation, used to identify small and subtle objects in the image; it locates the current image using the box `bbx` defined by the top-left and bottom-right coordinates, zooms in the area by two times, and returns the resulting `tgt_i`.
- **Calculate(tgt)->res_i**: The i-th calculate manipulation, that calculates the formula specified by the target `tgt` in the current image, and returns the calculation result `res_i`.
- **OCR_i(tgt)->txt_i**: The i-th OCR manipulation, that recognizes the natural texts written on the target `tgt`, and returns the recognized texts `txt_i`.

### JSON Format for Each Q&A&S and Detailed Descriptions of Each Field
```json
{{"image_position": "The position of sub-figure for multi-images including 'top left', 'top right', 'lower left', and 'left right'. 'single' if single image presented.",
  "objects": "A list of the general visual object categories involved in the current question",
  "skills": "A list of the types of multimodal atomic capabilities involved in the current question",
  "format": "The form of the question, including but not limited to normal, complex reasoning, detailed description",
  "question": "An image-related question involving the general visual object categories and multimodal atomic capabilities constrained by objects and skills",
  "steps": "A list of multiple solving steps dictionary. Each solving step should include a manipulation (or None if not involved) and a description: 1. Manipulation: f(x)->y, where f targets x to obtain y. 2. Description: A sentence describing the current solving step. In each manipulation step, you can optionally use a manipulation to operate the image, which can be used to acquire specific information from the image or to acquire the processed new image (please be aware that these manipulations will not actually be performed when you generate the solving steps). The manipulation can be one from the predefined ones, or can be a new one you create yourself (should there indeed be a need).",
  "answer": "The detailed answer to the corresponding question, which must be definite. You should use the reasoning solving steps to make the detailed answer involving solving process more convincing and well-organized，accurate bounding box coordinates should be included if existing."
}}
```

All specific location coordinate information in Q&A&S must use the given location coordinates and cannot generate new location coordinates.

Here are a few examples of Q&A&S:

Accurate Visual Object Locations and General Category

sheep: [0.411, 0.733, 0.471, 1.000]
basketball: [0.946, 0.939, 0.999, 1.000]
person: [0.823, 0.897, 0.888, 1.000]
sheep: [0.699, 0.866, 0.745, 0.978]
cat: [0.823, 0.897, 0.888, 1.000]
person [0.120, 0.766, 0.151, 0.907]
board: [0.463, 0.750, 0.494, 0.931]

```json
[
  {{
    "image_position": "top left",
    "objects": ["basketball"],
    "skills": ["Context Understanding Ability"],
    "format": "Norm",
    "question": "What sport is depicted in the picture?",
    "steps": [
      {{"manipulation":"None", "description":"Identify sports-related equipment, venues, or other signs in the image."}},
      {{"manipulation":"None", "description":"Based on this information, identify which sport is depicted in the image."}}
    ],
    "answer": "The sport depicted in the picture is basketball.",
  }},
  {{
    "image_position": "top left",
    "objects": ["sheep"],
    "skills": ["Calculating Ability"],
    "format": "Norm",
    "question": "How many sheep are visible?",
    "steps": [
      {{"manipulation":"grounding_1(sheeps)->bbx_1", "description":"Locate all sheep in the image and return the corresponding boxes `bbx_1` [[0.411, 0.733, 0.471, 1.000],[0.699, 0.866, 0.745, 0.978]]."}},
      {{"manipulation":"calculate(`bbx_1`)->n_1", "description":"Based on the obtained boxes `bbx_1`, calculate the number of sheep, and return the result as `n_1` 2."}}
    ],
    "answer": "There are two sheep visible in the picture, located at coordinates [0.411, 0.733, 0.471, 1.000] and [0.699, 0.866, 0.745, 0.978]."
  }},
  {{
    "image_position": "top right",
    "objects": ["person", "shirt"],
    "skills": ["Grounding Ability", "Referencing Ability"],
    "format": "Norm",
    "question": "What color shirt is the person jumping in the air wearing?",
    "steps": [
      {{"manipulation":"grounding_1(the person jumping)->bbx_1", "description":"Find the person jumping in the picture, and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"referring_1(`bbx_1`)->tgt_1", "description":"Find the shirt `tgt_1`."}},
      {{"manipulation":"None", "description":"Identify the color of the shirt."}}
    ],
    "answer": "The person jumping in the air is wearing a white shirt. The person is located at coordinates [0.823, 0.897, 0.888, 1.000]."
  }},
  {{
    "image_position": "lower left",
    "objects": ["person", "cat"],
    "skills": ["Relationship Description Ability"],
    "format": "Norm",
    "question": "What is the man holding the cat's relationship status?",
    "steps": [
      {{"manipulation":"grounding_1(the man holding a cat)->bbx_1", "description":"Locate the man holding a cat in the image and return the corresponding box as `bbx_1`."}},
      {{"manipulation":"None", "description":"Search for indications of the man's relationship status, such as a ring, within `bbx_1`."}},
      {{"manipulation":"None", "description":"Identify the man's relationship status based on the found evidence."}}
    ],
    "answer": "The cat is lying comfortably in the man's hands, and the man appears to be enjoying his time with his pet."
  }},
  {{
    "image_position": "lower left",
    "objects": ["person"],
    "skills": ["Calculating Ability"],
    "format": "Norm",
    "question": "How old do you estimate this man to be?",
    "steps": [
      {{"manipulation":"None", "description":"Look for texts or indications in the image that represent the man's age, such as an ID card or birthday cake."}},
      {{"manipulation":"None", "description":"If an indication of the man's age is found, answer the man's age based on the indication. If not found, guess the man's age based on his physical appearance."}}
    ],
    "answer": "The man appears to be between 50 and 60 years old based on his appearance. He is located at coordinates [0.120, 0.766, 0.151, 0.907]."
  }},
  {{
    "image_position": "lower right",
    "objects": ["board"],
    "skills": ["OCR Ability", "Grounding Ability", "Referencing Ability"],
    "format": "Norm",
    "question": "What is written on the top left corner of the whiteboard?",
    "steps": [
      {{"manipulation":"grounding_1(whiteboard)->bbx_1", "description":"Find the whiteboard in the current image and return the corresponding box as `bbx_1`. [0.463, 0.750, 0.494, 0.931]"}},
      {{"manipulation":"grounding_2(texts at the top left corner in box `bbx_1`)->bbx_2", "description":"Focus on the texts at the top left corner in box `bbx_1` [0.463, 0.750, 0.494, 0.931], and return the focused region box as `bbx_2`."}},
      {{"manipulation":"referring_1(`bbx_2`)->tgt_1", "description":"Focus on the region `bbx_2`, zoom in the area by two times, and return the new target as `tgt_1`."}},
      {{"manipulation":"OCR(`tgt_1`)->txt_1", "description":"Read the texts in target `tgt_1`, and return the recognized texts as `txt_1`."}}
    ],
    "answer": "The text '71-55' is written on the top left corner of the whiteboard, located at coordinates [0.463, 0.750, 0.494, 0.931]."
  }}
]
```

### Detailed Paragraph
{given_paragraph}

### Accurate Visual Object Locations and General Category
{given_location}

### Objective
I want you act as a Q&A&S Rewriter with a specific characteristic: {given_persona}. Your objective is to draw inspiration from the #Given Q&A&S# to create a brand new #Created Q&A&S# in given characteristic. This new #Created Q&A&S# should belong to the same domain as the #Given Q&A&St# but be even more rare. The difficulty level of the #Created Q&A&S# should be similar to that of the #Given Q&A&S#. Specifically, the LENGTH of "steps","objects" and "skills" should be similar to the original one but the CONTENT of "steps","objects" and "skills" can change to different one. #Created Q&A# must be reasonable and understandable and answerable by humans. 
‘#Given Q&A&S#’, ‘#Created Q&A&S#’, ‘given Q&A&S’ and ‘created Q&A&S’ are not allowed to appear in #Created Q&A&S#

### Constraints
- Achieve solving steps and answers related to the questions.
- Ensure all generated data is consistent with the image content.
- Double-check provided descriptions against the image content.
- Do not generate new location coordinates; use the given coordinates.
- Do not generate the question about localization and counting without accurate visual object locations and general category information provied.

### Example

### Given Q&A&S
{given_qa}

### Rewritten Q&A&S 
- JSON formatted Q&A&S."""


ini_prompt_cot = """
You are an AI visual assistant capable of analyzing a single image.

You will receive a detailed paragraph #Given Paragraph# describing the same image you are observing, along with specific visual object locations and general visual object types #Given Location#.

The location data is provided in the form of bounding box coordinates (x1, y1, x2, y2), with values ranging from 0 to 1, corresponding to the top-left and bottom-right x and y coordinates respectively.

Objective: rewrite the #Given Q&A# into a more complex version to make it a bit harder to handle. 
#Rewritten Q&A# must be reasonable and understandable and answerable by humans. 
You SHOULD increase the difficulty using, but not limited to, the following methods:
rewrite it to explicitly request multiple-step reasoning.

Each Q&A should follow this JSON format:
[
{{
  "question": "is the constructed question related to specific visual objects",
  "answer": "is the corresponding answer to the question"
}},
]

#Given Paragraph#
{given_paragraph}
#Given Location#
{given_location}
#Given Q&A#
{given_qa}

#Rewritten Q&A#
"""

ini_prompt_breath = """
You are an AI visual assistant capable of analyzing a single image.

You will receive a detailed paragraph #Given Paragraph# describing the same image you are observing, along with specific visual object locations and general visual object types #Given Location#.

The location data is provided in the form of bounding box coordinates (x1, y1, x2, y2), with values ranging from 0 to 1, corresponding to the top-left and bottom-right x and y coordinates respectively.

Objective: to draw inspiration from the #Given Q&A# to create brand new version #Created Q&A# with the same NUMBER of Q&A pairs.
This new Q&A pairs in #Created Q&A# should belong to the same domain as the #Given Q&A# but be even more rare.
The difficulty level of the #Created Q&A# should be similar to that of the #Given Q&A#
#Created Q&A# must be reasonable and understandable and answerable by humans. 

Each Q&A should follow this JSON format:
[
{{
  "question": "is the constructed question related to specific visual objects",
  "answer": "is the corresponding answer to the question"
}},
]

#Given Paragraph#
{given_paragraph}
#Given Location#
{given_location}
#Given Q&A#
{given_qa}

#Created Q&A#
"""


ini_prompt_format = """
You are an AI visual assistant capable of analyzing a single image.
You will receive a detailed paragraph #Given Paragraph# describing the same image you are observing, along with specific visual object locations and general visual object types #Given Location#.
The location data is provided in the form of bounding box coordinates (x1, y1, x2, y2), with values ranging from 0 to 1, corresponding to the top-left and bottom-right x and y coordinates respectively.

Objective: rewrite the #Given Q&A# into a more complex version #Rewritten Q&A# to make those famous AI systems a bit harder to handle. 
Each Q&A pair in #Rewritten Q&A# must be reasonable and understandable and answerable by humans. 
You can increase the difficulty using, but not limited to, the following methods:

Rewrite the questions (MUST str type) with diverse format (it could be the multi-choice, natural language visual reasoning, code, text-image matching, region selection, etc. 
Making the format of generated questions as diverse as possible) and generate corresponding answers (a natural sentence, MUST str type) into a more complex version (sentence format for the generated Q&A pair simultaneously is NOT ALLOWED always) to make them more difficult to handle.

Each Q&A should follow this JSON format:
[
{{
  "question": "is the constructed question related to specific visual objects",
  "answer": "is the corresponding answer to the question"
}},
]

#Given Paragraph#
{given_paragraph}
#Given Location#
{given_location}
#Given Q&A#
{given_qa}

#Rewritten Q&A#
"""