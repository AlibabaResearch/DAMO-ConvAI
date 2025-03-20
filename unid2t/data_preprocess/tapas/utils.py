import sys
import os


def convert_interaction_to_dict(interaction):

    data = dict()
    data_id = interaction.id
    assert type(data_id) == str
    table = interaction.table
    questions = interaction.questions

    data['id'] = data_id
    data['questions'] = dict()

    # question parse
    for question in questions:
        question_id = question.id

        question_original_text = question.original_text
        assert question_id not in data['questions'].keys()
        data['questions'][question_id] = str(question_original_text)

    # table parse
    table_dict = dict()
    table_id = table.table_id
    document_title = table.document_title
    table_caption = table.caption
    document_url = table.document_url
    alternative_document_urls = table.alternative_document_urls
    alternative_table_ids = table.alternative_table_ids
    context_heading = table.context_heading
    table_dict['id'] = str(table_id)
    table_dict['document_title'] = str(document_title)
    table_dict['caption'] = str(table_caption)
    table_dict['document_url'] = str(document_url)
    assert type(alternative_document_urls) == list
    assert type(alternative_table_ids) == list
    table_dict['alternative_document_urls'] = alternative_document_urls
    table_dict['alternative_table_ids'] = alternative_table_ids
    table_dict['context_heading'] = str(context_heading)


    table_dict['column'] = []
    table_dict['row'] = []
    columns = table.columns  # column
    # n_column = len(columns)
    for column in columns:
        table_dict['column'].append(str(column.text))

    rows = table.rows
    for row in rows:
        text_row = []
        for cell in row.cells:
            text_row.append(str(cell.text))

        table_dict['row'].append(text_row)

    data['table'] = table_dict

    return data






