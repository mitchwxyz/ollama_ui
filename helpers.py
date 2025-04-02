def replace_reasoning_tags(input_string):
    # Dictionary mapping old to new values
    replace_dict = {
        "<think>": "<details><summary>thinking...</summary>",
        "</think>": "</details>",
        "<reasoning>": "<details><summary>reasoning...</summary>",
        "</reasoning>": "</details>",
    }

    # Iterate over the dictionary and replace numbers in the string
    for old, new in replace_dict.items():
        input_string = input_string.replace(old, new)

    return input_string
