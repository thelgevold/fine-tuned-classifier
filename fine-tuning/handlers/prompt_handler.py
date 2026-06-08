from domain.categories import CATEGORY_OUTPUT_CODES


class PromptHandler:
    @staticmethod
    def create_categorize_query_prompt(question, categories):
        categories_text = "\n".join(
            f"{CATEGORY_OUTPUT_CODES[category]} = {category}" for category in categories
        )

        return f"""
Classify the homeowner question into exactly one label from the list below.
Return only the short label code from the list. Never return the category name, a number, a synonym, an explanation, or any other text.
Choose the best label based on the meaning of the question.

Valid labels:
{categories_text}

Question: {question}
Code:"""
