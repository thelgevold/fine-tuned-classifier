class PromptHandler:
    @staticmethod
    def create_categorize_query_prompt(question, categories, label_mode="code"):
        if label_mode == "category":
            categories_text = "\n".join(f"- {category}" for category in categories)
            return f"""
Classify the homeowner question into exactly one category from the list below.
Return only the category name from the list. Never return a code, a number, a synonym, an explanation, or any other text.
The answer must be exactly one category name from the list.
Choose the best category based on the meaning of the question.

Valid categories:
{categories_text}

Question: {question}
Category:"""

        categories_text = "\n".join(
            f"{code} = {category}" for category, code in categories.items()
        )

        return f"""
Classify the homeowner question into exactly one label from the list below.
Return only the short label code from the list. Never return the category name, a number, a synonym, an explanation, or any other text.
The answer must be exactly one uppercase two-letter code.
Choose the best label based on the meaning of the question.

Valid labels:
{categories_text}

Question: {question}
Code:"""
