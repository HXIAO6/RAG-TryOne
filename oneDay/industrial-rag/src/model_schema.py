from pydantic import BaseModel


class Answer(BaseModel):
    answer: str


data = {
    "answer": "10MPa"
}

result = Answer(**data)

print(result.answer)