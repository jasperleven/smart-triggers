# Smart Triggers API

FastAPI-сервис для AI-анализа комментариев:
- триггеры
- тональность
- уверенность

## Запуск локально
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
uvicorn main:app --reload